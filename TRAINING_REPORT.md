# PolarNav AI — Training Report (finals, 2026-09-19)

This file records what was and was not trained for the SIH26059 finals build,
so every caption in the app can be defended verbatim.

## 1. SmallUNet (SAR ice segmentation) — NOT TRAINED

| Item | Value |
|---|---|
| Labelled patches found (`data/train_patches/images` + `masks`) | **0** |
| Training run | **not executed** (`train_unet.py` refuses to start without patch pairs) |
| Acceptance bar (unchanged) | held-out Dice ≥ 0.60 **and** ≥ Otsu Dice + 0.05 on the same patches |
| Shipped inference path | **OpenCV Otsu** (Gaussian blur → Otsu threshold → morphological opening) |
| App caption shown | "Active model: OpenCV Otsu (fallback)" — honest, automatic |
| `models/unet_weights.pth` | absent; `models/*.pth` is git-ignored |

Why not trained: no labelled SAR ice/no-ice patch pairs were available on the
build machine. The SmallUNet architecture (matched to
`sea-ice-segmentation-u-net.ipynb`) and the full train → evaluate → gate →
save pipeline (`train_unet.py`) are in the repo and exercised by import; the
engine loads weights the moment a file that clears the bar is dropped into
`models/`.

Size note for a future trained model: the notebook-matched network has
≈8.6 M parameters (≈34 MB in fp32, ≈17 MB in fp16), which exceeds the
project's <10 MB weights check. Options when training becomes possible: keep
the `.pth` out of git and ship it by local copy (current `.gitignore`), or
shrink the filter progression. Not decided here.

Rehearsal line: "Otsu is the shipped default; the U-Net path activates only
when trained weights clear Dice ≥ 0.60 and beat Otsu by 0.05 — today it is
honestly off."

## 2. Ridge drift model — TRAINED on documented SYNTHETIC data

| Item | Value |
|---|---|
| Script | `train_drift.py` (not run by the app) |
| Training source | **SYNTHETIC (physics formula at lat -67.5 + Gaussian noise; see module docstring)** |
| Rows / held-out | 5000 / 1000 (20 %) |
| Features → targets | u_current, v_current, u_wind, v_wind, hours → dlat, dlon |
| Model | Pipeline(PolynomialFeatures(degree=2, interaction_only) -> Ridge), alpha = 0.001 |
| Held-out R² (Ridge) | 1.0000 |
| Held-out RMSE (Ridge) | 0.222 km |
| Held-out R² (physics baseline, same rows) | 1.0000 |
| Artifact | `models/drift_model.joblib` — 0.0011 MB (< 10 MB: True) — tracked in git |
| Engine self-test line | `Drift source: ridge (models/drift_model.joblib)` |
| Fallback | physics formula in `engine.predict_drift()` — always reachable, used per row on any model failure |

What this does and does not show: no observed iceberg-displacement dataset
(`data/features.csv`) was available, so the training targets are the
engine's own physics formula (current + 3 % wind, 1° ≈ 111 km, evaluated at
the patch mid-latitude −67.5°) plus Gaussian noise. The Ridge model therefore
**reproduces the physics**; it proves the train → serialize → load → infer →
fallback pipeline, not forecasting skill on real drift. Drop an observed
`data/features.csv` with the columns above and re-run the script to train on
real data — the report and this table then update automatically.

Rehearsal line: "The Ridge drift model is trained on synthetic physics
samples — pipeline proof, not observed data. The formula is the fallback."

## 3. KMeans (no training artifact)

Fitted on the fly, deterministic (`random_state = DEMO_SEED`): iceberg
risk-tier profiling in `build_risk_grid()` and concentration band edges for
the forecast overlay legend in `kmeans_band_edges()`. Self-test shows
`Concentration bands (%): [...] (source: kmeans)` when scikit-learn is
installed and `(source: fixed)` otherwise.

## 4. Hardware-optimized training (train_unet.py, 2026-09-19)

`train_unet.py` now resolves `DEVICE = "cuda" if torch.cuda.is_available() else
"cpu"` and sets `torch.backends.cudnn.benchmark = True`; the DataLoader uses
`num_workers = max(1, int(cpu_count * 0.85))` with `pin_memory=True` and
`persistent_workers=True`, and all tensors move with `non_blocking=True`. On
this machine (12 logical CPUs, CPU-only torch build) that resolves to 10
worker processes on CPU; the same code activates a GPU automatically the
moment a CUDA-enabled torch is installed, no script change needed.

Smoke-tested end-to-end with 8 disposable synthetic patch pairs (not part of
the repo): 2 epochs completed without error under the new device/worker
configuration, the bar correctly rejected the resulting weights (Dice 0.00 vs
Otsu 0.17 on 2 epochs of noise), and no weights were saved — confirming the
hardware changes don't affect the acceptance-bar safety logic. Test artifacts
were deleted; the "0 labelled patches" state in section 1 is unchanged.

## 5. NSIDC extent calibration fix (2026-09-19)

`SEAICE_EXTENT_MAX` was `19.0`; the real south-hemisphere range in
`seaice.csv` is 2.08-20.20 M km². Corrected to `20.5` so no real historical
day clips to exactly 100% synthetic ice coverage. `SEAICE_EXTENT_MIN` (2.0)
already safely bounded the real minimum (2.08) and was left unchanged.

## 6. Environment note (build machine, 2026-09-19)

scikit-learn 1.9.1 and joblib 1.6.0 installed via `setup_demo.bat`. The
torch CPU wheel (2.14.0) **failed to install** with `WinError 206: filename
too long` (Microsoft-Store Python's long site-packages path + torch's deep
license tree; `LongPathsEnabled = 0`). The partial torch tree it left behind
(which raised `OSError` on import) was removed; `import torch` now fails with a
clean `ModuleNotFoundError`, and `engine.py` guards the optional import with
`except Exception` either way, so the demo runs unaffected (Otsu path).
Resolving torch is optional for finals because no U-Net weights exist to load.
