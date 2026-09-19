# PolarNav AI — FINALS COMPLETION PLAN (SIH26059)

Plan written 2026-09-19. Read-only snapshot taken on this machine (system
Python 3.13.14, no venv). On approval, the first action is to copy this file
to the repo as `PLAN.md` (CLAUDE.md lists it as editable and references it).

## Context

Checkpoint A (Otsu-only offline baseline) is done and locked. Checkpoint B
(ML core: SmallUNet + Ridge + KMeans) is wired in `engine.py` but dormant:
torch/scikit-learn are not installed and no weights exist, so every ML path
falls back (Otsu / physics / flat risk). Two problem-statement gaps remain
before finals — Mission A (24-h sea-ice concentration forecast + GIS overlay)
and Mission B (real-time alerts + CPA zones) — plus the training decision,
the drift model, the full gate, repo hygiene, and demo-day rehearsal.

---

## STEP 0 — STATUS SNAPSHOT (measured, read-only)

| Check | Result |
|---|---|
| `engine.py` has `predict_risk_grid` / `cpa_km` / `classify_threat` | **NO / NO / NO** → Missions A and B NOT done |
| `app.py` has `ImageOverlay` / "Live Alerts" | **NO / NO** |
| `models/` dir | **does not exist**; no `unet_weights.pth`, no `drift_model.joblib` |
| `./patches/` img/mask pairs | **0** (dir absent). `train_unet.py` actually reads `data/train_patches/{images,masks}/` — also absent |
| torch installed / CUDA | **torch NOT installed**; sklearn/joblib NOT installed. GPU present: RTX 3050 Laptop 4 GB |
| Installed core | streamlit 1.63.0 (req pins 1.64.0), pandas 2.3.3, numpy 2.5.2 (req pins 2.5.3), folium 0.20.0, cv2 5.0.0, matplotlib 3.11.1 (unlisted), Pillow 12.3.0 |
| Playwright browsers | chromium-1243 + headless shell present; `streamlit.testing.v1.AppTest` imports OK |
| `python engine.py` | **PASS** — active path otsu, direct_crossings=4, risk_reduction=73.7% |
| git | branch `main`, 17 commits, HEAD `fc62295` "docs: sync README with notebook-matched SmallUNet + NSIDC synthetic SAR"; origin = `https://github.com/Lucifer0000007/PolarNav-AI.git`; **no tags**; **uncommitted:** `requirements.txt` (pandas 3.0.5→2.3.3, +pyarrow 18.1.0) |
| `.gitignore` | already excludes `venv/`, `routes.db`, `patches/`, `data/sar_sample.png`, `.claude/`, `CLAUDE.md` (CLAUDE.md is untracked) |
| Tracked junk (`git ls-files` grep venv/routes.db/patches/sar_sample/.pth/.joblib) | **none** |
| `PLAN.md`, `docs/`, `train_drift.py`, `TRAINING_REPORT.md` | all absent |

Existing engine facts that shape the design (reuse, don't reinvent):
- `predict_drift()` (engine.py:473) — physics: `u_total = uc + 0.03*uw`, `disp = u_total*hours*3600`, `dlat = dy/111000`, `dlon = dx/(111000*cos(lat))`.
- `latlon_to_grid()` (engine.py:67) / vectorized copy inside `build_risk_grid()` (engine.py:426-428): `r = size*(1-(lat-LAT_MIN))`, `c = size*(lon-LON_MIN)/lon_span`. Row 0 = north (lat -67), so an ImageOverlay with `origin="upper"` and bounds `[[LAT_MIN,LON_MIN],[LAT_MAX,LON_MAX]]` is correctly oriented.
- `predict_iceberg_drift()` (engine.py:510) already does nearest-wind-point lookup per iceberg — the drift-field builder mirrors this per grid cell.
- `route_metrics()` (engine.py:688) returns 8 keys; A* is fed `risk_grid` by callers (app.py:226, engine.py:880) so "astar routes on combined" is a caller-side change — `astar()` signature untouched.
- `_kmeans_risk_weights()` (engine.py:374) is the KMeans pattern (guarded, never raises) to copy for band edges.
- `KM_PER_CELL = 2.775`; grid spans 111 km (lat) × ~64 km (lon at 67.5°S). Wind CSV: current ~0.1 m/s + 3% of ~4 m/s wind ≈ 0.22 m/s → ~19 km / ~7 cells east in 24 h — the forecast shift will be clearly visible.
- `st.session_state.route_data` (app.py:251) is the plain-data dict the map is rebuilt from every rerun — new outputs (predicted grid, alerts) go there so they survive refresh (F7 pattern).
- SmallUNet param estimate (from layer table, engine.py:262-277): ≈8.6 M params ≈ **34 MB fp32** → would FAIL the <10 MB weights check even if trained. See risk register.

---

## STEP 1 — ORDERED PLAN

Rollback prep (before any edit):
```
git add requirements.txt && git commit -m "chore: pin pandas 2.3.3 + pyarrow 18.1.0 (matches demo machine)"
git tag finals-start
```

### Step A — Missions A + B (MAIN thread, serial, engine.py + app.py only)

**A1 engine.py — 24-h predicted risk grid** (additive, ~60 lines, after `build_risk_grid`)
- `build_drift_field(wind_df, size=GRID) -> Optional[np.ndarray]` shape `(size,size,2)` = per-cell `(u_total, v_total)` m/s via nearest wind point (vectorized: cell-centre lat/lon from `grid_to_latlon`, argmin over the wind table). Returns `None` on missing/empty/bad-column `wind_df`.
- `predict_risk_grid(risk_grid, drift_field, hours=24.0) -> np.ndarray`: per-cell displacement with the same physics as `predict_drift` (physics only — Ridge is per-iceberg, not per-cell); convert to grid shift `dr = -dlat*size`, `dc = dlon*size/lon_span`, round, clamp to edges; `np.maximum.at(pred, (rs,cs), risk)`; return `np.maximum(risk_grid, pred)`. `drift_field is None` → `risk_grid.copy()`. Never raises.
- `kmeans_band_edges(risk_predicted, n_bands=5) -> Tuple[List[float], str]`: KMeans(5) on nonzero cell concentrations → 4 sorted interior edges (% units), tag `"kmeans"`; fallback `[20,40,60,80]`, tag `"fixed"` (sklearn absent / <5 distinct values / any exception). Same guard style as `_kmeans_risk_weights`.
- `route_metrics(risk_grid, path, direct, predicted_grid=None)`: keep all 8 existing keys; add `current_exposure` (= `path_risk_score`), `predicted_exposure` (mean of `predicted_grid` along `path`; equals current when None), `predicted_crossings` (count >5 on predicted grid).
- Self-test: after building `risk_grid`, build field from a small inline wind table → `risk_pred = predict_risk_grid(...)`, `risk_combined = np.maximum(risk_grid, risk_pred)`; A* on `risk_combined`; metrics with `predicted_grid=risk_pred`; `assert m['predicted_exposure'] >= m['current_exposure']`; print both. Keep the scenario check (50–90 %, crossings>0) — it stays computed on the current grid.

**B1 engine.py — CPA + threat + reroute** (additive, ~50 lines, after `route_metrics`)
- `cpa_km(route_path_ll, track_ll) -> float`: route as list of (lat,lon); track = `[curr_ll, pred_ll]` densified to ~25 points; equirectangular distance (`cos(lat)` scaled) in km; min over all pairs. Empty input → `inf`.
- `classify_threat(cpa, predicted_crossings) -> str`: `"HIGH"` if `cpa < 5` or `predicted_crossings > 0`; `"MED"` if `cpa < 10`; else `"LOW"`.
- `suggest_reroute(risk_combined, start, goal, base_path, predicted_grid, risk_weight=50.0) -> Optional[Tuple[List, float]]`: `astar(..., risk_weight*2)`; return `(alt_path, delta_km)` only if alt exists AND alt `predicted_exposure` < base's; else `None`.
- Self-test: print `cpa_km` for the two sample icebergs, the threat class, and the reroute result (or "none").

**A2 + B2 app.py — overlay, legend, alert panel** (~70 lines, no new widgets)
- In the route button handler (app.py:206-261): `drift_field = build_drift_field(wind_df)`, `risk_pred = predict_risk_grid(risk_grid, drift_field)`, `risk_combined = np.maximum(...)`; A* + `suggest_reroute` on `risk_combined`; `route_metrics(..., predicted_grid=risk_pred)`; per-iceberg `cpa_km` → alerts list `[(level, text)]` with the exact PS strings:
  - `HIGH: Iceberg <id> CPA X.X km — alternate route suggested (+Y.Y km)` (or `— no better route within 24-h horizon` when `suggest_reroute` is None)
  - `MED: route passes within X.X km of <id> drift corridor`
  - `LOW: corridor clear for 24 h` only when no HIGH/MED
  - always: `Predicted risk>5 cells on route: N`
  - Store `risk_pred`, `band_edges`, `band_src`, `alerts` in `route_data`. The shown route is never switched — suggestion is text only.
- `_build_map()`: `rgba = _concentration_rgba(risk_pred, band_edges)` — numpy-only ice-blue ramp (5 band colours, alpha 0 below the first edge) → `folium.raster_layers.ImageOverlay(image=rgba, bounds=[[LAT_MIN,LON_MIN],[LAT_MAX,LON_MAX]], opacity=0.45, mercator_project=False)`. folium's own `write_png` base64-inlines the array → in-memory PNG, zero CDN. **No matplotlib** (unlisted dep; folium path is cleaner and already offline-proven).
- Below the map: `st.caption("Forecast horizon: 24 h | overlay = predicted concentration")` + legend caption listing the 5 bands with their edges and `bands: KMeans classes` / `bands: fixed 20 % steps` tag; then `st.markdown("### ⚠ Live Alerts (24-h horizon)")` and the stored alerts replayed via `st.error/st.warning/st.info`. Existing offline note + "Active Path" caption unchanged.
- Strict JSON (F9) keys unchanged.

Files: `engine.py`, `app.py`. Commits: `feat(A): 24-h predicted risk grid + concentration overlay`, `feat(B): CPA alerts + reroute suggestion (display-only)`.
Gate: `python engine.py` green (both exposures printed, assert passes, scenario PASS) → `/qa` GO (F1–F9 + overlay visible in t=2 s/t=8 s screenshots + ≥1 alert line + out-of-order clicks + refresh keeps alerts).

### Step B — Training decision (bg-1 runs; MAIN decides)
- Measured: **0 patches** → the `>=25` branch cannot fire today. Default outcome: **ship Otsu-active, captions already honest** (app.py:157-158 prints "OpenCV Otsu (fallback)").
- If the user supplies ≥25 labelled pairs before finals: bg-1 places them at `data/train_patches/{images,masks}/`, installs torch (GPU wheel only if the user OKs the ~2.5 GB download; else CPU wheel, epochs 30 on 40×40–256×256 patches is minutes), runs `python train_unet.py --epochs 30 --holdout 5`. Bar = `dice_unet >= 0.60 AND >= dice_otsu + 0.05` (already enforced in `train_unet.py:232`; on pass it saves to `models/unet_weights.pth`).
- Size check: ~34 MB expected → weights stay **out of git** (add `models/*.pth` to `.gitignore`), shipped by local copy only; or ship fp16 (~17 MB, still >10 MB). Relaxing the 10 MB bar or shrinking the net is the user's call — flagged, not assumed.
- Files: `TRAINING_REPORT.md` (new, root) — records patch count, decision, Dice numbers or "NOT TRAINED — 0 patches", and the honest-caption statement. Also `training_report.json` if training ran.
- Gate: `TRAINING_REPORT.md` written; if weights shipped, `python engine.py` prints `active path: unet`.

### Step C — Ridge drift model (`train_drift.py`, bg-1)
- No `features.csv` exists → **documented SYNTHETIC**: sample `(uc,vc,uw,vw,hours)` around the wind CSV ranges, target `(dlat,dlon)` from `engine.predict_drift` physics at lat −67.5 + small Gaussian noise; fit `sklearn.linear_model.Ridge`; hold out 20 %, report R². Save `models/drift_model.joblib` (KB-scale, <10 MB). Report line states plainly that the model reproduces the physics formula from synthetic samples — pipeline demonstration, not observed drift.
- engine.py: one-line `DRIFT_MODEL_PATH = "models/drift_model.joblib"` (currently repo-root; aligns with `UNET_WEIGHTS_PATH` and the spec). Self-test: add `print(f"Drift source: {'ridge' if load_drift_model() else 'physics'}")` next to the existing `active path` print.
- Prereq: `pip install scikit-learn==1.9.1 joblib==1.6.0` (+ torch CPU) via `setup_demo.bat` with Wi-Fi ON — this also upgrades streamlit 1.63→1.64 and numpy→2.5.3 per the pins; re-run `python engine.py` and a quick `streamlit run` afterwards (risk register).
- Files: `train_drift.py` (new), `engine.py` (2 lines), `models/drift_model.joblib`, `TRAINING_REPORT.md` (append). Commit: `feat(C): Ridge drift model trained on documented synthetic physics samples`.
- Gate: `python engine.py` prints `active path: otsu|unet` AND `Drift source: ridge`; scenario check still PASS.

### Step D — Full gate
- `/qa` → PASS table F1–F9 + GO. `/security-review` → zero network calls / keys (forbidden-import grep: requests, urllib, socket, genai, openai across engine.py/app.py/train_*.py). Weights <10 MB check (joblib yes; .pth if present — see Step B).
- Gate: GO + clean. No files edited in this step unless the gate finds a defect (fix → re-gate).

### Step E — Repo
- Per-step commits (A, B, C, docs). `README.md` sync only for factual consistency (new engine functions in the tech table, drift model status, `train_drift.py` in the tree). Push: `git push origin main` (and `git push origin finals-start`).
- Gate: `git ls-files | grep -Ei 'venv|routes\.db|patches|sar_sample|\.pth'` → empty (models/*.pth ignored; `drift_model.joblib` intentionally tracked).

### Step F — Demo-day rehearsal (this machine, Wi-Fi OFF)
- Wi-Fi ON: run `setup_demo.bat` once (Step C prereq). Wi-Fi OFF: `demo.bat` → browser opens `localhost:8501`; DevTools Network tab must show only `localhost` requests.
- 9-point manual click checklist:
  1. App loads offline, sidebar shows "Connectivity: OUTAGE SIMULATED".
  2. Click **Simulate Satellite Data Drop** → 3 ✅ receipts + NSIDC context line.
  3. Click **Detect Ice** → SAR + mask images, "Active model: …" caption honest, Ice cells metric.
  4. Click **Compute Risk-Aware Route** with defaults (5,5)→(35,35) → 5 metrics, green A* + red direct.
  5. Blue concentration overlay visible on the map; legend caption shows 5 bands + horizon line.
  6. "⚠ Live Alerts" block shows ≥1 alert + "Predicted risk>5 cells on route: N".
  7. Change goal to (20,20), recompute → metrics and alerts change (F6 live).
  8. Hard-refresh (F5) → route history table persists (F7); map + alerts redraw.
  9. Expand **Strict JSON** → keys status/mode=OFFLINE/route_waypoints/total_distance_km/risk_exposure/drift_predictions/generated_at.
- Gate: all 9 pass with Wi-Fi OFF; screenshot pack saved to `docs/training_evidence/`.

### Step G — Human paperwork (user, not the agent)
- Attach the research PDF (NSIDC Sea Ice Index citation, AI4Arctic/USNIC dataset provenance, notebook reference).
- Build the external deck (PPT is NEVER edited by the agent): slides for PS gaps A/B with the overlay + alert screenshots from `docs/training_evidence/`.
- Rehearsal lines: "Otsu is the shipped default; the U-Net path activates only when trained weights clear Dice ≥0.60 and beat Otsu by 0.05 — today it is honest-off / on." "The Ridge drift model is trained on synthetic physics samples — pipeline proof, not observed data." "Alerts are display-only; the captain retains authority."
- Backup video cue: record the 9-point run once with Wi-Fi OFF; keep on desktop as `polarnav_backup.mp4`.

---

## STEP 2 — EXECUTION MODEL

**Subagent map**
- **MAIN** — the ONLY writer of `engine.py` / `app.py`; serial: A1 → gate → B1 → gate → A2+B2 → gate → commit.
- **bg-1 (training/sandbox)** — `setup_demo.bat` install, `train_drift.py` authoring + run, `train_unet.py` run if patches appear; writes only `train_drift.py`, `models/`, `TRAINING_REPORT.md`, `training_report.json`. Never touches engine.py/app.py.
- **bg-2 (evidence pack)** — `docs/training_evidence/`: engine self-test log, /qa table, Playwright screenshots (t=2 s, t=8 s, overlay, alerts), Wi-Fi-OFF network-tab screenshot, checklist results. Read-only on code.

**Rollback**: `git tag finals-start` before the first edit (after committing the pending `requirements.txt`). One commit per step; any red gate → `git checkout finals-start -- engine.py app.py` for that step only.

**Scope-lock note**: `train_drift.py`, `TRAINING_REPORT.md`, `models/`, `docs/training_evidence/` are not in CLAUDE.md's editable list. This plan treats the user's finals brief as the authorising amendment; a one-line CLAUDE.md addition is recommended (CLAUDE.md is untracked, no git impact).

**Risk register**

| Risk | Mitigation |
|---|---|
| torch missing (now) | Engine already degrades: Otsu active, captions honest. Install via `setup_demo.bat` with Wi-Fi ON; CPU wheel only unless user OKs GPU wheel. |
| 0 patches → no U-Net training | Ship Otsu-active; `TRAINING_REPORT.md` states "NOT TRAINED — 0 labelled patches"; rehearsal line covers it. |
| Trained weights ≈34 MB > 10 MB bar | Keep `.pth` out of git; user decides fp16 / bar relax / smaller net. Not assumed. |
| Dice bar fails | Otsu stays active (train_unet.py already refuses to save). |
| A* on combined grid shifts `risk_reduction_pct` outside 50–90 | Gate catches it; fix = scenario retune (`DEMO_SEED`/default pair) per `/fix-scenario`, never by weakening the check. |
| `setup_demo.bat` bumps streamlit 1.63→1.64 / numpy | Re-run `python engine.py` + smoke `streamlit run`; if broken, pin requirements to installed versions (1-line edits). |
| Playwright browsers missing / flaky | Present today; if they vanish → MANUAL 9-point checklist, mark /qa step 4 "MANUAL PASS", never fail the gate on tooling. |
| Ridge trained on synthetic data | Documented in report + caption; physics fallback always reachable. |
| Overlay orientation flipped | Verified analytically (row 0 = north = LAT_MAX top edge, `origin="upper"`); screenshot check in /qa. |
| Wi-Fi-OFF regression | `tiles=None` kept; overlay is base64 data-URI; network tab checked in Step F. |

**Verification summary**: after every edit `python engine.py` (must pass); Step A `/qa` GO; Step D `/qa` GO + `/security-review` clean; Step E `git ls-files` clean; Step F 9/9 with Wi-Fi OFF.
