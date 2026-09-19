"""
train_unet.py - External training script for SmallUNet (SIH26059, Checkpoint B).

NOT run by the app or by any agent automatically. Run manually (Colab/Kaggle/
local CPU) to train engine.SmallUNet on labelled SAR ice-segmentation patches,
then decide whether the trained weights clear the acceptance bar before they
are ever committed as the active inference path.

Dataset layout expected (create this yourself, e.g. from AI4Arctic/USNIC):
    data/train_patches/images/<name>.png   grayscale SAR crop
    data/train_patches/masks/<name>.png    binary ice mask (0/255), same <name>

Acceptance bar (checked automatically, see main()):
    Dice(SmallUNet) >= 0.60  AND  Dice(SmallUNet) >= Dice(Otsu) + 0.05
    measured on the SAME held-out patches for both models.

Caveat: if data/sar_real.png or data/sar_real.tif exists, engine.resolve_sar_path()
would redirect any bare "data/sar_sample.png"-style call to it — this script
always passes each held-out patch's own explicit path to detect_ice(), so
that redirection does not apply here, but keep it in mind if you repurpose
this loader elsewhere.

Hardware: trains on GPU automatically the moment a CUDA-enabled torch is
installed (DEVICE below picks it up via torch.cuda.is_available(), no script
change needed); this machine's torch build is CPU-only, so it runs on CPU,
using ~85% of logical cores as DataLoader workers (CPU_WORKERS) with pinned,
persistent workers to keep them fed.

Usage:
    python train_unet.py --epochs 30 --holdout 5 --batch-size 4
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
except ImportError as e:
    raise SystemExit(
        "torch is required to run train_unet.py (it is NOT required by the "
        "app itself). Install the CPU wheel: "
        "pip install torch --index-url https://download.pytorch.org/whl/cpu"
    ) from e

from engine import SmallUNet, detect_ice, UNET_WEIGHTS_PATH

# Hardware utilization. DEVICE resolves to "cuda" the moment a CUDA-enabled
# torch is installed -- no other change needed; on this CPU-only build it
# resolves to "cpu" and training uses worker processes instead of a GPU.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True  # autotune conv algorithms for our fixed patch size (cuDNN/GPU only; a harmless no-op on CPU)

# Most of the machine, not all of it, so the OS/UI stays responsive.
CPU_WORKERS = max(1, int((os.cpu_count() or 1) * 0.85))

PATCH_DIR = "data/train_patches"
IMAGES_DIR = os.path.join(PATCH_DIR, "images")
MASKS_DIR = os.path.join(PATCH_DIR, "masks")
REPORT_PATH = "training_report.json"

DICE_BAR = 0.60
DICE_MARGIN_OVER_OTSU = 0.05


def _list_patch_names():
    if not os.path.isdir(IMAGES_DIR) or not os.path.isdir(MASKS_DIR):
        raise SystemExit(
            f"No training data found. Populate:\n"
            f"  {IMAGES_DIR}/<name>.png  (grayscale SAR crop)\n"
            f"  {MASKS_DIR}/<name>.png   (binary ice mask, same <name>)\n"
            f"then re-run this script."
        )
    names = sorted(
        os.path.splitext(os.path.basename(p))[0]
        for p in glob.glob(os.path.join(IMAGES_DIR, "*.png"))
    )
    names = [n for n in names if os.path.exists(os.path.join(MASKS_DIR, n + ".png"))]
    if not names:
        raise SystemExit(
            f"{IMAGES_DIR} and {MASKS_DIR} exist but no matching image/mask "
            f"pairs were found (matched by filename). Populate them with at "
            f"least a handful of labelled patches and re-run."
        )
    return names


def augment_patch(img: np.ndarray, mask: np.ndarray):
    """
    Port of sea-ice-segmentation-u-net.ipynb's augment_image(): random
    horizontal flip (p=0.5), random vertical flip (p=0.5), then a random
    rotation in +-5 degrees (continuous, matching the notebook's
    np.pi/36*uniform(-1,1) range), applied identically to image and mask.
    Uses cv2 (already a dependency) in place of tensorflow_addons.
    """
    if np.random.rand() < 0.5:
        img = np.fliplr(img)
        mask = np.fliplr(mask)
    if np.random.rand() < 0.5:
        img = np.flipud(img)
        mask = np.flipud(mask)

    angle = np.random.uniform(-5.0, 5.0)
    h, w = img.shape[:2]
    rot_mat = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    img = cv2.warpAffine(img, rot_mat, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask = cv2.warpAffine(mask, rot_mat, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return np.ascontiguousarray(img), np.ascontiguousarray(mask)


class PatchDataset(Dataset):
    """Loads (image, mask) pairs as float32 tensors, image in [0,1], mask in {0,1}."""

    def __init__(self, names, augment: bool = False):
        self.names = names
        self.augment = augment

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        img = cv2.imread(os.path.join(IMAGES_DIR, name + ".png"), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(os.path.join(MASKS_DIR, name + ".png"), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            raise RuntimeError(f"Failed to read patch pair for '{name}'")
        if self.augment:
            img, mask = augment_patch(img, mask)
        x = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
        y = torch.from_numpy((mask > 127).astype(np.float32)).unsqueeze(0)
        return x, y


def dice_score(pred_bin: np.ndarray, true_bin: np.ndarray) -> float:
    """Dice = 2*|A∩B| / (|A|+|B|); both arrays are 0/1 (or 0/255, normalized)."""
    pred = (pred_bin > 0).astype(np.float64)
    true = (true_bin > 0).astype(np.float64)
    inter = float((pred * true).sum())
    total = float(pred.sum() + true.sum())
    return 1.0 if total == 0 else (2.0 * inter) / total


def dice_loss(logits, target, eps=1e-6):
    probs = torch.sigmoid(logits)
    inter = (probs * target).sum()
    union = probs.sum() + target.sum()
    return 1.0 - (2.0 * inter + eps) / (union + eps)


def _holdout_val_loss(model, bce, names) -> float:
    """BCE+Dice loss on the held-out set, used only to drive the LR scheduler
    (mirrors the notebook's ReduceLROnPlateau monitoring val_loss) — never
    used for a gradient step, so this doesn't leak into training weights."""
    model.eval()
    losses = []
    with torch.no_grad():
        for name in names:
            img = cv2.imread(os.path.join(IMAGES_DIR, name + ".png"), cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(os.path.join(MASKS_DIR, name + ".png"), cv2.IMREAD_GRAYSCALE)
            x = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(DEVICE, non_blocking=True)
            y = torch.from_numpy((mask > 127).astype(np.float32)).unsqueeze(0).unsqueeze(0).to(DEVICE, non_blocking=True)
            logits = model(x)
            losses.append((bce(logits, y) + dice_loss(logits, y)).item())
    model.train()
    return float(np.mean(losses)) if losses else 0.0


def train(model, loader, holdout_names, epochs, lr=1e-3):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)  # matches notebook's Adam() default lr=0.001
    bce = nn.BCEWithLogitsLoss()
    # Mirrors the notebook's ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=8).
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.8, patience=8)
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for x, y in loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            opt.zero_grad()
            logits = model(x)
            loss = bce(logits, y) + dice_loss(logits, y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        train_loss = total_loss / max(1, len(loader))
        val_loss = _holdout_val_loss(model, bce, holdout_names)
        scheduler.step(val_loss)
        print(f"epoch {epoch + 1}/{epochs}  loss={train_loss:.4f}  val_loss={val_loss:.4f}  device={DEVICE}")


def evaluate_unet(model, names):
    model.eval()
    scores = []
    with torch.no_grad():
        for name in names:
            img = cv2.imread(os.path.join(IMAGES_DIR, name + ".png"), cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(os.path.join(MASKS_DIR, name + ".png"), cv2.IMREAD_GRAYSCALE)
            x = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(DEVICE, non_blocking=True)
            probs = torch.sigmoid(model(x))[0, 0].cpu().numpy()
            pred = (probs > 0.5).astype(np.uint8) * 255
            scores.append(dice_score(pred, mask))
    return float(np.mean(scores)) if scores else 0.0


def evaluate_otsu(names):
    # Reuses engine.detect_ice's actual Otsu path (use_unet=False) rather than
    # reimplementing blur/threshold/morphology separately here.
    scores = []
    for name in names:
        img_path = os.path.join(IMAGES_DIR, name + ".png")
        mask_path = os.path.join(MASKS_DIR, name + ".png")
        true_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        _, pred_mask, _, _ = detect_ice(img_path, use_unet=False)
        scores.append(dice_score(pred_mask, true_mask))
    return float(np.mean(scores)) if scores else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--holdout", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    names = _list_patch_names()
    if len(names) <= args.holdout:
        raise SystemExit(
            f"Only {len(names)} patch(es) found; need more than --holdout "
            f"({args.holdout}) so training has data left after the held-out "
            f"split. Add more patches or lower --holdout."
        )

    holdout_names = names[-args.holdout:]
    train_names = names[:-args.holdout]
    print(f"{len(train_names)} training patches, {len(holdout_names)} held out.")

    print(f"Device: {DEVICE} | CPU workers: {CPU_WORKERS}")
    loader = DataLoader(
        PatchDataset(train_names, augment=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=CPU_WORKERS,
        pin_memory=True,
        persistent_workers=True,
    )
    model = SmallUNet()
    train(model, loader, holdout_names, epochs=args.epochs, lr=args.lr)

    dice_unet = evaluate_unet(model, holdout_names)
    dice_otsu = evaluate_otsu(holdout_names)
    bar_passed = dice_unet >= DICE_BAR and dice_unet >= dice_otsu + DICE_MARGIN_OVER_OTSU

    print(f"\nHeld-out Dice - SmallUNet: {dice_unet:.4f}  Otsu: {dice_otsu:.4f}")
    print(f"Bar: Dice>={DICE_BAR:.2f} AND >=Otsu+{DICE_MARGIN_OVER_OTSU:.2f} "
          f"-> {'PASS' if bar_passed else 'FAIL'}")

    report = {
        "dice_unet": dice_unet,
        "dice_otsu": dice_otsu,
        "bar_passed": bar_passed,
        "holdout_patches": holdout_names,
        "epochs": args.epochs,
    }
    if bar_passed:
        weights_dir = os.path.dirname(UNET_WEIGHTS_PATH)
        if weights_dir:
            os.makedirs(weights_dir, exist_ok=True)
        torch.save(model.state_dict(), UNET_WEIGHTS_PATH)
        report["note"] = f"Bar passed - weights saved to {UNET_WEIGHTS_PATH}."
        print(report["note"])
    else:
        report["note"] = "Bar not met - Otsu remains the active path. Weights NOT saved."
        print(report["note"])

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
