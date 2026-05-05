"""
Segmentation methods for microbial stain isolation on historical paper.

Implements and evaluates four approaches:
  - Otsu thresholding (intensity-based)
  - LAB colour thresholding  ← used in the thesis pipeline
  - Gabor filter + Otsu      (texture-based)

The module can also be run directly to evaluate all three methods against
ground-truth masks and write per-image IoU / Dice scores to a CSV file.

Usage:
    python segmentation.py --images_dir data/images \\
                            --masks_dir  data/masks  \\
                            --out_csv    results/segmentation.csv \\
                            --save_preds
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------

def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float(inter) / float(union + 1e-9)


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    return (2.0 * float(inter)) / (float(pred.sum() + gt.sum()) + 1e-9)


def keep_largest_component(binary_mask: np.ndarray) -> np.ndarray:
    """Retain only the largest connected component (assumed to be the stain)."""
    m = (binary_mask > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num <= 1:
        return (m * 255).astype(np.uint8)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest).astype(np.uint8) * 255


def postprocess_mask(
    binary_mask: np.ndarray,
    k: int = 5,
    it_open: int = 1,
    it_close: int = 2,
) -> np.ndarray:
    """Remove noise (morphological open), fill small holes (close), keep largest component."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = (binary_mask > 0).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  kernel, iterations=it_open)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=it_close)
    return keep_largest_component(m)


def maybe_invert(mask: np.ndarray) -> np.ndarray:
    """Invert mask if the foreground covers more than 60 % of the image (background/foreground swap)."""
    if np.mean(mask == 255) > 0.6:
        return cv2.bitwise_not(mask)
    return mask


def read_gt_mask(mask_path: str) -> np.ndarray:
    """Load a ground-truth mask and binarise to {0, 255}."""
    m = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(mask_path)
    if len(m.shape) == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    return (m > 0).astype(np.uint8) * 255


# ---------------------------------------------------------------------------
# Segmentation methods
# ---------------------------------------------------------------------------

def segment_otsu(bgr: np.ndarray) -> np.ndarray:
    """Intensity-based segmentation: grayscale → Gaussian blur → Otsu threshold."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thr = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return postprocess_mask(maybe_invert(thr))


def segment_color_lab(bgr: np.ndarray, channel: str = "b") -> np.ndarray:
    """
    Colour-based segmentation using a single LAB channel.

    Converts to LAB colour space, applies Gaussian blur, and thresholds
    the selected channel with Otsu's method.

    Args:
        bgr:     Input image in BGR format.
        channel: LAB channel to use – 'L', 'a', or 'b' (default: 'b').
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    ch_map = {"L": 0, "a": 1, "b": 2}
    c = lab[:, :, ch_map[channel]]
    blurred = cv2.GaussianBlur(c, (5, 5), 0)
    _, thr = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return postprocess_mask(maybe_invert(thr))


def _gabor_texture_map(
    gray: np.ndarray,
    ksize: int = 31,
    sigmas: tuple = (3.0, 5.0),
    lambdas: tuple = (8.0, 12.0),
    gammas: tuple = (0.5,),
    thetas: tuple = (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
) -> np.ndarray:
    """Compute a Gabor filter response map (max response over all filter orientations)."""
    img = gray.astype(np.float32) / 255.0
    acc = np.zeros_like(img, dtype=np.float32)
    for sigma in sigmas:
        for lambd in lambdas:
            for gamma in gammas:
                for theta in thetas:
                    kernel = cv2.getGaborKernel(
                        (ksize, ksize), sigma, theta, lambd, gamma, psi=0, ktype=cv2.CV_32F,
                    )
                    resp = cv2.filter2D(img, cv2.CV_32F, kernel)
                    acc = np.maximum(acc, np.abs(resp))
    acc = cv2.normalize(acc, None, 0, 255, cv2.NORM_MINMAX)
    return acc.astype(np.uint8)


def segment_texture_gabor(bgr: np.ndarray) -> np.ndarray:
    """Texture-based segmentation: Gabor filter bank → Otsu threshold."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    tex = _gabor_texture_map(blurred)
    _, thr = cv2.threshold(tex, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return postprocess_mask(maybe_invert(thr))


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def evaluate_dataset(
    images_dir: Path,
    masks_dir: Path,
    out_csv: Path = Path("results/segmentation.csv"),
    save_preds: bool = False,
    preds_dir: Path = Path("results/preds"),
    lab_channel: str = "b",
) -> None:
    """
    Evaluate all three segmentation methods against ground-truth masks.

    For each image, computes IoU and Dice for Otsu, LAB, and Gabor.
    Writes per-image scores to a CSV file and prints mean ± std.
    """
    if save_preds:
        for sub in ("otsu", "lab", "gabor"):
            (preds_dir / sub).mkdir(parents=True, exist_ok=True)

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    img_files = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in exts)
    if not img_files:
        raise RuntimeError(f"No images found in {images_dir}")

    rows = []
    agg: dict[str, list[float]] = {k: [] for k in
                                    ["otsu_iou", "otsu_dice", "lab_iou",
                                     "lab_dice", "gabor_iou", "gabor_dice"]}

    for img_path in img_files:
        gt_path = masks_dir / img_path.name
        if not gt_path.exists():
            candidates = list(masks_dir.glob(img_path.stem + ".*"))
            if not candidates:
                print(f"[WARN] No ground-truth mask for {img_path.name}, skipping.")
                continue
            gt_path = candidates[0]

        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f"[WARN] Cannot read {img_path.name}, skipping.")
            continue

        gt      = read_gt_mask(str(gt_path))
        m_otsu  = segment_otsu(bgr)
        m_lab   = segment_color_lab(bgr, channel=lab_channel)
        m_gabor = segment_texture_gabor(bgr)

        scores = {
            "otsu_iou":  iou(m_otsu,  gt), "otsu_dice":  dice(m_otsu,  gt),
            "lab_iou":   iou(m_lab,   gt), "lab_dice":   dice(m_lab,   gt),
            "gabor_iou": iou(m_gabor, gt), "gabor_dice": dice(m_gabor, gt),
        }
        for k, v in scores.items():
            agg[k].append(v)
        rows.append([img_path.name, *scores.values()])

        if save_preds:
            cv2.imwrite(str(preds_dir / "otsu"  / img_path.name), m_otsu)
            cv2.imwrite(str(preds_dir / "lab"   / img_path.name), m_lab)
            cv2.imwrite(str(preds_dir / "gabor" / img_path.name), m_gabor)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "otsu_iou", "otsu_dice",
                         "lab_iou", "lab_dice", "gabor_iou", "gabor_dice"])
        writer.writerows(rows)

    def ms(key: str):
        v = np.array(agg[key])
        return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0

    print("\n=== Segmentation evaluation (mean ± std) ===")
    for label, iou_key, dice_key in [
        (f"Otsu",           "otsu_iou",  "otsu_dice"),
        (f"LAB({lab_channel})", "lab_iou",   "lab_dice"),
        ("Gabor",           "gabor_iou", "gabor_dice"),
    ]:
        im, is_ = ms(iou_key)
        dm, ds  = ms(dice_key)
        print(f"{label:10s}: IoU {im:.3f} ± {is_:.3f} | Dice {dm:.3f} ± {ds:.3f}")
    print(f"\nResults saved to: {out_csv}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate segmentation methods on a dataset")
    parser.add_argument("--images_dir",  type=Path, default=Path("data/images"))
    parser.add_argument("--masks_dir",   type=Path, default=Path("data/masks"))
    parser.add_argument("--out_csv",     type=Path, default=Path("results/segmentation.csv"))
    parser.add_argument("--preds_dir",   type=Path, default=Path("results/preds"))
    parser.add_argument("--save_preds",  action="store_true",
                        help="Save predicted masks to --preds_dir")
    parser.add_argument("--lab_channel", type=str,  default="b",
                        choices=["L", "a", "b"])
    args = parser.parse_args()
    evaluate_dataset(
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        out_csv=args.out_csv,
        save_preds=args.save_preds,
        preds_dir=args.preds_dir,
        lab_channel=args.lab_channel,
    )
