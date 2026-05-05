"""
Shared utilities for SVM-based microbial stain classification.

Provides:
  - Class-name normalisation (multiclass species labels)
  - Binary label parsing (bacteria vs. mould)
  - 49-dimensional feature extraction from segmented image regions
  - Misc helpers (image listing, saturation adjustment, grid-search CSV export)
"""

import csv
import re
from pathlib import Path

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

# ---------------------------------------------------------------------------
# Class name normalisation
# ---------------------------------------------------------------------------

NORMALIZATION_MAP: dict[str, str] = {
    "alternaria-alternata":              "alternaria_alternata",
    "aspergillus-ochraceus":             "aspergillus_ochraceus",
    "talaromyces-purpureogenus":         "talaromyces_purpureogenus",
    "micrococcus-sp":                    "micrococcus_sp",
    "sphingomonas-aquatilis":            "sphingomonas_aquatilis",
    "curtobacterium-flaccumfaciens":     "curtobacterium_flaccumfaciens",
    "rhodococcus-degradans":             "rhodococcus_degradans",
    "epicoccum-layunese":                "epicoccum_layunese",
    "penicilium-mangini":                "penicillium_mangini",
    "penicillium-mangini":               "penicillium_mangini",
    "arthrobacter-sp":                   "arthrobacter_sp",
    "arthrobacter-sp4":                  "arthrobacter_sp",
    "cladosporium-pseudocladosporioides":"cladosporium_pseudocladosporioides",
    "exiguobacterium-indicum":           "exiguobacterium_indicum",
}

KEYWORD_TO_CLASS: dict[str, str] = {
    "alternaria":      "alternaria_alternata",
    "aspergillus":     "aspergillus_ochraceus",
    "talaromyces":     "talaromyces_purpureogenus",
    "micrococcus":     "micrococcus_sp",
    "sphingomonas":    "sphingomonas_aquatilis",
    "curtobacterium":  "curtobacterium_flaccumfaciens",
    "rhodococcus":     "rhodococcus_degradans",
    "epicoccum":       "epicoccum_layunese",
    "penicilium":      "penicillium_mangini",
    "penicillium":     "penicillium_mangini",
    "arthrobacter":    "arthrobacter_sp",
    "cladosporium":    "cladosporium_pseudocladosporioides",
    "exiguobacterium": "exiguobacterium_indicum",
}

BACTERIA_KEYWORDS = [
    "staphylococcus", "bacillus", "bacteria", "bakteria", "baktéria",
    "escherichia", "micrococcus", "serratia", "pseudomonas", "sphingomonas",
    "curtobacterium", "arthrobacter", "rhodococcus", "exiguobacterium",
]

MOLD_KEYWORDS = [
    "alternaria", "aspergillus", "penicillium", "penicilium", "talaromyces",
    "cladosporium", "mucor", "rhizopus", "trichoderma", "fungus", "mold",
    "plesen", "pleseň", "epicoccum",
]


def normalize_class_name(raw: str) -> str | None:
    """Normalise a raw species token from a filename to a canonical label."""
    x = raw.strip().lower()
    for src, tgt in [
        ("á","a"), ("é","e"), ("í","i"), ("ó","o"), ("ú","u"),
        ("ň","n"), ("ľ","l"), ("š","s"), ("č","c"), ("ť","t"),
        ("ž","z"), ("ý","y"),
    ]:
        x = x.replace(src, tgt)
    x = re.sub(r"\s+", " ", x)

    if x in NORMALIZATION_MAP:
        return NORMALIZATION_MAP[x]
    x_dash = x.replace("_", "-").replace(" ", "-")
    if x_dash in NORMALIZATION_MAP:
        return NORMALIZATION_MAP[x_dash]
    for keyword, label in KEYWORD_TO_CLASS.items():
        if keyword in x:
            return label
    return None


# ---------------------------------------------------------------------------
# Label parsing from filename
# ---------------------------------------------------------------------------

def parse_species_from_filename(fname: str) -> str | None:
    """
    Extract the multiclass species label from a filename.

    Expected formats:
        <id>_<type>_<Species-name>_<device>.ext
        D2_<id>__<id>_<type>_<Species-name>_<device>.ext   (cross-device prefix)
    """
    stem = Path(fname).stem
    if "__" in stem:
        stem = stem.split("__", 1)[1]
    parts = stem.split("_")
    if len(parts) < 3:
        print(f"[WARN] Unexpected filename format: {fname}")
        return None
    species = normalize_class_name(parts[2])
    if species is None:
        print(f"[WARN] Unknown class in filename: {fname} → {parts[2]}")
    return species


def parse_binary_label_from_filename(fname: str) -> int | None:
    """Return 0 (bacteria) or 1 (mould), or None if the class cannot be determined."""
    stem = Path(fname).stem
    parts = stem.split("_")
    if len(parts) < 3:
        print(f"[WARN] Unexpected filename format: {fname}")
        return None
    cls = parts[2].strip().lower()
    for kw in BACTERIA_KEYWORDS:
        if kw in cls:
            return 0
    for kw in MOLD_KEYWORDS:
        if kw in cls:
            return 1
    print(f"[WARN] Unknown class in filename: {fname} → {cls}")
    return None


def label_to_name(label: int) -> str:
    return {0: "bacteria", 1: "mould"}.get(int(label), "unknown")


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def extract_id_from_name(name: str) -> str | None:
    """Parse the leading numeric ID from a filename (e.g. '007_a_...' → '7')."""
    match = re.match(r"^0*(\d+)_", Path(name).name)
    if match:
        return str(int(match.group(1)))
    stem = Path(name).stem
    return str(int(stem)) if stem.isdigit() else None


def list_images(folder: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in exts)


def adjust_saturation(bgr: np.ndarray, factor: float = 1.1) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


# ---------------------------------------------------------------------------
# Feature extraction  (49 features)
# ---------------------------------------------------------------------------

def extract_features(bgr: np.ndarray, mask_255: np.ndarray) -> np.ndarray:
    """
    Extract a 49-dimensional feature vector from the masked stain region.

    Breakdown:
      LAB colour stats    3 channels × 5 (mean, std, p10, p50, p90)  = 15
      Grayscale stats     5 stats + Laplacian variance                =  6
      GLCM texture        6 properties × 2 (mean, std)               = 12
      LBP histogram       10 bins                                     = 10
      Shape               area, perimeter, compactness                =  3
      Edge density        Canny mean inside mask                      =  1
      Colour ratios       R/G, G/B                                    =  2
                                                               total  = 49
    """
    mask = mask_255 > 0
    if mask.sum() < 50:
        mask = np.ones(bgr.shape[:2], dtype=bool)

    feats: list[float] = []

    # LAB colour
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    for ch in range(3):
        v = lab[:, :, ch][mask].astype(np.float32)
        feats += [float(v.mean()), float(v.std()), *np.percentile(v, [10, 50, 90]).tolist()]

    # Grayscale
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gv = gray[mask].astype(np.float32)
    feats += [float(gv.mean()), float(gv.std()), *np.percentile(gv, [10, 50, 90]).tolist()]
    feats.append(float(cv2.Laplacian(gray, cv2.CV_32F)[mask].var()))

    # GLCM
    gray_q = (np.where(mask, gray, 0) // 8).astype(np.uint8)
    glcm = graycomatrix(
        gray_q, distances=[1, 2],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=32, symmetric=True, normed=True,
    )
    for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]:
        v = graycoprops(glcm, prop).ravel()
        feats += [float(v.mean()), float(v.std())]

    # LBP
    lbp = local_binary_pattern(gray, 8, 1, method="uniform")
    hist, _ = np.histogram(lbp[mask], bins=np.arange(0, 12), density=True)
    feats += hist.astype(np.float32).tolist()

    # Shape
    contours, _ = cv2.findContours(
        (mask.astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    if contours:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        perim = cv2.arcLength(c, True)
        compact = (perim ** 2) / (4 * np.pi * area) if area > 0 else 0.0
        feats += [float(area), float(perim), float(compact)]
    else:
        feats += [0.0, 0.0, 0.0]

    # Edge density
    feats.append(float(cv2.Canny(gray, 50, 150)[mask].mean()) if mask.sum() > 0 else 0.0)

    # Colour ratios
    b_ch, g_ch, r_ch = cv2.split(bgr)
    r_m = float(r_ch[mask].mean()) if mask.sum() > 0 else 0.0
    g_m = float(g_ch[mask].mean()) if mask.sum() > 0 else 0.0
    b_m = float(b_ch[mask].mean()) if mask.sum() > 0 else 0.0
    feats += [(r_m + 1e-6) / (g_m + 1e-6), (g_m + 1e-6) / (b_m + 1e-6)]

    return np.array(feats, dtype=np.float32)


# ---------------------------------------------------------------------------
# Grid search persistence
# ---------------------------------------------------------------------------

def save_grid_results(grid, out_dir: Path) -> None:
    """Write GridSearchCV results and best parameters to CSV / TXT files."""
    grid_csv = out_dir / "grid_search_results.csv"
    with open(grid_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "mean_test_score", "std_test_score", "param_C", "param_gamma"])
        r = grid.cv_results_
        for i in range(len(r["params"])):
            writer.writerow([
                r["rank_test_score"][i],
                round(r["mean_test_score"][i], 6),
                round(r["std_test_score"][i], 6),
                r["param_clf__C"][i],
                r["param_clf__gamma"][i],
            ])

    best_txt = out_dir / "best_params.txt"
    with open(best_txt, "w", encoding="utf-8") as f:
        f.write(f"Best CV f1_macro : {grid.best_score_:.6f}\n")
        f.write(f"Best params      : {grid.best_params_}\n")

    print(f"[INFO] Grid results → {grid_csv}")
    print(f"[INFO] Best params  → {best_txt}")
