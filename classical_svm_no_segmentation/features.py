"""
Shared feature extraction and classifier definitions for the
no-segmentation baseline experiments.

Two feature vectors are provided:
  features_16  – LAB colour statistics + Laplacian variance
  features_49  – full vector identical to the segmentation-based SVM,
                 but computed over the entire image (no mask applied)
"""

import numpy as np
import cv2
from pathlib import Path
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def features_16(bgr: np.ndarray) -> np.ndarray:
    """
    16-dimensional feature vector (no segmentation):
      LAB colour statistics  3 channels × 5 (mean, std, p10, p50, p90) = 15
      Laplacian variance                                                 =  1
    """
    feats = []
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    for ch in range(3):
        v = lab[:, :, ch].astype(np.float32).ravel()
        feats += [v.mean(), v.std(), *np.percentile(v, [10, 50, 90])]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    feats.append(float(cv2.Laplacian(gray, cv2.CV_32F).var()))
    return np.array(feats, dtype=np.float32)


def features_49(bgr: np.ndarray) -> np.ndarray:
    """
    49-dimensional feature vector (no segmentation).

    Identical to the segmented SVM feature vector in classical_svm/utils.py,
    but the mask covers the entire image instead of the segmented region.

    Breakdown:
      LAB colour stats    3 × 5                = 15
      Grayscale stats     5 + Laplacian var     =  6
      GLCM texture        6 props × 2           = 12
      LBP histogram       10 bins               = 10
      Shape               area, perim, compact  =  3
      Edge density        1                     =  1
      Colour ratios       R/G, G/B              =  2
                                         total  = 49
    """
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
    gray_q = (gray // 8).astype(np.uint8)
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
    hist, _ = np.histogram(lbp[mask], bins=np.arange(0, 11), density=True)
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
    feats.append(float(cv2.Canny(gray, 50, 150)[mask].mean()) / 255.0)

    # Colour ratios
    b_m = float(bgr[:, :, 0][mask].mean())
    g_m = float(bgr[:, :, 1][mask].mean())
    r_m = float(bgr[:, :, 2][mask].mean())
    feats += [(r_m + 1e-6) / (g_m + 1e-6), (g_m + 1e-6) / (b_m + 1e-6)]

    assert len(feats) == 49, f"Expected 49 features, got {len(feats)}"
    return np.array(feats, dtype=np.float32)


# ---------------------------------------------------------------------------
# Classifier suite
# ---------------------------------------------------------------------------

def build_models(seed: int = 42) -> dict:
    return {
        "LogReg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                       random_state=seed)),
        ]),
        "SVM(RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", class_weight="balanced", random_state=seed)),
        ]),
        "RandomForest": RandomForestClassifier(
            n_estimators=500, class_weight="balanced_subsample", random_state=seed,
        ),
    }


# ---------------------------------------------------------------------------
# Label parsing
# ---------------------------------------------------------------------------

MOLD_KEYWORDS     = {"alternaria", "aspergillus", "talaromyces", "epicoccum",
                     "penicilium", "penicillium", "cladosporium"}
BACTERIA_KEYWORDS = {"micrococcus", "sphingomonas", "curtobacterium",
                     "rhodococcus", "arthrobacter", "exiguobacterium"}


def parse_binary_label(fname: str) -> int | None:
    """Return 0 (bacteria) or 1 (mould) from a filename, or None if unknown."""
    stem = Path(fname).stem
    if "__" in stem:
        stem = stem.split("__", 1)[1]
    parts = stem.split("_")
    if len(parts) < 3:
        return None
    raw = parts[2].strip().lower()
    for src, tgt in [("á","a"),("é","e"),("í","i"),("ó","o"),("ú","u"),("ň","n")]:
        raw = raw.replace(src, tgt)
    if any(kw in raw for kw in MOLD_KEYWORDS):
        return 1
    if any(kw in raw for kw in BACTERIA_KEYWORDS):
        return 0
    return None
