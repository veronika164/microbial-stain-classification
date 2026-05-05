"""
Baseline comparison without segmentation – cross-validation.

Compares 16-feature vs 49-feature vectors across three classifiers
(Logistic Regression, SVM, Random Forest) using stratified k-fold CV.
Labels are loaded from a CSV file.

CSV format (semicolon-separated):
    id;label
    37_a_alternaria_oppo.jpg;pleseň

Usage:
    python compare_cv.py --images_dir data/images \\
                          --csv_path   data/labels.csv \\
                          --out_dir    results/compare_cv \\
                          --folds      5
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate

from features import build_models, features_16, features_49


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

LABEL_MAP = {"bakteria": 0, "baktéria": 0, "bacteria": 0,
             "plesen": 1, "pleseň": 1, "mould": 1, "mold": 1}


def load_dataset(images_dir: Path, csv_path: Path, feat_fn):
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f, delimiter=";"))

    X, y = [], []
    for row in rows:
        img_id    = str(row["id"]).strip()
        label_str = str(row["label"]).strip().lower()
        if label_str not in LABEL_MAP:
            print(f"[WARN] Unknown label '{label_str}', skipping.")
            continue

        img_path = images_dir / img_id
        if not img_path.exists():
            candidates = list(images_dir.glob(Path(img_id).stem + ".*"))
            if not candidates:
                print(f"[WARN] Image not found: {img_id}")
                continue
            img_path = candidates[0]

        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f"[WARN] Cannot read: {img_path.name}")
            continue

        X.append(feat_fn(bgr))
        y.append(LABEL_MAP[label_str])

    if not X:
        raise RuntimeError(f"No samples loaded from {csv_path}")
    return np.vstack(X), np.array(y, dtype=np.int64)


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def run_cv(X, y, folds: int = 5, seed: int = 42) -> dict:
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scoring = {"acc": "accuracy", "f1": "f1",
               "prec": "precision", "rec": "recall"}
    results = {}
    for name, clf in build_models(seed).items():
        scores = cross_validate(clf, X, y, cv=cv, scoring=scoring)
        results[name] = {
            "accuracy":  (scores["test_acc"].mean(),  scores["test_acc"].std()),
            "f1":        (scores["test_f1"].mean(),   scores["test_f1"].std()),
            "precision": (scores["test_prec"].mean(), scores["test_prec"].std()),
            "recall":    (scores["test_rec"].mean(),  scores["test_rec"].std()),
        }
    return results


def print_results(label: str, results: dict):
    print(f"\n{'='*60}\n  {label}\n{'='*60}")
    for model, metrics in results.items():
        print(f"\n  {model}")
        for metric, (m, s) in metrics.items():
            print(f"    {metric:<12}: {m:.3f} ± {s:.3f}")


def save_results(out_file: Path, label: str, results: dict):
    with open(out_file, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*60}\n  {label}\n{'='*60}\n")
        for model, metrics in results.items():
            f.write(f"\n  {model}\n")
            for metric, (m, s) in metrics.items():
                f.write(f"    {metric:<12}: {m:.3f} ± {s:.3f}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_file = args.out_dir / "results.txt"
    out_file.write_text(
        "Baseline comparison (no segmentation) – cross-validation\n",
        encoding="utf-8",
    )

    for n_feats, feat_fn in [(16, features_16), (49, features_49)]:
        print(f"\n[INFO] Loading dataset – {n_feats} features...")
        X, y = load_dataset(args.images_dir, args.csv_path, feat_fn)
        print(f"       {len(y)} samples | distribution: {np.bincount(y).tolist()}")
        label = f"{n_feats} features (no segmentation) – {args.folds}-fold CV"
        results = run_cv(X, y, folds=args.folds, seed=args.seed)
        print_results(label, results)
        save_results(out_file, label, results)

    print(f"\n[INFO] Results saved to: {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Baseline comparison (no segmentation) – cross-validation"
    )
    parser.add_argument("--images_dir", type=Path, default=Path("data/images"))
    parser.add_argument("--csv_path",   type=Path, default=Path("data/labels.csv"))
    parser.add_argument("--out_dir",    type=Path, default=Path("results/compare_cv"))
    parser.add_argument("--folds",      type=int,  default=5)
    parser.add_argument("--seed",       type=int,  default=42)
    main(parser.parse_args())
