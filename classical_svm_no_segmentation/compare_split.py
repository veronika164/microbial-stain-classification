"""
Baseline comparison without segmentation – train / test split.

Compares 16-feature vs 49-feature vectors across three classifiers
(Logistic Regression, SVM, Random Forest) using a fixed train/test split.
Labels are inferred from filenames (no CSV required).

Usage:
    python compare_split.py --train_dir data/train/images \\
                             --test_dir  data/test/images  \\
                             --out_dir   results/compare_split
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from features import build_models, features_16, features_49, parse_binary_label


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(images_dir: Path, feat_fn):
    X, y = [], []
    for img_path in sorted(images_dir.glob("*")):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        label = parse_binary_label(img_path.name)
        if label is None:
            print(f"[WARN] Cannot determine label: {img_path.name}")
            continue
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f"[WARN] Cannot read: {img_path.name}")
            continue
        X.append(feat_fn(bgr))
        y.append(label)
    if not X:
        raise RuntimeError(f"No samples found in {images_dir}")
    return np.vstack(X), np.array(y, dtype=np.int64)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_experiment(label: str, X_train, y_train, X_test, y_test, seed: int = 42):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  Train: {len(y_train)} | Test: {len(y_test)}")
    print(f"  Train distribution: {np.bincount(y_train).tolist()}")
    print(f"  Test  distribution: {np.bincount(y_test).tolist()}")
    print(f"{'='*60}")

    results = {}
    for name, clf in build_models(seed).items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        report = classification_report(
            y_test, y_pred, target_names=["bacteria", "mould"],
            digits=4, zero_division=0,
        )
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n  {name}\n{report}")
        print(f"  Confusion matrix:\n{cm}")
        results[name] = (report, cm)
    return results


def save_results(out_file: Path, label: str, results: dict):
    with open(out_file, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*60}\n  {label}\n{'='*60}\n")
        for model_name, (report, cm) in results.items():
            f.write(f"\n  {model_name}\n{report}\n  Confusion matrix:\n{cm}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_file = args.out_dir / "results.txt"
    out_file.write_text(
        "Baseline comparison (no segmentation) – train/test split\n",
        encoding="utf-8",
    )

    for n_feats, feat_fn in [(16, features_16), (49, features_49)]:
        print(f"\n[INFO] Loading dataset – {n_feats} features...")
        X_train, y_train = load_dataset(args.train_dir, feat_fn)
        X_test,  y_test  = load_dataset(args.test_dir,  feat_fn)
        label = f"{n_feats} features (no segmentation)"
        results = run_experiment(label, X_train, y_train, X_test, y_test, seed=args.seed)
        save_results(out_file, label, results)

    print(f"\n[INFO] Results saved to: {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Baseline comparison (no segmentation) – train/test split"
    )
    parser.add_argument("--train_dir", type=Path, default=Path("data/train/images"))
    parser.add_argument("--test_dir",  type=Path, default=Path("data/test/images"))
    parser.add_argument("--out_dir",   type=Path, default=Path("results/compare_split"))
    parser.add_argument("--seed",      type=int,  default=42)
    main(parser.parse_args())
