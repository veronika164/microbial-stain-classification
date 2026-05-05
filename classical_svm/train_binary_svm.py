"""
Binary SVM classifier: bacteria vs. mould.

Trains an RBF-SVM on LAB-segmented stain images using a 49-dimensional
feature vector (colour, texture, shape, edges) and evaluates on a held-out
test set.

Usage:
    python train_binary_svm.py --train_dir data/train/images \\
                                --test_dir  data/test/images  \\
                                --out_dir   results/binary
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from segmentation import segment_color_lab
from utils import (
    adjust_saturation, extract_features, extract_id_from_name,
    label_to_name, list_images, parse_binary_label_from_filename,
)


def load_dataset(images_dir: Path, lab_channel: str = "b"):
    X, y, names = [], [], []
    for img_path in list_images(images_dir):
        label = parse_binary_label_from_filename(img_path.name)
        if label is None:
            continue
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f"[WARN] Cannot read: {img_path.name}")
            continue
        bgr = adjust_saturation(bgr)
        mask = segment_color_lab(bgr, channel=lab_channel)
        X.append(extract_features(bgr, mask))
        y.append(label)
        names.append(img_path.name)

    if not X:
        raise RuntimeError(f"No samples found in {images_dir}")
    return np.vstack(X), np.array(y), names


def main(args):
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading training data...")
    X_train, y_train, _ = load_dataset(args.train_dir, args.lab_channel)
    print(f"       {len(X_train)} samples | class distribution: {np.bincount(y_train)}")

    print("[INFO] Loading test data...")
    X_test, y_test_raw, test_names = load_dataset(args.test_dir, args.lab_channel)
    print(f"       {len(X_test)} samples")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)),
    ])
    print("[INFO] Training model...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    # --- Predictions CSV ---
    pred_csv = args.out_dir / "predictions.csv"
    valid_true, valid_pred = [], []
    with open(pred_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "id", "true_label", "pred_label",
                         "correct", "confidence", "prob_bacteria", "prob_mould"])
        for fname, true_lbl, pred_lbl, prob in zip(test_names, y_test_raw, y_pred, y_prob):
            correct, true_name = "", ""
            if true_lbl is not None:
                true_name = label_to_name(true_lbl)
                correct = int(int(true_lbl) == int(pred_lbl))
                valid_true.append(int(true_lbl))
                valid_pred.append(int(pred_lbl))
            writer.writerow([
                fname, extract_id_from_name(fname), true_name,
                label_to_name(int(pred_lbl)), correct,
                round(float(max(prob)), 3),
                round(float(prob[0]), 3), round(float(prob[1]), 3),
            ])
    print(f"[INFO] Predictions → {pred_csv}")

    # --- Metrics ---
    if not valid_true:
        print("[INFO] No true labels found; skipping metrics.")
        return

    acc   = accuracy_score(valid_true, valid_pred)
    prec  = precision_score(valid_true, valid_pred, zero_division=0)
    rec   = recall_score(valid_true, valid_pred, zero_division=0)
    f1    = f1_score(valid_true, valid_pred, zero_division=0)
    cm    = confusion_matrix(valid_true, valid_pred)
    report = classification_report(
        valid_true, valid_pred, target_names=["bacteria", "mould"], zero_division=0,
    )

    print("\n=== Results ===")
    print(f"Accuracy  : {acc:.4f} ({acc*100:.2f} %)")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1-score  : {f1:.4f}")
    print("Confusion matrix:\n", cm)

    metrics_csv = args.out_dir / "metrics.csv"
    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for name, val in [("accuracy", acc), ("precision", prec), ("recall", rec), ("f1_score", f1)]:
            writer.writerow([name, round(val, 4)])

    np.savetxt(args.out_dir / "confusion_matrix.csv", cm, delimiter=",", fmt="%d")

    with open(args.out_dir / "report.txt", "w", encoding="utf-8") as f:
        f.write(f"Accuracy  : {acc:.4f} ({acc*100:.2f} %)\n")
        f.write(f"Precision : {prec:.4f}\n")
        f.write(f"Recall    : {rec:.4f}\n")
        f.write(f"F1-score  : {f1:.4f}\n\n")
        f.write("Confusion matrix:\n")
        f.write(str(cm))
        f.write("\n\nClassification report:\n")
        f.write(report)

    print(f"[INFO] Metrics → {metrics_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Binary SVM: bacteria vs. mould")
    parser.add_argument("--train_dir",   type=Path, default=Path("data/train/images"))
    parser.add_argument("--test_dir",    type=Path, default=Path("data/test/images"))
    parser.add_argument("--out_dir",     type=Path, default=Path("results/binary"))
    parser.add_argument("--lab_channel", type=str,  default="b",
                        help="LAB channel used for segmentation: 'a' or 'b'")
    main(parser.parse_args())
