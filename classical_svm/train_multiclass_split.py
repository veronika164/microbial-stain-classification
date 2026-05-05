"""
Multiclass SVM – train / test split with GridSearchCV.

Trains an RBF-SVM to classify 12 microbial species. Hyperparameters (C, gamma)
are tuned via 3-fold cross-validation on the training set using GridSearchCV.
The best model is then evaluated on the held-out test set.

Usage:
    python train_multiclass_split.py --train_dir data/train/images \\
                                      --test_dir  data/test/images  \\
                                      --out_dir   results/multiclass_split
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from segmentation import segment_color_lab
from utils import (
    adjust_saturation, extract_features, extract_id_from_name,
    list_images, parse_species_from_filename, save_grid_results,
)


def load_dataset(images_dir: Path, lab_channel: str = "b"):
    X, y_names, names = [], [], []
    for img_path in list_images(images_dir):
        species = parse_species_from_filename(img_path.name)
        if species is None:
            continue
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f"[WARN] Cannot read: {img_path.name}")
            continue
        bgr = adjust_saturation(bgr)
        mask = segment_color_lab(bgr, channel=lab_channel)
        X.append(extract_features(bgr, mask))
        y_names.append(species)
        names.append(img_path.name)
    if not X:
        raise RuntimeError(f"No samples found in {images_dir}")
    return np.vstack(X), np.array(y_names), names


def main(args):
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading training data...")
    X_train, y_train_names, _ = load_dataset(args.train_dir, args.lab_channel)
    print(f"[INFO] Loading test data...")
    X_test, y_test_names, test_names = load_dataset(args.test_dir, args.lab_channel)

    print(f"\n[INFO] Train class distribution:\n       {Counter(y_train_names)}")
    print(f"[INFO] Test  class distribution:\n       {Counter(y_test_names)}")

    le = LabelEncoder()
    le.fit(y_train_names)
    unknown = sorted(set(y_test_names) - set(le.classes_))
    if unknown:
        raise RuntimeError(f"Test set contains classes not in training set: {unknown}")

    y_train = le.transform(y_train_names)
    y_test  = le.transform(y_test_names)
    class_names = list(le.classes_)

    base_model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced",
                    random_state=args.seed)),
    ])
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=args.seed)
    param_grid = {
        "clf__C":     [0.1, 1, 10, 50, 100],
        "clf__gamma": ["scale", 0.1, 0.01, 0.001],
    }

    print("\n[INFO] Running GridSearchCV...")
    grid = GridSearchCV(
        base_model, param_grid, scoring="f1_macro",
        cv=cv, n_jobs=-1, verbose=1, refit=True,
    )
    grid.fit(X_train, y_train)
    print(f"[INFO] Best CV f1_macro : {grid.best_score_:.4f}")
    print(f"[INFO] Best params      : {grid.best_params_}")
    save_grid_results(grid, args.out_dir)

    model = grid.best_estimator_
    y_pred  = model.predict(X_test)
    y_prob  = model.predict_proba(X_test)

    acc   = accuracy_score(y_test, y_pred)
    rep_d = classification_report(y_test, y_pred, target_names=class_names,
                                  output_dict=True, zero_division=0)
    rep_t = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
    cm    = confusion_matrix(y_test, y_pred)

    prec_m = rep_d["macro avg"]["precision"]
    rec_m  = rep_d["macro avg"]["recall"]
    f1_m   = rep_d["macro avg"]["f1-score"]
    f1_w   = rep_d["weighted avg"]["f1-score"]

    print("\n=== Test results ===")
    print(f"Accuracy        : {acc:.4f}")
    print(f"Precision macro : {prec_m:.4f}")
    print(f"Recall macro    : {rec_m:.4f}")
    print(f"F1 macro        : {f1_m:.4f}")
    print(f"F1 weighted     : {f1_w:.4f}")
    print("Confusion matrix:\n", cm)

    # Predictions CSV
    with open(args.out_dir / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "id", "true_label", "pred_label", "correct"]
                        + [f"prob_{c}" for c in class_names])
        for i, fname in enumerate(test_names):
            writer.writerow([
                fname, extract_id_from_name(fname),
                class_names[y_test[i]], class_names[y_pred[i]],
                int(y_test[i] == y_pred[i]),
            ] + [round(float(p), 6) for p in y_prob[i]])

    # Metrics CSV
    with open(args.out_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for name, val in [
            ("accuracy", acc), ("precision_macro", prec_m), ("recall_macro", rec_m),
            ("f1_macro", f1_m), ("f1_weighted", f1_w),
            ("best_cv_f1_macro", grid.best_score_),
            ("best_C", grid.best_params_["clf__C"]),
            ("best_gamma", grid.best_params_["clf__gamma"]),
        ]:
            writer.writerow([name, round(val, 4) if isinstance(val, float) else val])

    # Confusion matrix CSV
    with open(args.out_dir / "confusion_matrix.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([""] + class_names)
        for i, row in enumerate(cm):
            writer.writerow([class_names[i]] + row.tolist())

    # Per-class report CSV
    with open(args.out_dir / "report_per_class.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "precision", "recall", "f1-score", "support"])
        for cls in class_names + ["macro avg", "weighted avg"]:
            r = rep_d[cls]
            writer.writerow([cls, round(r["precision"], 4), round(r["recall"], 4),
                             round(r["f1-score"], 4), int(r["support"])])

    # Text report
    with open(args.out_dir / "report.txt", "w", encoding="utf-8") as f:
        f.write(f"Train samples    : {len(y_train)}\n")
        f.write(f"Test samples     : {len(y_test)}\n")
        f.write(f"Features         : {X_train.shape[1]}\n")
        f.write(f"Best CV f1_macro : {grid.best_score_:.4f}\n")
        f.write(f"Best params      : {grid.best_params_}\n")
        f.write(f"Accuracy         : {acc:.4f}\n")
        f.write(f"Precision macro  : {prec_m:.4f}\n")
        f.write(f"Recall macro     : {rec_m:.4f}\n")
        f.write(f"F1 macro         : {f1_m:.4f}\n")
        f.write(f"F1 weighted      : {f1_w:.4f}\n\n")
        f.write("Confusion matrix:\n")
        f.write(str(cm))
        f.write("\n\nClassification report:\n")
        f.write(rep_t)

    print(f"[INFO] Results saved to: {args.out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multiclass SVM with train/test split and GridSearchCV"
    )
    parser.add_argument("--train_dir",   type=Path, default=Path("data/train/images"))
    parser.add_argument("--test_dir",    type=Path, default=Path("data/test/images"))
    parser.add_argument("--out_dir",     type=Path, default=Path("results/multiclass_split"))
    parser.add_argument("--lab_channel", type=str,  default="b")
    parser.add_argument("--seed",        type=int,  default=42)
    main(parser.parse_args())
