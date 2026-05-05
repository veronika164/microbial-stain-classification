"""
Binary image classifier using ResNet18 with transfer learning.

Classifies microbial stain images into two categories: bacteria vs. mould.
Expects an ImageFolder-compatible directory structure:

    data_dir/
        train/
            bacteria/
            mould/
        val/
            bacteria/
            mould/
        test/
            bacteria/
            mould/

The best model (highest validation F1) is saved as best_model.pt.

Usage:
    python train_resnet_binary.py --data_dir data/D2_split \\
                                   --out_dir  results/resnet18_binary
"""

import argparse
import copy
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main(args):
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    train_tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(args.data_dir / "train", transform=train_tf)
    val_ds   = datasets.ImageFolder(args.data_dir / "val",   transform=eval_tf)
    test_ds  = datasets.ImageFolder(args.data_dir / "test",  transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False)

    class_names = train_ds.classes
    print(f"[INFO] Classes: {class_names}")
    print(f"[INFO] Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_weights = copy.deepcopy(model.state_dict())
    best_val_f1  = 0.0
    train_losses, val_losses = [], []

    for epoch in range(args.epochs):
        print(f"\n[INFO] Epoch {epoch + 1}/{args.epochs}")

        # --- Train ---
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        train_losses.append(running_loss / len(train_ds))

        # --- Validation ---
        model.eval()
        running_loss = 0.0
        y_true, y_pred = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                running_loss += criterion(outputs, labels).item() * inputs.size(0)
                y_true.extend(labels.cpu().tolist())
                y_pred.extend(torch.argmax(outputs, dim=1).cpu().tolist())
        val_losses.append(running_loss / len(val_ds))
        val_f1 = f1_score(y_true, y_pred, average="binary")

        print(f"  Train loss : {train_losses[-1]:.4f}")
        print(f"  Val loss   : {val_losses[-1]:.4f}")
        print(f"  Val F1     : {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), args.out_dir / "best_model.pt")
            print("  [INFO] New best model saved.")

    # --- Test ---
    model.load_state_dict(best_weights)
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(torch.argmax(model(inputs), dim=1).cpu().tolist())

    acc    = accuracy_score(y_true, y_pred)
    f1     = f1_score(y_true, y_pred, average="binary")
    cm     = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)

    print("\n=== Test results ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1       : {f1:.4f}")
    print("Confusion matrix:\n", cm)
    print("\nClassification report:\n", report)

    with open(args.out_dir / "test_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Accuracy : {acc:.4f}\nF1       : {f1:.4f}\n\n")
        f.write("Confusion matrix:\n")
        f.write(str(cm))
        f.write("\n\nClassification report:\n")
        f.write(report)

    # Loss curve
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label="train")
    plt.plot(val_losses,   label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_dir / "loss_curve.png", dpi=200)
    plt.close()

    print(f"\n[INFO] Results saved to: {args.out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Binary ResNet18 classifier")
    parser.add_argument("--data_dir",  type=Path, default=Path("data"),
                        help="Root with train/val/test subfolders")
    parser.add_argument("--out_dir",   type=Path, default=Path("results/resnet18_binary"))
    parser.add_argument("--epochs",    type=int,  default=15)
    parser.add_argument("--batch_size",type=int,  default=16)
    parser.add_argument("--lr",        type=float,default=1e-4)
    parser.add_argument("--img_size",  type=int,  default=224)
    parser.add_argument("--seed",      type=int,  default=42)
    main(parser.parse_args())