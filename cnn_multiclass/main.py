import copy
import csv
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms


# ===== PATHS (EDIT HERE) =====
TRAIN_DIR = Path(r"C:\Archiv\FMFI\DP\pokusy\cnn_multi\data\D2_D3_mixed_multiclass\train\images")
TEST_DIR  = Path(r"C:\Archiv\FMFI\DP\pokusy\cnn_multi\data\D2_D3_mixed_multiclass\test\images")
#OUT_DIR   = Path(r"C:\Archiv\FMFI\DP\pokusy\cnn_multi\outputs\resnet18_multiclass_mixed")

OUT_DIR = Path(r"C:\Archiv\FMFI\DP\pokusy\cnn_multi\outputs\resnet18_multiclass_mixed")


from pathlib import Path

train_files = {p.name for p in Path(r"C:\Archiv\FMFI\DP\pokusy\cnn_multi\data\D2_multiclass\train\images").iterdir()}
test_files  = {p.name for p in Path(r"C:\Archiv\FMFI\DP\pokusy\cnn_multi\data\D2_multiclass\test\images").iterdir()}

overlap = train_files & test_files
print(f"Prekryv: {len(overlap)} súborov")
print(overlap)

# ===== HYPERPARAMETRE =====
BATCH_SIZE = 4
NUM_EPOCHS = 30
LR         = 1e-4
SEED       = 42
IMG_SIZE   = 224
VAL_SPLIT  = 0.2

#MODEL_NAME = "efficientnet_b0"   # možnosti: "resnet18", "efficientnet_b0", "densenet121"
MODEL_NAME = "resnet18"


# ===== PARSOVANIE NÁZVOV SÚBOROV =====

NORMALIZATION_MAP = {
    "alternaria-alternata": "alternaria_alternata",
    "aspergillus-ochraceus": "aspergillus_ochraceus",
    "talaromyces-purpureogenus": "talaromyces_purpureogenus",
    "micrococcus-sp": "micrococcus_sp",
    "sphingomonas-aquatilis": "sphingomonas_aquatilis",
    "curtobacterium-flaccumfaciens": "curtobacterium_flaccumfaciens",
    "rhodococcus-degradans": "rhodococcus_degradans",
    "epicoccum-layunese": "epicoccum_layunese",
    "penicilium-mangini": "penicillium_mangini",
    "penicillium-mangini": "penicillium_mangini",
    "arthrobacter-sp": "arthrobacter_sp",
    "arthrobacter-sp4": "arthrobacter_sp",
    "cladosporium-pseudocladosporioides": "cladosporium_pseudocladosporioides",
    "exiguobacterium-indicum": "exiguobacterium_indicum",
}

KEYWORD_TO_CLASS = {
    "alternaria": "alternaria_alternata",
    "aspergillus": "aspergillus_ochraceus",
    "talaromyces": "talaromyces_purpureogenus",
    "micrococcus": "micrococcus_sp",
    "sphingomonas": "sphingomonas_aquatilis",
    "curtobacterium": "curtobacterium_flaccumfaciens",
    "rhodococcus": "rhodococcus_degradans",
    "epicoccum": "epicoccum_layunese",
    "penicilium": "penicillium_mangini",
    "penicillium": "penicillium_mangini",
    "arthrobacter": "arthrobacter_sp",
    "cladosporium": "cladosporium_pseudocladosporioides",
    "exiguobacterium": "exiguobacterium_indicum",
}


def build_model(model_name: str, num_classes: int):
    model_name = model_name.lower()

    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif model_name == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT
        model = models.densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    else:
        raise ValueError(f"Neznámy model: {model_name}")

    return model

def normalize_class_name(raw: str):
    x = raw.strip().lower()
    for sk, en in [("á","a"),("é","e"),("í","i"),("ó","o"),("ú","u"),
                   ("ň","n"),("ľ","l"),("š","s"),("č","c"),("ť","t"),
                   ("ž","z"),("ý","y")]:
        x = x.replace(sk, en)
    x = re.sub(r"\s+", " ", x)
    if x in NORMALIZATION_MAP:
        return NORMALIZATION_MAP[x]
    x2 = x.replace("_", "-").replace(" ", "-")
    if x2 in NORMALIZATION_MAP:
        return NORMALIZATION_MAP[x2]
    for key, value in KEYWORD_TO_CLASS.items():
        if key in x:
            return value
    return None


def parse_species_from_filename(fname: str):
    stem = Path(fname).stem
    if "__" in stem:
        stem = stem.split("__", 1)[1]
    parts = stem.split("_")
    if len(parts) < 3:
        print(f"[WARN] Zlý formát názvu: {fname}")
        return None
    species = normalize_class_name(parts[2])
    if species is None:
        print(f"[WARN] Neznáma trieda: {fname} -> {parts[2]}")
    return species


# ===== VLASTNÝ DATASET =====

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

class StainDataset(Dataset):
    def __init__(self, img_dir: Path, class_to_idx: dict, transform=None):
        self.transform = transform
        self.samples   = []
        for p in sorted(img_dir.iterdir()):
            if p.suffix.lower() not in EXTS:
                continue
            species = parse_species_from_filename(p.name)
            if species is None or species not in class_to_idx:
                continue
            self.samples.append((p, class_to_idx[species]))
        print(f"[INFO] {img_dir.name}: {len(self.samples)} vzoriek")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def main():
    torch.manual_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Zistíme triedy z train priečinka
    all_species = set()
    for p in TRAIN_DIR.iterdir():
        if p.suffix.lower() not in EXTS:
            continue
        s = parse_species_from_filename(p.name)
        if s:
            all_species.add(s)

    class_names  = sorted(all_species)
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    num_classes  = len(class_names)
    print(f"[INFO] Triedy ({num_classes}): {class_names}")

    # ===== TRANSFORMÁCIE =====
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    # ===== DATASETY =====
    full_train_ds = StainDataset(TRAIN_DIR, class_to_idx, transform=eval_tf)
    test_ds       = StainDataset(TEST_DIR,  class_to_idx, transform=eval_tf)

    # Auto val split 80/20 – stratifikovaný
    indices = list(range(len(full_train_ds)))
    labels  = [full_train_ds.samples[i][1] for i in indices]

    train_idx, val_idx = train_test_split(
        indices, test_size=VAL_SPLIT, stratify=labels, random_state=SEED
    )

    # Train subset dostane augmentáciu
    train_ds_aug = StainDataset(TRAIN_DIR, class_to_idx, transform=train_tf)
    train_ds     = Subset(train_ds_aug, train_idx)
    val_ds       = Subset(full_train_ds, val_idx)

    print(f"[INFO] Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    # ===== MODEL =====
    model = build_model(MODEL_NAME, num_classes)
    model = model.to(device)

    print(f"[INFO] Použitý model: {MODEL_NAME}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_f1    = 0.0
    train_losses   = []
    val_losses     = []

    # ===== TRÉNOVANIE =====
    for epoch in range(NUM_EPOCHS):
        print(f"\n[INFO] Epoch {epoch+1}/{NUM_EPOCHS}")

        # TRAIN
        model.train()
        running_loss = 0.0
        for inputs, lbls in train_loader:
            inputs, lbls = inputs.to(device), lbls.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), lbls)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_train_loss = running_loss / len(train_idx)
        train_losses.append(epoch_train_loss)

        # VAL
        model.eval()
        running_loss   = 0.0
        y_true, y_pred = [], []
        with torch.no_grad():
            for inputs, lbls in val_loader:
                inputs, lbls = inputs.to(device), lbls.to(device)
                out  = model(inputs)
                running_loss += criterion(out, lbls).item() * inputs.size(0)
                preds = torch.argmax(out, dim=1)
                y_true.extend(lbls.cpu().numpy().tolist())
                y_pred.extend(preds.cpu().numpy().tolist())

        epoch_val_loss = running_loss / len(val_idx)
        val_losses.append(epoch_val_loss)
        val_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        print(f"Train loss: {epoch_train_loss:.4f}")
        print(f"Val loss  : {epoch_val_loss:.4f}")
        print(f"Val F1    : {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1    = val_f1
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), OUT_DIR / "best_model.pt")
            print("[INFO] Saved new best model.")

    model.load_state_dict(best_model_wts)

    # ===== TEST =====
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, lbls in test_loader:
            inputs = inputs.to(device)
            preds  = torch.argmax(model(inputs), dim=1)
            y_true.extend(lbls.numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

    acc    = accuracy_score(y_true, y_pred)
    f1     = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    report = classification_report(
        y_true, y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        digits=4,
        zero_division=0
    )


    print("\n=== TEST RESULTS ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 macro: {f1:.4f}")
    print("Confusion matrix:")
    print(cm)
    print("\nClassification report:")
    print(report)

    with open(OUT_DIR / "test_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Multiclass {MODEL_NAME} classification report\n")
        f.write("=========================================\n\n")
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"F1 macro: {f1:.4f}\n\n")
        f.write("Confusion matrix:\n")
        f.write(str(cm))
        f.write("\n\nClassification report:\n")
        f.write(report)

    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label="train_loss")
    plt.plot(val_losses,   label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "loss_curve.png", dpi=200)
    plt.close()

    print(f"\n[INFO] Saved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()