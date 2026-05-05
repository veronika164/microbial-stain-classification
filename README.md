# Microbial Stain Identification on Historical Paper
# Identifikácia mikrobiálnych škvŕn na historickom papieri

---

## English

This repository contains the experimental code developed as part of a diploma thesis focused on **automatic identification of microbial stains on historical paper** using image analysis and machine learning.

The pipeline consists of two main stages: **segmentation** (isolating stain regions from background) and **classification** (identifying the type of microorganism). Both classical machine learning and deep learning approaches were evaluated.

### Repository Structure

```
microbial-stain-classification/
├── README.md
├── classical_svm/                       # Classical ML: LAB segmentation + SVM
│   ├── segmentation.py                  # Segmentation methods: Otsu, LAB, Gabor + batch evaluation
│   ├── utils.py                         # Shared utilities: 49-feature extraction, label parsing
│   ├── train_binary_svm.py              # Binary SVM classifier (bacteria vs. mould)
│   └── train_multiclass_split.py        # Multiclass SVM with GridSearchCV (12 species)
│
├── classical_svm_no_segmentation/       # Classical ML: classification without segmentation
│   ├── features.py                      # Feature vectors (16 and 49) + classifier definitions
│   ├── compare_split.py                 # 16 vs 49 features comparison – train/test split
│
├── cnn_binary/                          # Deep learning: binary classification (bacteria vs. mould)
│   └── train_resnet_binary.py           # ResNet18 with transfer learning
│
└── cnn_multiclass/                      # Deep learning: multiclass classification (12 species)
    └── main.py                          # ResNet18 / EfficientNet-B0 / DenseNet121 training & evaluation
```

### Methods Overview

| Module | Approach | Best Result |
|---|---|---|
| `classical_svm` | LAB segmentation + SVM (49 features) | 87.94 % accuracy (mixed dataset) |
| `classical_svm_no_segmentation` | SVM without segmentation | — |
| `cnn_binary` | ResNet18, transfer learning | 100 % (D2), 97.52 % (cross-device) |
| `cnn_multiclass` | DenseNet121 / EfficientNet-B0 | 96 % / 95.98 % (mixed dataset) |

### Pre-trained Models

The best-performing CNN models are available as a [GitHub Release](../../releases):

| Model | Task | Dataset | Accuracy | F1-macro |
|---|---|---|---|---|
| ResNet18 | binary | D2 | 1.000 | 1.000 |
| ResNet18 | binary | D2 train, D3 test | 0.97 % | 0.97 |
| DenseNet121 | multiclass | mixed D2+D3 | 0.96 | 0.94 |
| EfficientNet-B0 | multiclass | mixed D2+D3 | 0.95 | 0.93 |

### Requirements

- Python 3.10+
- PyTorch, torchvision
- scikit-learn, scikit-image
- OpenCV
- NumPy, matplotlib

---

## Slovenčina

Tento repozitár obsahuje experimentálny kód vytvorený v rámci diplomovej práce zameranej na **automatickú identifikáciu mikrobiálnych škvŕn na historickom papieri** pomocou analýzy obrazu a strojového učenia.

Pipeline pozostáva z dvoch hlavných fáz: **segmentácia** (vyčlenenie oblastí škvŕn z pozadia) a **klasifikácia** (identifikácia druhu mikroorganizmu). Boli porovnané klasické prístupy strojového učenia aj metódy hlbokého učenia.

### Prehľad modulov

| Modul | Prístup | Najlepší výsledok |
|---|---|---|
| `classical_svm` | Segmentácia v priestore LAB + SVM (49 príznakov) | 87,94 % (kombinovaný dataset) |
| `classical_svm_no_segmentation` | SVM bez segmentácie | — |
| `cnn_binary` | ResNet18, prenosové učenie | 100 % (D2), 97,52 % (cross-device) |
| `cnn_multiclass` | DenseNet121 / EfficientNet-B0 | 96 % / 95,98 % (kombinovaný dataset) |

### Natrénované modely

Najlepšie CNN modely sú dostupné ako [GitHub Release](../../releases).

### Datasety

Experimenty pracujú s tromi konfiguráciami datasetu:
- **D2** – snímky z jedného zobrazovacieho zariadenia
- **D3** – snímky z druhého zobrazovacieho zariadenia
- **mixed (D2+D3)** – kombinovaný dataset z oboch zariadení

### Závislosti

- Python 3.10+
- PyTorch, torchvision
- scikit-learn, scikit-image
- OpenCV
- NumPy, matplotlib

---

*Diplomová práca, Fakulta matematiky, fyziky a informatiky, Univerzita Komenského v Bratislave, 2026*
