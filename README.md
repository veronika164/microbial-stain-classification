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
├── classical_svm/                      # Classical ML: segmentation + SVM classification
│   ├── extract_features.py             # Feature extraction from segmented regions (49 features)
│   ├── feature_importance.py           # Feature importance analysis
│   ├── classify.py                     # Binary and multiclass SVM classification
│   ├── multiclass.py                   # Multiclass SVM experiments
│   ├── experiment_saturation.py        # Saturation-based segmentation experiments
│   ├── experiment_preprocessing.py     # Preprocessing pipeline experiments
│   ├── prepare_multiclass_cross_device.py  # Cross-device dataset preparation
│   ├── prepare_multiclass_mixed_device.py  # Mixed-device dataset preparation
│   ├── test_svm_multiclass.py          # Multiclass SVM evaluation
│   ├── test_binary_full_vector.py      # Binary classification with full feature vector
│   ├── test.py                         # General testing script
│   ├── image_gen.py                    # Image generation utilities
│   └── results/
│       ├── results_segmentation.csv    # Segmentation evaluation results
│       ├── confusion_matrix_D3.png     # Confusion matrix – D3 dataset
│       └── confusion_matrix_mixed.png  # Confusion matrix – mixed dataset
│
├── classical_svm_no_segmentation/      # Classical ML: classification without segmentation
│   ├── csv_utils.py                    # CSV loading and utilities
│   ├── features.py                     # Feature definitions
│   ├── test_49.py                      # Evaluation with 49-feature vector
│   ├── test_d2.py                      # Evaluation on D2 dataset
│   ├── test_priznaky.py                # Feature-level analysis
│   └── train_eval.py                   # Training and evaluation pipeline
│
├── cnn_binary/                         # Deep learning: binary classification (mould vs. bacteria)
│   ├── image.py                        # Image loading and augmentation utilities
│   ├── prepare_binary_split.py         # Train/test split preparation
│   ├── prepare_binary_cross_device.py  # Cross-device split preparation
│   ├── prepare_binary_mixed_device.py  # Mixed-device split preparation
│   └── train_resnet_binary.py          # ResNet18 binary classifier training
│
└── cnn_multiclass/                     # Deep learning: multiclass classification (12 species)
    └── main.py                         # ResNet18 / EfficientNet-B0 / DenseNet121 training & evaluation
```

### Methods Overview

| Module | Approach | Best Result |
|---|---|---|
| `classical_svm` | LAB segmentation + SVM (49 features) | 87.94 % accuracy (mixed dataset) |
| `classical_svm_no_segmentation` | SVM without segmentation | — |
| `cnn_binary` | ResNet18, transfer learning | 100 % (D2), 97.52 % (cross-device) |
| `cnn_multiclass` | DenseNet121 / EfficientNet-B0 | 96 % / 95.98 % (mixed dataset) |

### Requirements

- Python 3.10+
- PyTorch, torchvision
- scikit-learn
- OpenCV
- NumPy, pandas, matplotlib

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

### Datasety

Experimenty pracujú s tromi konfiguráciami datasetu:
- **D2** – snímky z jedného zobrazovacie zariadenia
- **D3** – snímky z druhého zobrazovacieho zariadenia
- **mixed (D2+D3)** – kombinovaný dataset z oboch zariadení

### Závislostiami

- Python 3.10+
- PyTorch, torchvision
- scikit-learn
- OpenCV
- NumPy, pandas, matplotlib

---

*Diplomová práca, Fakulta matematiky, fyziky a informatiky, Univerzita Komenského v Bratislave, 2026*
