# Session 18 - Task 3: Stratified K-Fold Sampling Class Distribution Verification

## Task Overview
Use `StratifiedKFold` from scikit-learn on the Iris flower dataset to split samples across 5 folds and verify class count distribution equality.

---

## Fold Class Distribution Verification

| Fold | Setosa (Class 0) | Versicolor (Class 1) | Virginica (Class 2) | Total Test Samples |
| :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | 10 | 10 | 10 | **30** |
| **Fold 2** | 10 | 10 | 10 | **30** |
| **Fold 3** | 10 | 10 | 10 | **30** |
| **Fold 4** | 10 | 10 | 10 | **30** |
| **Fold 5** | 10 | 10 | 10 | **30** |

---

## Technical Significance
- **Equal Representation**: Every test fold contains exactly **33.3% Setosa**, **33.3% Versicolor**, and **33.3% Virginica**, preserving original population proportions.
- **Prevents Evaluation Bias**: Standard unstratified sampling could omit entire minority classes in smaller folds, introducing extreme variance into model evaluation.
