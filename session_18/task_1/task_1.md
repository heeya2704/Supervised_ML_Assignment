# Session 18 - Task 1: 5-Fold Cross-Validation on Iris Dataset with KNN

## Task Overview
Load the Iris flower dataset from scikit-learn and compute 5-fold cross-validation accuracy using `KNeighborsClassifier`.

---

## 5-Fold Cross-Validation Breakdown

- **Model**: `KNeighborsClassifier(n_neighbors=5)`
- **Dataset**: Iris Flower Dataset ($N=150$ samples, 4 features, 3 target classes)
- **Folds ($k=5$)**:

| Fold | Accuracy |
| :--- | :--- |
| Fold 1 | **0.9667 (96.67%)** |
| Fold 2 | **1.0000 (100.00%)** |
| Fold 3 | **0.9333 (93.33%)** |
| Fold 4 | **0.9667 (96.67%)** |
| Fold 5 | **1.0000 (100.00%)** |
| **Mean CV Accuracy** | **0.9733 (97.33%)** |
| **Standard Deviation** | **$\pm 0.0249$** |
