# Session 18 - Task 2: 10-Fold Cross-Validation on Wine Dataset with Decision Tree

## Task Overview
Perform 10-fold cross-validation using `cross_val_score()` with a `DecisionTreeClassifier` on the Wine dataset and record individual fold accuracy scores.

---

## 10-Fold Cross-Validation Performance Breakdown

- **Model**: `DecisionTreeClassifier(random_state=42)`
- **Dataset**: Wine Recognition Dataset ($N=178$ samples, 13 features)
- **Results per Fold**:

| Fold Number | Accuracy |
| :---: | :---: |
| Fold 1 | **0.8889 (88.89%)** |
| Fold 2 | **0.8889 (88.89%)** |
| Fold 3 | **0.8333 (83.33%)** |
| Fold 4 | **0.9444 (94.44%)** |
| Fold 5 | **0.8889 (88.89%)** |
| Fold 6 | **0.8889 (88.89%)** |
| Fold 7 | **0.8889 (88.89%)** |
| Fold 8 | **0.8824 (88.24%)** |
| Fold 9 | **0.8824 (88.24%)** |
| Fold 10 | **1.0000 (100.0%)** |
| **Mean Accuracy** | **0.8986 (89.86%)** |
| **Std Dev** | **$\pm 0.0401$** |
