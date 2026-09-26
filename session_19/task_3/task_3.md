# Session 19 - Task 3: SMOTE Oversampling + GridSearchCV Hyperparameter Tuning

## Task Overview
Apply Synthetic Minority Over-sampling Technique (`SMOTE`) from `imblearn.over_sampling` to balance an imbalanced dataset (90% Majority, 10% Minority) prior to running hyperparameter optimization via `GridSearchCV`.

---

## Class Resampling Transformation

- **Original Training Set**:
  - Class 0 (Majority): 1,260 samples (90.0%)
  - Class 1 (Minority): 140 samples (10.0%)
- **Resampled Training Set (Post-SMOTE)**:
  - Class 0 (Majority): 1,260 samples (50.0%)
  - Class 1 (Minority): 1,260 samples (50.0%) — **100% Balanced**

---

## Best Hyperparameters & Evaluation

```python
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10],
    'min_samples_split': [2, 5]
}
```

- **Best Hyperparameters**: `{'max_depth': 10, 'min_samples_split': 2, 'n_estimators': 50}`
- **Test Set Minority Class Recall**: **91.67% (0.9167)**
- **Test Set Minority Class F1-Score**: **0.8000**
