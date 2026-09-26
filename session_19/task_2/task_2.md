# Session 19 - Task 2: RandomizedSearchCV Tuning for Instagram Engagement Classifier

## Task Overview
Tune the regularization parameter `C` and kernel coefficient `gamma` of a Support Vector Classifier (`SVC`) on an Instagram engagement prediction dataset using `RandomizedSearchCV` limited to `n_iter=10`.

---

## Hyperparameter Distribution & Search Setup

```python
param_distributions = {
    'C': loguniform(1e-2, 1e2),
    'gamma': loguniform(1e-3, 1e1),
    'kernel': ['rbf', 'sigmoid']
}
```

- **Iterations ($n\_iter$)**: 10 sampled combinations
- **Cross-Validation**: 5-Fold CV ($10 \times 5 = 50$ total model fits)

---

## Results & Selected Hyperparameters

- **Best Parameters**:
  - `C`: **13.5684**
  - `gamma`: **0.0217**
  - `kernel`: **`'rbf'`**
- **Best CV Accuracy**: **88.33% (0.8833)**
- **Test Set Accuracy**: **89.44% (0.8944)**
