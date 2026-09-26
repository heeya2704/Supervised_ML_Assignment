# Session 19 - Task 5: AI-Suggested Parameter Grid Tuning Rationale

## Task Overview
Implement an AI-suggested parameter grid for `RandomForestClassifier` on the Breast Cancer dataset using `GridSearchCV` and document the theoretical rationale behind each hyperparameter range.

---

## Parameter Grid & Search Rationale

```python
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 6, 10, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'criterion': ['gini', 'entropy']
}
```

### Parameter Selection Rationale
1. **`n_estimators` (`[50, 100, 200]`)**: Tests ensemble size trade-offs. Higher tree counts reduce variance up to a point of diminishing returns.
2. **`max_depth` (`[3, 6, 10, None]`)**: Controls maximum tree depth. Shallow depths ($3-6$) prevent overfitting, while `None` allows full expansion until pure leaves.
3. **`min_samples_split` (`[2, 5]`)**: Sets the minimum number of samples required to split an internal node, preventing tiny noisy splits.
4. **`min_samples_leaf` (`[1, 2]`)**: Guarantees leaf nodes retain a minimum sample count, smoothing decision boundaries.
5. **`criterion` (`['gini', 'entropy']`)**: Compares Gini Impurity against Information Gain (Entropy) to optimize split quality.

---

## Results Summary

- **Best Parameters**: `{'criterion': 'entropy', 'max_depth': 6, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 50}`
- **Cross-Validation F1-Score**: **0.9696**
- **Test Set Accuracy**: **95.80% (0.9580)**
- **Test Set F1-Score**: **0.9667**
