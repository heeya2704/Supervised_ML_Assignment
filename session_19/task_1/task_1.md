# Session 19 - Task 1: GridSearchCV Hyperparameter Tuning for Product Reviews Classifier

## Task Overview
Tune the `max_depth` and `n_estimators` hyperparameters of a `RandomForestClassifier` on an online product reviews sentiment dataset using exhaustive grid search (`GridSearchCV`).

---

## Hyperparameter Grid Configuration

```python
param_grid = {
    'max_depth': [3, 5, 10, None],
    'n_estimators': [50, 100, 200]
}
```

- **Cross-Validation Folds**: 5 Folds ($5 \times 12 = 60$ total model fits)
- **Scoring Metric**: Accuracy

---

## Best Hyperparameters & Tuning Results

- **Best Parameters**: `{'max_depth': 10, 'n_estimators': 200}`
- **Best 5-Fold CV Score**: **89.33% (0.8933)**
- **Test Set Accuracy**: **90.44% (0.9044)**

```text
                     precision    recall  f1-score   support

Negative Review (0)       0.91      0.90      0.90       225
Positive Review (1)       0.90      0.91      0.91       225

           accuracy                           0.90       450
          macro avg       0.90      0.90      0.90       450
       weighted avg       0.90      0.90      0.90       450
```
