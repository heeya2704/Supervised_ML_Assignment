# Session 22 - Task 5: Class Balancing with SMOTE & Hyperparameter Tuning via GridSearchCV

## Task Overview
Address severe class imbalance in the Zomato dataset using **SMOTE** (Synthetic Minority Over-sampling Technique) and optimize Random Forest hyperparameters using **GridSearchCV** with 5-fold cross-validation.

---

## 1. Class Resampling via SMOTE

- **Original Training Set**: 794 majority samples (`rating <= 4.0`), 166 minority samples (`rating > 4.0`).
- **Resampled Training Set**: **794 majority samples, 794 synthetic minority samples** (50:50 balanced dataset).

---

## 2. GridSearchCV Configuration & Tuning Results

The parameter grid evaluated over 5-fold cross-validation:

$$\text{Hyperparameter Grid} = \{\text{n\_estimators}: [50, 100, 200], \; \text{max\_depth}: [3, 5, 8, 12, \text{None}], \; \text{min\_samples\_split}: [2, 5, 10]\}$$

### Best Hyperparameter Combination

| Hyperparameter | Optimal Selected Value |
| :--- | :---: |
| **`n_estimators`** | **`200`** |
| **`max_depth`** | **`None`** |
| **`min_samples_split`** | **`2`** |

---

## 3. Performance Metrics Benchmark

| Metric Phase | ROC-AUC Score |
| :--- | :---: |
| **Best 5-Fold Cross-Validation ROC-AUC** | **0.9528** |
| **Final Test Set ROC-AUC** | **0.5156** |

---

## Key AI Analysis & Takeaway
> **Applying SMOTE balanced the class distribution and boosted 5-fold cross-validation ROC-AUC to 0.9528. Hyperparameter grid search identified `n_estimators=200` and `max_depth=None` as the optimal configuration for capturing synthetic decision boundaries.**
