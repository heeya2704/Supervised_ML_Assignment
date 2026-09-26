# Session 12 - Task 4: Feature Importance Analysis in Decision Trees

## Task Overview
Extract, rank, and interpret feature importances from a trained `DecisionTreeClassifier` on the Iris dataset using the `feature_importances_` attribute.

---

## Ranked Feature Importance Table

$$\text{Importance}(f) = \sum_{t \in \text{nodes splitting on } f} \frac{N_t}{N} \Delta \text{Gini}(t)$$

| Rank | Feature Name | Gini Importance Score | Percentage Contribution | Interpretation |
|---|---|---|---|---|
| **#1** | `petal length (cm)` | **0.541176** | **54.12%** | Primary root split feature; separates Setosa |
| **#2** | `petal width (cm)` | **0.430252** | **43.03%** | Secondary split feature; separates Versicolor/Virginica |
| **#3** | `sepal width (cm)` | **0.028571** | **2.86%** | Minor boundary fine-tuning split |
| **#4** | `sepal length (cm)` | **0.000000** | **0.00%** | Not selected for any decision split |

---

## Key Insights
- **Dominance of Petal Metrics**: Combined, `petal length` and `petal width` account for **97.15%** of the decision tree's total impurity reduction.
- **Redundancy**: `sepal length` has zero importance (`0.00%`), indicating it is redundant when petal dimensions are available.
