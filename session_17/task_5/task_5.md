# Session 17 - Task 5: Metric Evaluation under Extreme Class Imbalance (2% Positive)

## Task Overview
Re-evaluate classification metrics on an extremely imbalanced dataset (98% Legitimate, 2% Fraud) and identify the most misleading performance metric.

---

## Comparison of Computed Metrics

| Metric | Computed Value | Interpretation |
| :--- | :--- | :--- |
| **Accuracy** | **98.22%** | **Highly Misleading**: Driven almost entirely by correctly predicting the 98% majority class. |
| **ROC-AUC** | **0.9614** | **Robust**: Evaluates relative ranking of predicted fraud probabilities across all thresholds. |
| **Log Loss** | **0.0541** | **Calibrated**: Reflects low uncertainty in probability predictions. |

---

## Two-Line Explanation of Most Misleading Metric
> **Accuracy is the most misleading metric under extreme class imbalance because a naive model predicting 100% legitimate transactions achieves 98% accuracy while completely failing to detect any fraudulent transactions.**  
> **In contrast, ROC-AUC and Log Loss evaluate ranked probability predictions and minority class recall, providing a far more realistic measure of model effectiveness.**
