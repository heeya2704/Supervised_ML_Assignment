# Session 17 - Task 2: ROC Curve & AUC Score Visualization

## Task Overview
Plot the Receiver Operating Characteristic (ROC) curve for the trained Logistic Regression fraud model using `sklearn.metrics.roc_curve` and display the AUC score on the plot.

---

## Visualization
The generated ROC plot is saved to [`roc_curve.png`](roc_curve.png).

---

## Mathematical Breakdown

$$\text{FPR} = \frac{FP}{TN + FP}, \quad \text{TPR} = \frac{TP}{TP + FN}$$

- **AUC Score**: Measures the classifier's ability to rank positive (fraud) cases higher than negative (legitimate) cases across all probability thresholds.
- **Computed AUC Score**: **0.9575 (95.75%)**
