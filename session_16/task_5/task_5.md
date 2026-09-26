# Session 16 - Task 5: Receiver Operating Characteristic (ROC) Curve & AUC Score Analysis

## Task Overview
Evaluate a binary classification model (Fraud Detection) using Receiver Operating Characteristic (ROC) curves and Area Under Curve (AUC) scores across varying classification decision thresholds.

---

## Core Concepts & Formulas

### 1. True Positive Rate ($\text{TPR}$) / Recall / Sensitivity
$$\text{TPR} = \frac{TP}{TP + FN}$$

### 2. False Positive Rate ($\text{FPR}$) / ($1 - \text{Specificity}$)
$$\text{FPR} = \frac{FP}{TN + FP}$$

### 3. Area Under the ROC Curve ($\text{AUC}$)
- **$\text{AUC} = 1.0$**: Perfect classifier (100% separation between classes).
- **$\text{AUC} = 0.5$**: Random baseline classifier (no predictive power).
- **$\text{AUC} < 0.5$**: Classifier performing worse than random guessing.

---

## Optimal Decision Threshold Selection (Youden's J Statistic)

Youden's J statistic maximizes the distance between the ROC curve and the random diagonal baseline:
$$J = \text{TPR} - \text{FPR} = \text{Sensitivity} + \text{Specificity} - 1$$

The decision threshold that yields the maximum value of $J$ represents the optimal trade-off between maximizing true detections and minimizing false alarms.

---

## Generated Visualization
The ROC curve plot is generated and saved as [`roc_curve.png`](roc_curve.png).
