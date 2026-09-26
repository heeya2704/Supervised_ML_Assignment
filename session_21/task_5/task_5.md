# Session 21 - Task 5: Symmetric Mean Absolute Percentage Error (SMAPE) Implementation

## Task Overview
Implement custom **Symmetric Mean Absolute Percentage Error (SMAPE)** function to evaluate Random Forest temperature forecasting predictions and analyze why SMAPE is preferred over standard MAPE.

---

## Mathematical Formula

$$\text{SMAPE} = \frac{100\%}{n} \sum_{i=1}^{n} \frac{|y_i - \hat{y}_i|}{(|y_i| + |\hat{y}_i|) / 2}$$

---

## Metric Results

| Evaluation Metric | Score |
| :--- | :---: |
| **Calculated SMAPE** | **5.3442%** |
| **Baseline MAE** | **1.4995°C** |

---

## AI Learning Takeaway & Explanation
> **Unlike standard MAPE (which is asymmetric, penalizes over-forecasting far more severely than under-forecasting, and fails when actual values are zero), SMAPE normalizes residual errors by the average of actual and predicted absolute values, bounding error strictly between 0% and 200% and providing fair, symmetric evaluation across all time steps.**
