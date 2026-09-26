# Session 5 - Task 4: Model Evaluation Metrics

## Task Overview
Evaluate the performance of the Simple Linear Regression model using statistical metrics: **Mean Absolute Error (MAE)**, **Mean Squared Error (MSE)**, **Root Mean Squared Error (RMSE)**, and **$R^2$ Score (Coefficient of Determination)**.

---

## Actual vs. Predicted Weekly Battery Percentage

| Day | Instagram Usage (Hours) | Actual Battery % | Predicted Battery % | Absolute Difference |
|---|---|---|---|---|
| Monday | 1.5 hrs | 88.0% | 87.36% | 0.64% |
| Tuesday | 2.0 hrs | 82.0% | 81.34% | 0.66% |
| Wednesday | 3.5 hrs | 65.0% | 63.29% | 1.71% |
| Thursday | 4.0 hrs | 58.0% | 57.27% | 0.73% |
| Friday | 2.5 hrs | 72.0% | 75.32% | 3.32% |
| Saturday | 5.5 hrs | 40.0% | 39.22% | 0.78% |
| Sunday | 6.0 hrs | 32.0% | 33.20% | 1.20% |

---

## Performance Summary Table

| Metric | Formula | Value | Interpretation |
|---|---|---|---|
| **MAE** | $\frac{1}{n} \sum \|y - \hat{y}\|$ | **1.2921%** | Predictions off by only ~1.29 percentage points on average. |
| **MSE** | $\frac{1}{n} \sum (y - \hat{y})^2$ | **2.4865** | Low average squared residual error. |
| **RMSE** | $\sqrt{\text{MSE}}$ | **1.5769%** | Low error standard deviation across all 7 test days. |
| **$R^2$ Score** | $1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}}$ | **0.9933 (99.33%)** | **99.33%** of battery drain variance is explained by Instagram usage time. |

---

## Key Takeaways
- The model exhibits an exceptionally strong fit ($R^2 = 0.9933$), demonstrating that daily Instagram usage time is an effective linear predictor of end-of-day phone battery drain.
