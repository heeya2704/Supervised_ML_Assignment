# Session 7 - Task 1: Zomato Delivery Time Error Metrics (MSE & RMSE)

## Task Overview
Calculate Mean Squared Error (MSE) and Root Mean Squared Error (RMSE) for 10 Zomato food delivery orders given actual and predicted delivery times.

---

## 10 Zomato Orders Telemetry Data

| Order # | Actual Delivery Time (mins) | Predicted Delivery Time (mins) | Residual Error ($y - \hat{y}$) | Squared Error ($(y - \hat{y})^2$) |
|---|---|---|---|---|
| Order 1 | 25 min | 28 min | -3 min | 9 |
| Order 2 | 30 min | 27 min | +3 min | 9 |
| Order 3 | 45 min | 48 min | -3 min | 9 |
| Order 4 | 20 min | 24 min | -4 min | 16 |
| Order 5 | 35 min | 32 min | +3 min | 9 |
| Order 6 | 50 min | 55 min | -5 min | 25 |
| Order 7 | 28 min | 30 min | -2 min | 4 |
| Order 8 | 40 min | 37 min | +3 min | 9 |
| Order 9 | 32 min | 35 min | -3 min | 9 |
| Order 10 | 22 min | 20 min | +2 min | 4 |

---

## Metric Calculations & Formulae

$$\text{MSE} = \frac{1}{10} \sum_{i=1}^{10} (y_i - \hat{y}_i)^2 = \frac{103}{10} = \mathbf{10.30 \text{ min}^2}$$

$$\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{10.30} = \mathbf{3.2094 \text{ minutes}}$$

---

## Interpretation
- **Mean Squared Error (MSE = 10.30):** Represents average squared discrepancy in delivery time estimates.
- **Root Mean Squared Error (RMSE = 3.21 minutes):** On average, Zomato ETA predictions deviate from actual delivery times by approximately **3.21 minutes**.
