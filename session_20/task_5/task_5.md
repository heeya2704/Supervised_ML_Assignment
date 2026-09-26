# Session 20 - Task 5: ARIMA Model Fitting & 7-Day Temperature Forecast

## Task Overview
Fit an $\text{ARIMA}(p, d, q)$ time series model using `statsmodels.tsa.arima.model.ARIMA` and generate a 7-day out-of-sample temperature forecast with $95\%$ confidence bounds.

---

## Model Selection Rationale

- **$d = 1$**: Determined via the Augmented Dickey-Fuller (ADF) test ($p \ge 0.05$ on raw series, $p < 0.0001$ on first difference).
- **$p = 2$**: Autoregressive order capturing short-term thermal inertia over recent days.
- **$q = 2$**: Moving Average order modeling random shock residuals.

---

## 7-Day Out-of-Sample Forecast Table

| Date | Forecasted Temp (°C) | 95% Confidence Interval |
| :---: | :---: | :---: |
| **Day 1** | **20.21°C** | 16.68°C – 23.74°C |
| **Day 2** | **20.67°C** | 16.71°C – 24.63°C |
| **Day 3** | **21.05°C** | 16.77°C – 25.33°C |
| **Day 4** | **21.36°C** | 16.82°C – 25.90°C |
| **Day 5** | **21.60°C** | 16.87°C – 26.33°C |
| **Day 6** | **21.78°C** | 16.92°C – 26.64°C |
| **Day 7** | **21.91°C** | 16.98°C – 26.84°C |

---

## Forecast Plot
The 7-day out-of-sample temperature forecast visualization with confidence ribbons is saved to [`arima_forecast_plot.png`](arima_forecast_plot.png).
