# Session 20 - Task 2: Seasonal Decomposition Analysis

## Task Overview
Decompose the daily temperature time series into its constituent **Observed**, **Trend**, **Seasonal**, and **Residual** components using `statsmodels.tsa.seasonal.seasonal_decompose(period=365)`.

---

## Additive Decomposition Model Formula

$$Y[t] = T[t] + S[t] + e[t]$$

Where:
- $Y[t]$: Observed temperature at day $t$.
- $T[t]$: Long-term underlying climate trend.
- $S[t]$: Recurring annual seasonal component ($period = 365$).
- $e[t]$: Irregular random residual noise component.

---

## Component Breakdown & Plot
The full 4-panel decomposition chart is saved to [`seasonal_decomposition.png`](seasonal_decomposition.png).

- **Trend Component**: Smooth trajectory displaying long-term directional movement.
- **Seasonal Component**: Perfect 365-day repeating annual wave ranging $\pm 10^\circ\text{C}$ relative to baseline.
- **Residual Component**: Random stationary noise centered near zero with constant variance.
