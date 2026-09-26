# Session 20 - Task 3: Augmented Dickey-Fuller (ADF) Stationarity Test

## Task Overview
Execute the Augmented Dickey-Fuller (ADF) test using `statsmodels.tsa.stattools.adfuller` to assess whether the daily temperature time series is stationary or requires differencing.

---

## Hypothesis Framework

- **Null Hypothesis ($H_0$)**: The time series possesses a unit root (is non-stationary).
- **Alternative Hypothesis ($H_1$)**: The time series is stationary (no unit root).

---

## Empirical Test Results

| Series Version | ADF Statistic | $p$-value | Critical Value ($5\%$) | Stationarity Verdict |
| :--- | :---: | :---: | :---: | :---: |
| **Raw Original Series** | **-2.6738** | **0.0787** | **-2.8656** | **Non-Stationary ($p \ge 0.05$)** |
| **First Difference ($d=1$)** | **-14.2185** | **$1.68 \times 10^{-26}$** | **-2.8656** | **Stationary ($p < 0.05$)** |

---

## Key Takeaway
Because the raw series $p$-value ($0.0787$) exceeds $0.05$, the original time series is non-stationary due to strong annual seasonality and trend. Taking the first difference ($d=1$) yields strong stationarity ($p < 0.0001$).
