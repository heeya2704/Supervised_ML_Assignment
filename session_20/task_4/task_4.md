# Session 20 - Task 4: Moving Average Smoothing Analysis (Window = 7 Days)

## Task Overview
Apply a 7-day Simple Moving Average (SMA) filter to smooth high-frequency day-to-day fluctuations in the temperature time series and overlay the original vs smoothed trajectories.

---

## Mathematical Formula

$$\text{SMA}_{t, k} = \frac{1}{k} \sum_{i=0}^{k-1} Y_{t-i}$$

Where $k = 7$ days.

---

## Comparison Plot
The comparison chart is saved to [`moving_average_plot.png`](moving_average_plot.png).

- **Noise Suppression**: High-frequency day-to-day thermal spikes (random measurement noise) are effectively filtered out.
- **Trend Highlight**: The 7-day rolling window preserves weekly macro trends while rendering seasonal inflection points far clearer.
