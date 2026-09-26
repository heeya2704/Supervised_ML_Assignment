# Session 20: Time Series Analysis & Forecasting Foundations

This folder contains Python scripts, visualization plots, and markdown write-ups for all tasks in **Session 20: Classical Time Series Decomposition, Stationarity Testing (ADF), Moving Average Smoothing, and ARIMA Forecasting**.

## Directory Structure
- [`dataset/`](dataset/): Daily temperature dataset for Ahmedabad (`ahmedabad_temperature.csv`).
- [`task_1/`](task_1/): Visualizing raw daily temperature time series and identifying trend and annual seasonality.
- [`task_2/`](task_2/): Classical additive time series decomposition (`Observed = Trend + Seasonal + Residual`).
- [`task_3/`](task_3/): Stationarity evaluation using Augmented Dickey-Fuller (ADF) test ($d=1$ required).
- [`task_4/`](task_4/): 7-Day Simple Moving Average (SMA) smoothing filter.
- [`task_5/`](task_5/): Fitting $\text{ARIMA}(2,1,2)$ model and generating 7-day out-of-sample temperature forecast with $95\%$ confidence interval bounds.

## How to Run Python Scripts
Run individual task scripts from the workspace root:
```bash
python session_20/task_1/task_1.py
python session_20/task_2/task_2.py
python session_20/task_3/task_3.py
python session_20/task_4/task_4.py
python session_20/task_5/task_5.py
```
