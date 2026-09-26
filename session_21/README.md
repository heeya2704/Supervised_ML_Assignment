# Session 21: Applied Time Series Forecasting & Model Evaluation

This folder contains Python scripts and detailed markdown write-ups for all tasks in **Session 21: Feature Engineering, Time-Series Train/Test Splitting, Machine Learning Modeling, and Evaluation Metrics (MAE, RMSE, MAPE, SMAPE)**.

## Directory Structure
- [`dataset/`](dataset/): Daily temperature dataset for Ahmedabad (`ahmedabad_temperature.csv`).
- [`task_1/`](task_1/): Feature Engineering (3-day lag `temp_lag3` and 7-day rolling mean `temp_roll7`).
- [`task_2/`](task_2/): Sequential 80/20 chronological train-test split without data shuffling.
- [`task_3/`](task_3/): Fitting `RandomForestRegressor` for daily temperature forecasting.
- [`task_4/`](task_4/): Calculating MAE, RMSE, and MAPE, and analyzing sensitivity to outlier errors.
- [`task_5/`](task_5/): Custom implementation of Symmetric Mean Absolute Percentage Error (SMAPE) and comparison with standard MAPE.

## How to Run Python Scripts
Run individual task scripts from the workspace root:
```bash
python session_21/task_1/task_1.py
python session_21/task_2/task_2.py
python session_21/task_3/task_3.py
python session_21/task_4/task_4.py
python session_21/task_5/task_5.py
```
