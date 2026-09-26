# Session 21 - Task 3: RandomForestRegressor Time Series Temperature Forecast

## Task Overview
Train a `RandomForestRegressor` model using engineered lag (`temp_lag1`, `temp_lag2`, `temp_lag3`) and rolling average (`temp_roll7`) features to predict next-day temperature.

---

## Model Architecture & Feature Input

- **Regressor**: `RandomForestRegressor(n_estimators=100, random_state=42)`
- **Predictors**: `['temp_lag1', 'temp_lag2', 'temp_lag3', 'temp_roll7']`
- **Target Variable**: Next-day actual temperature ($\text{Temp}_{t}$)

---

## Prediction Output

- **Target Forecast Date**: **2026-01-01**
- **Predicted Temperature**: **18.72°C**
