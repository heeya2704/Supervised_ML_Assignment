"""
Session 21 - Task 5: Symmetric Mean Absolute Percentage Error (SMAPE) Implementation
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def calculate_smape(y_true, y_pred):
    """
    Calculates Symmetric Mean Absolute Percentage Error (SMAPE).
    
    Formula:
        SMAPE = (100% / n) * sum( |y_true - y_pred| / ((|y_true| + |y_pred|) / 2) )
        
    Parameters:
        y_true (array-like): Ground truth target values
        y_pred (array-like): Model predicted target values
        
    Returns:
        float: Percentage SMAPE score bounded between 0% and 200%
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    
    # Avoid zero division when both true and pred are 0
    smape_elements = np.where(denominator == 0, 0, numerator / denominator)
    return np.mean(smape_elements) * 100.0

def main():
    print("=" * 75)
    print("SESSION 21 - TASK 5: SMAPE Metric Calculation & AI Learning Analysis")
    print("=" * 75)

    # 1. Load dataset & fit model
    df = pd.read_csv("session_21/dataset/ahmedabad_temperature.csv", parse_dates=["date"])
    df["temp_lag1"] = df["temperature_celsius"].shift(1)
    df["temp_lag2"] = df["temperature_celsius"].shift(2)
    df["temp_roll7"] = df["temperature_celsius"].rolling(window=7).mean()

    df_clean = df.dropna().reset_index(drop=True)
    X = df_clean[["temp_lag1", "temp_lag2", "temp_roll7"]]
    y = df_clean["temperature_celsius"]

    train_size = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    # 2. Compute SMAPE
    smape_score = calculate_smape(y_test, y_pred)
    mae_score = mean_absolute_error(y_test, y_pred)

    print(f"\n1. COMPUTED SMAPE RESULTS:")
    print("-" * 65)
    print(f"   • Calculated SMAPE Score = {smape_score:.4f}%")
    print(f"   • Baseline MAE Score     = {mae_score:.4f}°C")
    print("-" * 65)

    print("\n2. KEY TAKEAWAY LEARNED FROM AI EXPLANATION:")
    print("Unlike standard MAPE (which is asymmetric and penalizes over-forecasting far more severely than under-forecasting and fails when actual value is zero), SMAPE normalizes residuals by the average of actual and predicted absolute values, bounding error between 0% and 200% and providing fair, symmetric evaluation across all time steps.")
    print("=" * 75)

if __name__ == "__main__":
    main()
