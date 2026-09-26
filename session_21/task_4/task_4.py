"""
Session 21 - Task 4: Time Series Regression Metrics (MAE, RMSE, MAPE)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

def mean_absolute_percentage_error_custom(y_true, y_pred):
    """Calculates Mean Absolute Percentage Error (MAPE)"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def main():
    print("=" * 75)
    print("SESSION 21 - TASK 4: Time Series Regression Evaluation Metrics")
    print("=" * 75)

    # 1. Load dataset & create features
    df = pd.read_csv("session_21/dataset/ahmedabad_temperature.csv", parse_dates=["date"])
    df["temp_lag1"] = df["temperature_celsius"].shift(1)
    df["temp_lag2"] = df["temperature_celsius"].shift(2)
    df["temp_lag3"] = df["temperature_celsius"].shift(3)
    df["temp_roll7"] = df["temperature_celsius"].rolling(window=7).mean()

    df_clean = df.dropna().reset_index(drop=True)
    X = df_clean[["temp_lag1", "temp_lag2", "temp_lag3", "temp_roll7"]]
    y = df_clean["temperature_celsius"]

    # Time-series Split
    train_size = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    # Train model
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)

    # 2. Calculate Evaluation Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = mean_absolute_percentage_error_custom(y_test, y_pred)

    print(f"\n1. COMPUTED REGRESSION METRICS:")
    print("-" * 65)
    print(f"   • Mean Absolute Error (MAE)            = {mae:.4f}°C")
    print(f"   • Root Mean Squared Error (RMSE)       = {rmse:.4f}°C")
    print(f"   • Mean Absolute Percentage Error (MAPE)= {mape:.2f}% ({mape/100:.4f})")
    print("-" * 65)

    print("\nONE-LINE EXPLANATION OF SENSITIVITY TO LARGE ERRORS:")
    print("Root Mean Squared Error (RMSE) is the most sensitive metric to large prediction errors because squaring individual residuals heavily penalizes extreme outliers compared to linear metrics like MAE.")
    print("=" * 75)

if __name__ == "__main__":
    main()
