"""
Session 21 - Task 3: RandomForestRegressor Time Series Temperature Forecasting
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def main():
    print("=" * 75)
    print("SESSION 21 - TASK 3: RandomForest Time Series Forecasting Model")
    print("=" * 75)

    # 1. Load dataset & engineer lag/rolling features
    df = pd.read_csv("session_21/dataset/ahmedabad_temperature.csv", parse_dates=["date"])
    df["temp_lag1"] = df["temperature_celsius"].shift(1)
    df["temp_lag2"] = df["temperature_celsius"].shift(2)
    df["temp_lag3"] = df["temperature_celsius"].shift(3)
    df["temp_roll7"] = df["temperature_celsius"].rolling(window=7).mean()

    df_clean = df.dropna().reset_index(drop=True)

    # Features and Target
    feature_cols = ["temp_lag1", "temp_lag2", "temp_lag3", "temp_roll7"]
    X = df_clean[feature_cols]
    y = df_clean["temperature_celsius"]

    # Time-based Train-Test Split (80% Train, 20% Test - no shuffle)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    # 2. Train Random Forest Regressor
    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_regressor.fit(X_train, y_train)

    # 3. Forecast Next Available Day Temperature
    latest_row = df.tail(7)
    next_date = df["date"].max() + pd.Timedelta(days=1)
    
    next_features = pd.DataFrame([{
        "temp_lag1": df["temperature_celsius"].iloc[-1],
        "temp_lag2": df["temperature_celsius"].iloc[-2],
        "temp_lag3": df["temperature_celsius"].iloc[-3],
        "temp_roll7": df["temperature_celsius"].tail(7).mean()
    }])

    next_temp_pred = rf_regressor.predict(next_features)[0]

    print(f"\n1. MODEL TRAINING SUMMARY:")
    print(f"   • Training Samples : {len(X_train)}")
    print(f"   • Testing Samples  : {len(X_test)}")
    print(f"   • Features Used    : {feature_cols}")

    print(f"\n2. NEXT-DAY OUT-OF-SAMPLE FORECAST:")
    print("-" * 65)
    print(f"   • Forecast Date              : {next_date.strftime('%Y-%m-%d (%A)')}")
    print(f"   • Predicted Temperature (°C) : {next_temp_pred:.2f}°C")
    print("-" * 65)
    print("=" * 75)

if __name__ == "__main__":
    main()
