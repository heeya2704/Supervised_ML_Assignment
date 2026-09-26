"""
Session 21 - Task 1: Lag & Rolling Feature Engineering for Temperature Forecasting
"""

import pandas as pd

def main():
    print("=" * 75)
    print("SESSION 21 - TASK 1: Feature Engineering (3-Day Lag & 7-Day Rolling Mean)")
    print("=" * 75)

    # 1. Load dataset
    df = pd.read_csv("session_21/dataset/ahmedabad_temperature.csv", parse_dates=["date"])

    # 2. Engineer Lag & Rolling Average Features
    df["temp_lag3"] = df["temperature_celsius"].shift(3)
    df["temp_roll7"] = df["temperature_celsius"].rolling(window=7).mean()

    print(f"\n1. FEATURE CREATION SUMMARY:")
    print(f"   • Original Temperature Column : 'temperature_celsius'")
    print(f"   • Created Feature 1          : 'temp_lag3' (3-day lag shift)")
    print(f"   • Created Feature 2          : 'temp_roll7' (7-day rolling window average)")

    # Drop NaN values resulting from shift and rolling window
    df_clean = df.dropna().reset_index(drop=True)

    print(f"\n2. SAMPLE FEATURE MATRIX (First 10 Clean Rows):")
    print("-" * 65)
    print(df_clean[["date", "temperature_celsius", "temp_lag3", "temp_roll7"]].head(10).to_string(index=False))
    print("-" * 65)
    print("=" * 75)

if __name__ == "__main__":
    main()
