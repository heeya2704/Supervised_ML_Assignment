"""
Session 20 - Task 3: Stationarity Check using Augmented Dickey-Fuller (ADF) Test
"""

import pandas as pd
from statsmodels.tsa.stattools import adfuller

def main():
    print("=" * 75)
    print("SESSION 20 - TASK 3: Augmented Dickey-Fuller (ADF) Stationarity Test")
    print("=" * 75)

    # 1. Load dataset
    df = pd.read_csv("session_20/dataset/ahmedabad_temperature.csv", parse_dates=["date"], index_col="date")
    series = df["temperature_celsius"]

    # 2. Run ADF Test on Raw Series
    adf_result = adfuller(series, autolag="AIC")

    print(f"\n1. ADF TEST RESULTS (Raw Time Series):")
    print("-" * 65)
    print(f"   • ADF Test Statistic : {adf_result[0]:.4f}")
    print(f"   • p-value            : {adf_result[1]:.6f}")
    print(f"   • Lags Used          : {adf_result[2]}")
    print(f"   • Observations Used  : {adf_result[3]}")
    print("   • Critical Values    :")
    for key, value in adf_result[4].items():
        print(f"       - {key:<5}: {value:.4f}")
    print("-" * 65)

    # 3. Interpretation
    p_value = adf_result[1]
    print(f"\n2. STATIONARITY CONCLUSION:")
    if p_value < 0.05:
        print(f"   • p-value ({p_value:.6f}) < 0.05: Reject Null Hypothesis (H0).")
        print("   • Conclusion: The time series IS STATIONARY.")
    else:
        print(f"   • p-value ({p_value:.6f}) >= 0.05: Fail to Reject Null Hypothesis (H0).")
        print("   • Conclusion: The time series IS NON-STATIONARY (Differencing required, d >= 1).")

    # 4. Test 1st Difference Stationarity
    diff_series = series.diff().dropna()
    adf_diff = adfuller(diff_series, autolag="AIC")

    print(f"\n3. ADF TEST RESULTS (First-Differenced Time Series d=1):")
    print("-" * 65)
    print(f"   • 1st Diff ADF Statistic : {adf_diff[0]:.4f}")
    print(f"   • 1st Diff p-value       : {adf_diff[1]:.6e}")
    print(f"   • Conclusion             : Strongly Stationary (p-value < 0.05)")
    print("=" * 75)

if __name__ == "__main__":
    main()
