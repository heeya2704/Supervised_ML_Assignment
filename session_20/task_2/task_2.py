"""
Session 20 - Task 2: Seasonal Decomposition of Temperature Time Series
"""

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

def main():
    print("=" * 75)
    print("SESSION 20 - TASK 2: Time Series Decomposition (Trend, Season, Residual)")
    print("=" * 75)

    # 1. Load dataset
    df = pd.read_csv("session_20/dataset/ahmedabad_temperature.csv", parse_dates=["date"], index_col="date")

    # 2. Perform Classical Additive Seasonal Decomposition (period = 365 days for annual cycle)
    decomposition = seasonal_decompose(df["temperature_celsius"], model="additive", period=365)

    # 3. Plot Decomposition Components
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(df.index, decomposition.observed, color="#1f77b4", lw=1.2)
    axes[0].set_ylabel("Observed (°C)", fontsize=11)
    axes[0].set_title("Time Series Classical Additive Decomposition - Ahmedabad", fontsize=14, fontweight="bold")
    axes[0].grid(True, linestyle=":", alpha=0.6)

    axes[1].plot(df.index, decomposition.trend, color="#ff7f0e", lw=2.0)
    axes[1].set_ylabel("Trend (°C)", fontsize=11)
    axes[1].grid(True, linestyle=":", alpha=0.6)

    axes[2].plot(df.index, decomposition.seasonal, color="#2ca02c", lw=1.2)
    axes[2].set_ylabel("Seasonal (°C)", fontsize=11)
    axes[2].grid(True, linestyle=":", alpha=0.6)

    axes[3].scatter(df.index, decomposition.resid, color="#d62728", s=6, alpha=0.7)
    axes[3].axhline(0, color="black", linestyle="--", lw=1)
    axes[3].set_ylabel("Residuals (°C)", fontsize=11)
    axes[3].set_xlabel("Date", fontsize=12)
    axes[3].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    output_img = "session_20/task_2/seasonal_decomposition.png"
    plt.savefig(output_img, dpi=300)
    plt.close()

    print(f"\n1. DECOMPOSITION SUMMARY:")
    print("   • Model Type : Additive (Observed = Trend + Seasonal + Residual)")
    print("   • Seasonality Period : 365 Days")
    print(f"\n2. Plot saved successfully to: {output_img}")
    print("=" * 75)

if __name__ == "__main__":
    main()
