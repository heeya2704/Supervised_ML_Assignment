"""
Session 20 - Task 4: Moving Average Smoothing (Window Size = 7 Days)
"""

import pandas as pd
import matplotlib.pyplot as plt

def main():
    print("=" * 75)
    print("SESSION 20 - TASK 4: 7-Day Moving Average Smoothing")
    print("=" * 75)

    # 1. Load dataset
    df = pd.read_csv("session_20/dataset/ahmedabad_temperature.csv", parse_dates=["date"], index_col="date")

    # 2. Compute 7-day Moving Average Smoothing
    df["temp_sma7"] = df["temperature_celsius"].rolling(window=7).mean()

    print(f"\n1. MOVING AVERAGE SUMMARY:")
    print(f"   • Original Series Mean : {df['temperature_celsius'].mean():.2f}°C")
    print(f"   • Smoothed Series Mean : {df['temp_sma7'].dropna().mean():.2f}°C")
    print(f"   • Window Size          : 7 Days (Weekly Smoothing)")

    # 3. Plot Original vs Smoothed Time Series
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["temperature_celsius"], color="#aec7e8", lw=1.0, alpha=0.8, label="Original Daily Temperature (°C)")
    plt.plot(df.index, df["temp_sma7"], color="#1f77b4", lw=2.2, label="7-Day Moving Average (Smoothed)")

    plt.title("Ahmedabad Daily Temperature - Original vs 7-Day Moving Average", fontsize=14, fontweight="bold")
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Temperature (°C)", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right", fontsize=11)
    plt.tight_layout()

    output_img = "session_20/task_4/moving_average_plot.png"
    plt.savefig(output_img, dpi=300)
    plt.close()

    print(f"\n2. Plot saved successfully to: {output_img}")
    print("=" * 75)

if __name__ == "__main__":
    main()
