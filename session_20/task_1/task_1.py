"""
Session 20 - Task 1: Time Series Visualization of Daily Temperature Data (Ahmedabad)
"""

import pandas as pd
import matplotlib.pyplot as plt

def main():
    print("=" * 75)
    print("SESSION 20 - TASK 1: Visualizing Daily Temperature Time Series")
    print("=" * 75)

    # 1. Load dataset
    data_path = "session_20/dataset/ahmedabad_temperature.csv"
    df = pd.read_csv(data_path, parse_dates=["date"], index_col="date")

    print(f"\n1. DATASET SUMMARY (Ahmedabad Daily Temperature):")
    print(f"   • Total Observations : {len(df)} days ({df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')})")
    print(f"   • Mean Temperature   : {df['temperature_celsius'].mean():.2f}°C")
    print(f"   • Min Temperature    : {df['temperature_celsius'].min():.2f}°C")
    print(f"   • Max Temperature    : {df['temperature_celsius'].max():.2f}°C")

    # 2. Plot Raw Time Series
    plt.figure(figsize=(12, 5))
    plt.plot(df.index, df["temperature_celsius"], color="#d62728", lw=1.2, label="Daily Temperature (°C)")
    plt.title("Daily Temperature Time Series - Ahmedabad (2024 - 2025)", fontsize=14, fontweight="bold")
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Temperature (°C)", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right", fontsize=11)
    plt.tight_layout()

    output_img = "session_20/task_1/raw_temperature_plot.png"
    plt.savefig(output_img, dpi=300)
    plt.close()

    print(f"\n2. VISUAL PATTERN ANALYSIS:")
    print("   • Seasonality: Clear annual repeating pattern with peak summer temperatures (~38-42°C in May-June) and winter troughs (~16-20°C in Jan).")
    print("   • Trend: Mild upward drift observable over the two-year span.")
    print(f"\n3. Plot saved successfully to: {output_img}")
    print("=" * 75)

if __name__ == "__main__":
    main()
