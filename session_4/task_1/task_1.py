"""
Session 4 - Task 1
Apply StandardScaler on numeric features of IPL dataset and save to ipl_scaled.csv.
"""

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

def main():
    print("=" * 70)
    print("SESSION 4 - TASK 1: StandardScaler on IPL Player Stats")
    print("=" * 70)

    csv_path = os.path.join(os.path.dirname(__file__), "..", "dataset", "ipl_stats.csv")
    df = pd.read_csv(csv_path)

    numeric_cols = ['runs', 'wickets', 'matches', 'strike_rate']
    print(f"\nOriginal Data Head:\n{df.head()}")

    # Initialize StandardScaler
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(df[numeric_cols])

    df_scaled = df.copy()
    df_scaled[numeric_cols] = scaled_array

    # Save to ipl_scaled.csv inside task_1 folder and dataset folder
    out_path_task = os.path.join(os.path.dirname(__file__), "ipl_scaled.csv")
    out_path_dataset = os.path.join(os.path.dirname(__file__), "..", "dataset", "ipl_scaled.csv")

    df_scaled.to_csv(out_path_task, index=False)
    df_scaled.to_csv(out_path_dataset, index=False)

    print(f"\n[+] Scaled data saved successfully to: {out_path_task}")
    print("\nScaled Data Head (Mean = 0, Std = 1):")
    print("-" * 70)
    print(df_scaled.head())
    print("-" * 70)
    print("Verification - Scaled Mean & Std:")
    for col in numeric_cols:
        print(f"  {col:<12} -> Mean: {df_scaled[col].mean():.4f}, Std: {df_scaled[col].std(ddof=0):.4f}")
    print("=" * 70)

if __name__ == "__main__":
    main()
