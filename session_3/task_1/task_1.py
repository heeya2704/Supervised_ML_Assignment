"""
Session 3 - Task 1
Load IPL player stats dataset and count missing values per column.
"""

import os
import pandas as pd

def main():
    print("=" * 70)
    print("SESSION 3 - TASK 1: Missing Values Identification in IPL Dataset")
    print("=" * 70)

    csv_path = os.path.join(os.path.dirname(__file__), "..", "dataset", "ipl_player_stats.csv")
    df = pd.read_csv(csv_path)

    print(f"\nDataset Dimensions: {df.shape[0]} rows x {df.shape[1]} columns\n")
    
    missing_counts = df.isnull().sum()
    
    print("Number of Missing (NaN) Values Per Column:")
    print("-" * 50)
    for col, count in missing_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {col:<15} : {count:>2} missing ({percentage:>5.1f}%)")
    print("-" * 50)
    print(f"Total Missing Values across DataFrame: {df.isnull().sum().sum()}")
    print("=" * 70)

if __name__ == "__main__":
    main()
