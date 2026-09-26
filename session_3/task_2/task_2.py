"""
Session 3 - Task 2
Impute missing values in 'player_age' with median age using fillna().
"""

import os
import pandas as pd

def main():
    print("=" * 70)
    print("SESSION 3 - TASK 2: Median Imputation for 'player_age'")
    print("=" * 70)

    csv_path = os.path.join(os.path.dirname(__file__), "..", "dataset", "ipl_player_stats.csv")
    df = pd.read_csv(csv_path)

    print("\nOriginal 'player_age' column (with NaNs):")
    print(df[['player_name', 'player_age']])

    # Compute median age ignoring NaNs
    median_age = df['player_age'].median()
    print(f"\nCalculated Median Player Age: {median_age:.1f} years\n")

    # Impute missing values with median
    df['player_age'] = df['player_age'].fillna(median_age)

    print("Updated 'player_age' column (after fillna with median):")
    print("-" * 50)
    print(df[['player_name', 'player_age']])
    print("-" * 50)
    print(f"Remaining missing values in 'player_age': {df['player_age'].isnull().sum()}")
    print("=" * 70)

if __name__ == "__main__":
    main()
