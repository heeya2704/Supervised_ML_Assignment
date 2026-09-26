"""
Session 3 - Task 3
Replace missing values in 'team' column with constant 'Unknown' and print first 10 rows.
"""

import os
import pandas as pd

def main():
    print("=" * 70)
    print("SESSION 3 - TASK 3: Constant Imputation ('Unknown') for Categorical 'team'")
    print("=" * 70)

    csv_path = os.path.join(os.path.dirname(__file__), "..", "dataset", "ipl_player_stats.csv")
    df = pd.read_csv(csv_path)

    print("\nMissing values in 'team' before imputation:", df['team'].isnull().sum())

    # Impute missing team values with constant 'Unknown'
    df['team'] = df['team'].fillna('Unknown')

    print("\nMissing values in 'team' after imputation:", df['team'].isnull().sum())
    print("\nFirst 10 rows of updated DataFrame:")
    print("-" * 70)
    print(df[['player_id', 'player_name', 'team', 'player_role', 'venue']].head(10))
    print("=" * 70)

if __name__ == "__main__":
    main()
