"""
Session 3 - Task 4
Apply one-hot encoding on 'venue' column using pandas get_dummies().
"""

import os
import pandas as pd

def main():
    print("=" * 70)
    print("SESSION 3 - TASK 4: One-Hot Encoding on 'venue' using get_dummies()")
    print("=" * 70)

    csv_path = os.path.join(os.path.dirname(__file__), "..", "dataset", "ipl_player_stats.csv")
    df = pd.read_csv(csv_path)

    print("\nOriginal 'venue' column sample:")
    print(df[['player_name', 'venue']].head())

    # Apply one-hot encoding on 'venue' column
    df_encoded = pd.get_dummies(df, columns=['venue'], prefix='venue', dtype=int)

    venue_cols = [col for col in df_encoded.columns if col.startswith('venue_')]

    print("\nOne-Hot Encoded Venue Columns:")
    for col in venue_cols:
        print(f"  - {col}")

    print("\nResulting DataFrame with One-Hot Encoded Venue Columns (First 8 rows):")
    print("-" * 70)
    display_cols = ['player_name'] + venue_cols
    print(df_encoded[display_cols].head(8))
    print("=" * 70)

if __name__ == "__main__":
    main()
