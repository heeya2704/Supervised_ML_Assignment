"""
Session 4 - Task 5
Outlier detection and removal using Interquartile Range (IQR) method on BookMyShow user_ratings.
"""

import os
import pandas as pd
import numpy as np

def main():
    print("=" * 70)
    print("SESSION 4 - TASK 5: IQR Outlier Detection on BookMyShow User Ratings")
    print("=" * 70)

    csv_path = os.path.join(os.path.dirname(__file__), "..", "dataset", "bookmyshow_ratings.csv")
    df = pd.read_csv(csv_path)

    initial_row_count = len(df)
    print(f"\nInitial Dataset Row Count: {initial_row_count} rows")

    ratings = df['user_ratings']

    # Step 1: Calculate Q1, Q3, and IQR
    Q1 = ratings.quantile(0.25)
    Q3 = ratings.quantile(0.75)
    IQR = Q3 - Q1

    # Step 2: Compute lower and upper bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    print(f"\nIQR Outlier Threshold Calculation:")
    print(f"  - 25th Percentile (Q1) : {Q1:.4f}")
    print(f"  - 75th Percentile (Q3) : {Q3:.4f}")
    print(f"  - IQR (Q3 - Q1)        : {IQR:.4f}")
    print(f"  - Lower Bound (Q1 - 1.5*IQR): {lower_bound:.4f}")
    print(f"  - Upper Bound (Q3 + 1.5*IQR): {upper_bound:.4f}")

    # Step 3: Identify outliers
    outliers_df = df[(df['user_ratings'] < lower_bound) | (df['user_ratings'] > upper_bound)]
    clean_df = df[(df['user_ratings'] >= lower_bound) & (df['user_ratings'] <= upper_bound)]

    rows_dropped = len(outliers_df)
    final_row_count = len(clean_df)

    print("\nIdentified Outlier Rows:")
    print("-" * 50)
    print(outliers_df[['movie_id', 'movie_name', 'user_ratings']])
    print("-" * 50)

    print("\n" + "=" * 70)
    print(f"OUTLIER REMOVAL SUMMARY:")
    print(f"  - Initial Rows : {initial_row_count}")
    print(f"  - Outliers Dropped : {rows_dropped} rows")
    print(f"  - Clean Rows Remaining : {final_row_count} rows")
    print("=" * 70)

if __name__ == "__main__":
    main()
