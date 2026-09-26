"""
Session 4 - Task 2
Apply MinMaxScaler on Flipkart product features and print min/max before & after scaling.
"""

import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def main():
    print("=" * 70)
    print("SESSION 4 - TASK 2: MinMaxScaler on Flipkart Product Features")
    print("=" * 70)

    csv_path = os.path.join(os.path.dirname(__file__), "..", "dataset", "flipkart_products.csv")
    df = pd.read_csv(csv_path)

    features = ['price', 'rating', 'number_of_reviews', 'discount']

    print("\nBEFORE SCALING (Raw Data Ranges):")
    print("-" * 65)
    print(f"{'Feature Name':<22} | {'Min Value':<15} | {'Max Value':<15}")
    print("-" * 65)
    for col in features:
        print(f"{col:<22} | {df[col].min():<15.2f} | {df[col].max():<15.2f}")
    print("-" * 65)

    # Initialize MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = df.copy()
    df_scaled[features] = scaler.fit_transform(df[features])

    print("\nAFTER SCALING (MinMaxScaler Scaled to [0, 1]):")
    print("-" * 65)
    print(f"{'Feature Name':<22} | {'Scaled Min':<15} | {'Scaled Max':<15}")
    print("-" * 65)
    for col in features:
        print(f"{col:<22} | {df_scaled[col].min():<15.4f} | {df_scaled[col].max():<15.4f}")
    print("-" * 65)

    print("\nSample Scaled Rows (First 5 Products):")
    print(df_scaled[['product_id'] + features].head())
    print("=" * 70)

if __name__ == "__main__":
    main()
