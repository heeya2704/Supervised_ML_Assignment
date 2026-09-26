"""
Session 6 - Task 4
Standardized Coefficients & Feature Importance Analysis for Mobile Phone Prices.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

def main():
    print("=" * 70)
    print("SESSION 6 - TASK 4: Feature Importance & Standardized Coefficients")
    print("=" * 70)

    dataset_path = os.path.join(os.path.dirname(__file__), "..", "dataset", "mobile_prices.csv")
    df = pd.read_csv(dataset_path)

    feature_cols = ['ram_gb', 'storage_gb', 'battery_mah', 'camera_mp', 'screen_size_inch']
    X = df[feature_cols]
    y = df['price_inr']

    # Fit Standardized Linear Regression to compare absolute importance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model_std = LinearRegression()
    model_std.fit(X_scaled, y)

    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Standardized Coef': model_std.coef_,
        'Abs Importance': np.abs(model_std.coef_)
    }).sort_values(by='Abs Importance', ascending=False).reset_index(drop=True)

    print("\nSTANDARDIZED FEATURE IMPORTANCE RANKING:")
    print("-" * 65)
    print(f"{'Rank':<5} | {'Feature':<20} | {'Standardized Coef':<20} | {'Impact':<10}")
    print("-" * 65)
    for rank, row in importance_df.iterrows():
        print(f"{rank+1:<5} | {row['Feature']:<20} | {row['Standardized Coef']:<20.2f} | Rs. {row['Abs Importance']:>8.2f}")
    print("-" * 65)

    most_influential = importance_df.iloc[0]['Feature']
    print(f"\nMOST INFLUENTIAL FEATURE: {most_influential.upper()}")
    print("\nREASONING (2-3 Sentences):")
    print(f"Based on the standardized regression coefficients, '{most_influential}' exhibits the largest absolute standardized magnitude.")
    print("This indicates that a one standard deviation change in RAM/storage creates the largest direct swing in total mobile phone retail price.")
    print("Consumers and manufacturers recognize high memory tiers as the primary driver of flagship vs budget price segmentation.")
    print("=" * 70)

if __name__ == "__main__":
    main()
