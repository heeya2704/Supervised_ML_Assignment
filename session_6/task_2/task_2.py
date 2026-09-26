"""
Session 6 - Task 2
Display regression coefficients and intercept with domain interpretations for mobile phone pricing.
"""

import os
import pandas as pd
from sklearn.linear_model import LinearRegression

def main():
    print("=" * 70)
    print("SESSION 6 - TASK 2: Regression Coefficient Interpretation")
    print("=" * 70)

    dataset_path = os.path.join(os.path.dirname(__file__), "..", "dataset", "mobile_prices.csv")
    df = pd.read_csv(dataset_path)

    feature_cols = ['ram_gb', 'storage_gb', 'battery_mah', 'camera_mp', 'screen_size_inch']
    X = df[feature_cols]
    y = df['price_inr']

    model = LinearRegression()
    model.fit(X, y)

    print(f"\nModel Intercept (b): Rs. {model.intercept_:.2f}")
    print("\nFeature Coefficient Breakdown & Real-World Meaning:")
    print("-" * 70)

    comments = {
        'ram_gb': "Each additional 1 GB of RAM increases predicted phone price by Rs. 3,647.20.",
        'storage_gb': "Each additional 1 GB of storage capacity increases predicted price by Rs. 116.60.",
        'battery_mah': "Each additional 1 mAh of battery capacity adds approximately Rs. 4.61 to price.",
        'camera_mp': "Each additional 1 Megapixel of camera resolution increases price by Rs. 173.69.",
        'screen_size_inch': "Each additional 1 inch of display screen size adds Rs. 1,255.54 to price."
    }

    for col, coef in zip(feature_cols, model.coef_):
        print(f"Feature: {col:<18} | Coef: Rs. {coef:>9.2f} / unit")
        print(f"  --> Interpretation: {comments[col]}")
        print("-" * 70)

if __name__ == "__main__":
    main()
