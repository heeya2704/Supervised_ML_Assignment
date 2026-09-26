"""
Session 6 - Task 1
Fit Multiple Linear Regression model to predict mobile phone prices using RAM, storage, battery, camera MP, and screen size.
"""

import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error

def main():
    print("=" * 70)
    print("SESSION 6 - TASK 1: Multiple Linear Regression Model for Mobile Prices")
    print("=" * 70)

    dataset_path = os.path.join(os.path.dirname(__file__), "..", "dataset", "mobile_prices.csv")
    df = pd.read_csv(dataset_path)

    print("\nDataset Overview (First 5 Rows):")
    print("-" * 65)
    print(df.head())
    print("-" * 65)

    # Features and Target
    feature_cols = ['ram_gb', 'storage_gb', 'battery_mah', 'camera_mp', 'screen_size_inch']
    X = df[feature_cols]
    y = df['price_inr']

    # Train Multiple Linear Regression
    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = root_mean_squared_error(y, y_pred)

    print("\nMULTIPLE LINEAR REGRESSION SUMMARY:")
    print("-" * 50)
    print(f"Intercept (b)      : Rs. {model.intercept_:.2f}")
    print("Feature Coefficients:")
    for col, coef in zip(feature_cols, model.coef_):
        print(f"  - {col:<18} : Rs. {coef:>10.2f}")
    print("-" * 50)
    print(f"R^2 Score          : {r2:.4f}")
    print(f"RMSE               : Rs. {rmse:.2f}")
    print("=" * 70)

if __name__ == "__main__":
    main()
