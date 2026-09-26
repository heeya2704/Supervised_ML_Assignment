"""
Session 9 - Task 2: Lasso Regression (L1 Regularization) and Feature Coefficient Inspection
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

def main():
    print("=" * 70)
    print("SESSION 9 - TASK 2: Lasso Regression (L1 Regularization)")
    print("=" * 70)

    csv_path = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'zomato_ratings.csv')
    df = pd.read_csv(csv_path)

    X = df.drop(columns=['rating'])
    y = df['rating']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features before applying Lasso
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Fit Lasso Regression (alpha=0.03)
    lasso = Lasso(alpha=0.03, random_state=42)
    lasso.fit(X_train_scaled, y_train)

    print(f"\nModel Intercept (b): {lasso.intercept_:.4f}")
    print("\nLasso Coefficients (L1 Regularization, alpha=0.03):")
    print("-" * 55)
    print(f"{'Feature Name':<25} | {'Scaled Coefficient':<20}")
    print("-" * 55)

    zero_count = 0
    for name, coef in zip(X.columns, lasso.coef_):
        status = " (ZEROED OUT)" if abs(coef) < 1e-5 else ""
        if abs(coef) < 1e-5:
            zero_count += 1
        print(f"{name:<25} | {coef:<20.6f}{status}")

    print("-" * 55)
    print(f"Total Features Evaluated: {len(X.columns)}")
    print(f"Features Set Exactly to Zero: {zero_count}")
    print(f"Non-Zero Features Retained: {len(X.columns) - zero_count}")
    print("=" * 70)

if __name__ == "__main__":
    main()
