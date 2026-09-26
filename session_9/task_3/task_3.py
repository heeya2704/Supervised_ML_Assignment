"""
Session 9 - Task 3: Ridge vs Lasso Coefficient Comparison
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge

def main():
    print("=" * 75)
    print("SESSION 9 - TASK 3: Ridge (L2) vs Lasso (L1) Coefficient Comparison")
    print("=" * 75)

    csv_path = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'zomato_ratings.csv')
    df = pd.read_csv(csv_path)

    X = df.drop(columns=['rating'])
    y = df['rating']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Fit Lasso & Ridge models with comparable alpha=10.0 for Ridge / alpha=0.03 for Lasso
    lasso = Lasso(alpha=0.03, random_state=42)
    lasso.fit(X_train_scaled, y_train)

    ridge = Ridge(alpha=10.0, random_state=42)
    ridge.fit(X_train_scaled, y_train)

    print(f"\n{'Feature Name':<22} | {'Lasso (L1) Coef':<18} | {'Ridge (L2) Coef':<18} | {'Key Observation':<20}")
    print("-" * 85)

    for name, l_coef, r_coef in zip(X.columns, lasso.coef_, ridge.coef_):
        if abs(l_coef) < 1e-5 and abs(r_coef) > 1e-5:
            obs = "Lasso=0, Ridge!=0"
        elif abs(l_coef) < 1e-5 and abs(r_coef) < 1e-5:
            obs = "Both Near Zero"
        else:
            obs = "Both Active"
        print(f"{name:<22} | {l_coef:<18.6f} | {r_coef:<18.6f} | {obs:<20}")

    print("-" * 85)
    print("\nSUMMARY OF DIFFERENCES:")
    print("1. Lasso (L1 Penalty) enforces sparsity by driving irrelevant feature weights EXACTLY to zero (e.g. noise_feature_1, 2, 3, parking_available).")
    print("2. Ridge (L2 Penalty) shrinks weights smoothly towards zero, but NEVER sets any weight exactly to zero (retains non-zero weights for all features).")
    print("=" * 75)

if __name__ == "__main__":
    main()
