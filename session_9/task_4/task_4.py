"""
Session 9 - Task 4: ElasticNet Regression and Effect of l1_ratio Parameter Tuning
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge, ElasticNet

def main():
    print("=" * 85)
    print("SESSION 9 - TASK 4: ElasticNet Regression Across Various l1_ratio Values")
    print("=" * 85)

    csv_path = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'zomato_ratings.csv')
    df = pd.read_csv(csv_path)

    X = df.drop(columns=['rating'])
    y = df['rating']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    l1_ratios = [0.1, 0.5, 0.9]
    alpha_val = 0.05

    results = {}
    for ratio in l1_ratios:
        enet = ElasticNet(alpha=alpha_val, l1_ratio=ratio, random_state=42)
        enet.fit(X_train_scaled, y_train)
        results[f"l1_ratio={ratio}"] = enet.coef_

    # Pure Lasso and Pure Ridge for reference
    lasso = Lasso(alpha=0.03, random_state=42).fit(X_train_scaled, y_train)
    ridge = Ridge(alpha=10.0, random_state=42).fit(X_train_scaled, y_train)
    results["Lasso (L1=1.0)"] = lasso.coef_
    results["Ridge (L1=0.0)"] = ridge.coef_

    coef_df = pd.DataFrame(results, index=X.columns)

    print("\nCOEFFICIENT COMPARISON MATRIX:")
    print("-" * 85)
    print(f"{'Feature':<22} | {'l1_ratio=0.1':<12} | {'l1_ratio=0.5':<12} | {'l1_ratio=0.9':<12} | {'Lasso':<10} | {'Ridge':<10}")
    print("-" * 85)

    for idx, row in coef_df.iterrows():
        print(f"{idx:<22} | {row['l1_ratio=0.1']:<12.5f} | {row['l1_ratio=0.5']:<12.5f} | {row['l1_ratio=0.9']:<12.5f} | {row['Lasso (L1=1.0)']:<10.5f} | {row['Ridge (L1=0.0)']:<10.5f}")

    print("-" * 85)
    print("\nKEY OBSERVATIONS ON l1_ratio:")
    print("1. Low l1_ratio (0.1): ElasticNet behaves closely to Ridge regression (smooth weight shrinkage, fewer zero coefficients).")
    print("2. High l1_ratio (0.9): ElasticNet behaves closely to Lasso regression (aggressive sparsity, setting uninformative noise features to zero).")
    print("3. Balanced l1_ratio (0.5): Combines stability in correlated feature groups (from Ridge) with sparsity (from Lasso).")
    print("=" * 85)

if __name__ == "__main__":
    main()
