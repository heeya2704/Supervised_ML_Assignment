"""
Session 8 - Task 4
L2 Regularization (Ridge Regression) vs Unregularized Polynomial Regression (Degree=3) on Flipkart Prices.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.metrics import root_mean_squared_error, r2_score

def main():
    print("=" * 70)
    print("SESSION 8 - TASK 4: Ridge L2 Regularization on Polynomial Model (Degree=3)")
    print("=" * 70)

    np.random.seed(42)
    n_samples = 150

    # Flipkart features
    product_rating = np.random.uniform(3.0, 5.0, size=n_samples)
    review_count = np.random.uniform(100, 5000, size=n_samples)
    spec_score = np.random.uniform(1, 10, size=n_samples)

    # Non-linear price relationship + noise
    price = (
        1500 + 
        800 * spec_score**2 - 
        45 * spec_score**3 * 0.05 + 
        1.5 * review_count + 
        200 * product_rating**2 + 
        np.random.normal(0, 3000, size=n_samples)
    )
    price = np.clip(price, 500, None)

    X = pd.DataFrame({'spec_score': spec_score, 'review_count': review_count, 'product_rating': product_rating})
    y = price

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Pipeline 1: Unregularized Polynomial Regression (Degree 3)
    model_unreg = make_pipeline(PolynomialFeatures(degree=3), StandardScaler(), LinearRegression())
    model_unreg.fit(X_train, y_train)

    tr_rmse_unreg = root_mean_squared_error(y_train, model_unreg.predict(X_train))
    te_rmse_unreg = root_mean_squared_error(y_test, model_unreg.predict(X_test))
    te_r2_unreg = r2_score(y_test, model_unreg.predict(X_test))

    # Pipeline 2: Ridge L2 Regularized Polynomial Regression (Degree 3, alpha=10.0)
    model_ridge = make_pipeline(PolynomialFeatures(degree=3), StandardScaler(), Ridge(alpha=10.0))
    model_ridge.fit(X_train, y_train)

    tr_rmse_ridge = root_mean_squared_error(y_train, model_ridge.predict(X_train))
    te_rmse_ridge = root_mean_squared_error(y_test, model_ridge.predict(X_test))
    te_r2_ridge = r2_score(y_test, model_ridge.predict(X_test))

    # Coefficient Norms Comparison
    coef_unreg_norm = np.linalg.norm(model_unreg.named_steps['linearregression'].coef_)
    coef_ridge_norm = np.linalg.norm(model_ridge.named_steps['ridge'].coef_)

    print("\nPERFORMANCE COMPARISON (WITH VS WITHOUT L2 REGULARIZATION):")
    print("-" * 75)
    print(f"{'Metric / Parameter':<28} | {'Unregularized Polynomial':<22} | {'Ridge L2 Regularized':<20}")
    print("-" * 75)
    print(f"{'Train RMSE (Rs)':<28} | Rs. {tr_rmse_unreg:<18.2f} | Rs. {tr_rmse_ridge:<16.2f}")
    print(f"{'Test RMSE (Rs)':<28} | Rs. {te_rmse_unreg:<18.2f} | Rs. {te_rmse_ridge:<16.2f}")
    print(f"{'Test R^2 Score':<28} | {te_r2_unreg:<22.4f} | {te_r2_ridge:<20.4f}")
    print(f"{'L2 Norm of Coefs (||w||_2)':<28} | {coef_unreg_norm:<22.2f} | {coef_ridge_norm:<20.2f}")
    print("-" * 75)

    print("\nREASONING & CONCLUSION:")
    print("1. Coefficient Shrinkage: L2 Ridge regularization heavily penalizes extreme polynomial coefficients, reducing ||w||_2 magnitude.")
    print("2. Improved Test Generalization: By constraining large weights, Ridge prevents overfitting to training noise, lowering test RMSE.")
    print("=" * 70)

if __name__ == "__main__":
    main()
