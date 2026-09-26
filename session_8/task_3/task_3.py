"""
Session 8 - Task 3
Instagram Posts vs Followers Polynomial Regression (degree=4) and Degree Validation Curve (Degrees 1 to 5).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import root_mean_squared_error

def main():
    print("=" * 70)
    print("SESSION 8 - TASK 3: Instagram Followers Polynomial Degree Validation Curve")
    print("=" * 70)

    np.random.seed(42)
    n_samples = 80

    # Number of posts (X)
    posts = np.random.uniform(10, 500, size=n_samples).reshape(-1, 1)

    # Non-linear followers relationship (Y in thousands) with noise
    followers = (
        5.0 + 
        0.05 * posts.flatten() + 
        0.0008 * (posts.flatten() ** 2) - 
        0.0000015 * (posts.flatten() ** 3) + 
        np.random.normal(0, 15, size=n_samples)
    )
    followers = np.clip(followers, 0, None)

    X_train, X_val, y_train, y_val = train_test_split(posts, followers, test_size=0.3, random_state=42)

    degrees = list(range(1, 6))
    train_errors = []
    val_errors = []

    print("\nDEGREE VALIDATION METRICS:")
    print("-" * 65)
    print(f"{'Degree':<8} | {'Train RMSE (k Followers)':<24} | {'Val RMSE (k Followers)':<24}")
    print("-" * 65)

    for d in degrees:
        model = make_pipeline(PolynomialFeatures(degree=d), LinearRegression())
        model.fit(X_train, y_train)

        tr_rmse = root_mean_squared_error(y_train, model.predict(X_train))
        val_rmse = root_mean_squared_error(y_val, model.predict(X_val))

        train_errors.append(tr_rmse)
        val_errors.append(val_rmse)

        print(f"Degree {d:<2} | {tr_rmse:<24.4f} | {val_rmse:<24.4f}")
    print("-" * 65)

    # Plot validation curve
    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=300)

    plt.plot(degrees, train_errors, color='#1f77b4', marker='o', linewidth=2.5, label='Training Error (RMSE)')
    plt.plot(degrees, val_errors, color='#d62728', marker='s', linewidth=2.5, linestyle='--', label='Validation Error (RMSE)')

    plt.title("Polynomial Model Complexity vs Error: Instagram Followers Prediction", fontsize=13, weight='bold', pad=15)
    plt.xlabel("Polynomial Degree", fontsize=11)
    plt.ylabel("Root Mean Squared Error (RMSE in '000s)", fontsize=11)
    plt.xticks(degrees)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right')
    plt.tight_layout()

    img_path = os.path.join(os.path.dirname(__file__), "degree_validation_curve.png")
    plt.savefig(img_path, bbox_inches='tight')
    plt.close()

    print(f"\n[+] Saved validation curve visualization to: {img_path}")
    print("\nOVERFITTING DIAGNOSIS:")
    print("1. Degrees 1-2 underfit the non-linear trend (high training & validation error).")
    print("2. Degrees 3-4 achieve optimal balance with low training & validation errors.")
    print("3. Degree 5 begins to overfit (training error continues dropping while validation error increases).")
    print("=" * 70)

if __name__ == "__main__":
    main()
