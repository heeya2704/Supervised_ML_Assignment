"""
Session 8 - Task 2
Linear vs Polynomial Regression (degree=3) comparison on Zomato Restaurant Ratings vs Number of Reviews.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

def main():
    print("=" * 70)
    print("SESSION 8 - TASK 2: Linear vs Polynomial (Degree=3) Fit Comparison")
    print("=" * 70)

    np.random.seed(42)
    # Number of reviews (X)
    n_reviews = np.sort(np.random.uniform(50, 2500, size=60)).reshape(-1, 1)

    # Non-linear relationship for Zomato ratings (Y)
    # Logarithmic / diminishing returns shape with noise
    ratings = 2.2 + 0.5 * np.log(n_reviews.flatten()) + np.random.normal(0, 0.15, size=60)
    ratings = np.clip(ratings, 1.0, 5.0)

    # Model 1: Linear Regression (Degree 1)
    lin_model = LinearRegression()
    lin_model.fit(n_reviews, ratings)
    y_pred_lin = lin_model.predict(n_reviews)
    r2_lin = r2_score(ratings, y_pred_lin)

    # Model 2: Polynomial Regression (Degree 3)
    poly_model = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())
    poly_model.fit(n_reviews, ratings)
    y_pred_poly = poly_model.predict(n_reviews)
    r2_poly = r2_score(ratings, y_pred_poly)

    print("\nMODEL FIT COMPARISON METRICS:")
    print("-" * 55)
    print(f"Linear Regression R^2 Score     : {r2_lin:.4f}")
    print(f"Polynomial (Degree=3) R^2 Score : {r2_poly:.4f}")
    print("-" * 55)

    # Smooth curve generation for plotting
    x_smooth = np.linspace(50, 2500, 200).reshape(-1, 1)
    y_lin_smooth = lin_model.predict(x_smooth)
    y_poly_smooth = poly_model.predict(x_smooth)

    # Plot chart
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

    plt.scatter(n_reviews, ratings, color='#d62728', alpha=0.7, s=70, zorder=3, 
                edgecolors='black', label='Zomato Restaurant Reviews Data')
    plt.plot(x_smooth, y_lin_smooth, color='#1f77b4', linestyle='--', linewidth=2.5, 
             label=f'Linear Fit (R^2 = {r2_lin:.2f})')
    plt.plot(x_smooth, y_poly_smooth, color='#2ca02c', linewidth=2.8, 
             label=f'Polynomial Degree-3 Fit (R^2 = {r2_poly:.2f})')

    plt.title("Zomato Restaurant Ratings vs Number of Reviews: Model Fit Comparison", fontsize=13, weight='bold', pad=15)
    plt.xlabel("Number of Reviews", fontsize=11)
    plt.ylabel("Zomato Rating (Scale 1.0 - 5.0)", fontsize=11)
    plt.ylim(2.5, 5.2)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right', framealpha=0.95)
    plt.tight_layout()

    img_path = os.path.join(os.path.dirname(__file__), "linear_vs_polynomial.png")
    plt.savefig(img_path, bbox_inches='tight')
    plt.close()

    print(f"\n[+] Saved plot visualization to: {img_path}")
    print("\nOBSERVATIONS:")
    print("1. The linear regression model fails to capture the curvature (underfits the diminishing returns).")
    print("2. The Degree-3 polynomial model successfully fits the non-linear plateauing trend of restaurant ratings.")
    print("=" * 70)

if __name__ == "__main__":
    main()
