"""
Session 5 - Task 2
Fit LinearRegression model predicting battery % from Instagram hours. Print slope and intercept.
"""

import numpy as np
from sklearn.linear_model import LinearRegression

def main():
    print("=" * 70)
    print("SESSION 5 - TASK 2: Fitting Simple Linear Regression Model")
    print("=" * 70)

    # Weekly data arrays
    instagram_hours = np.array([1.5, 2.0, 3.5, 4.0, 2.5, 5.5, 6.0]).reshape(-1, 1)
    battery_percentage = np.array([88, 82, 65, 58, 72, 40, 32])

    # Fit Linear Regression Model
    model = LinearRegression()
    model.fit(instagram_hours, battery_percentage)

    slope = model.coef_[0]
    intercept = model.intercept_

    print("\nMODEL FITTING RESULTS:")
    print("-" * 50)
    print(f"Slope (Coefficient, m)  : {slope:.4f}")
    print(f"Y-Intercept (b)         : {intercept:.4f}")
    print("-" * 50)
    print(f"\nFitted Regression Line Equation:\n  Battery % = {intercept:.2f} + ({slope:.2f}) * (Instagram Hours)")
    print("=" * 70)
    print("\nINTERPRETATION:")
    print(f"- Intercept ({intercept:.2f}%): Expected battery remaining with 0 hours of Instagram usage.")
    print(f"- Slope ({slope:.2f}%/hr): For every 1 additional hour spent on Instagram, battery percentage drops by approximately {abs(slope):.2f}%.")
    print("=" * 70)

if __name__ == "__main__":
    main()
