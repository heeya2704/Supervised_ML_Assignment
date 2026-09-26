"""
Session 5 - Task 3
Making Battery Percentage Predictions using Fitted Simple Linear Regression Model.
"""

import numpy as np
from sklearn.linear_model import LinearRegression

def main():
    print("=" * 70)
    print("SESSION 5 - TASK 3: Battery Percentage Predictions for New Usage Inputs")
    print("=" * 70)

    # Weekly training data
    instagram_hours = np.array([1.5, 2.0, 3.5, 4.0, 2.5, 5.5, 6.0]).reshape(-1, 1)
    battery_percentage = np.array([88, 82, 65, 58, 72, 40, 32])

    # Fit model
    model = LinearRegression()
    model.fit(instagram_hours, battery_percentage)

    # New inputs for prediction
    test_hours = np.array([[3.0], [4.5], [7.0]])
    predictions = model.predict(test_hours)

    print("\nPREDICTION RESULTS FOR NEW USAGE SCENARIOS:")
    print("-" * 60)
    print(f"{'Instagram Usage (Hrs)':<24} | {'Predicted Battery %':<20}")
    print("-" * 60)
    for hrs, pred in zip(test_hours.flatten(), predictions):
        # Clip battery % to valid range [0, 100] for logical safety presentation
        clipped_pred = max(0.0, min(100.0, pred))
        print(f"{hrs:<24.1f} | {pred:<8.2f}% (Clipped: {clipped_pred:.1f}%)")
    print("-" * 60)

    print("\nVERIFICATION VIA REGRESSION EQUATION:")
    print(f"Equation: Battery % = {model.intercept_:.2f} + ({model.coef_[0]:.2f}) * (Hours)")
    for hrs in test_hours.flatten():
        calc_val = model.intercept_ + model.coef_[0] * hrs
        print(f"  - For {hrs:.1f} hrs: {model.intercept_:.2f} + ({model.coef_[0]:.2f} * {hrs}) = {calc_val:.2f}%")
    print("=" * 70)

if __name__ == "__main__":
    main()
