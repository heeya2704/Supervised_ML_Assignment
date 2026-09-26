"""
Session 5 - Task 4
Model Evaluation Metrics (MSE, RMSE, MAE, R^2 Score) for Simple Linear Regression.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def main():
    print("=" * 70)
    print("SESSION 5 - TASK 4: Simple Linear Regression Evaluation Metrics")
    print("=" * 70)

    # Weekly training data
    instagram_hours = np.array([1.5, 2.0, 3.5, 4.0, 2.5, 5.5, 6.0]).reshape(-1, 1)
    y_true = np.array([88, 82, 65, 58, 72, 40, 32])

    # Fit model & predict
    model = LinearRegression()
    model.fit(instagram_hours, y_true)
    y_pred = model.predict(instagram_hours)

    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print("\nACTUAL vs PREDICTED VALUES:")
    print("-" * 65)
    print(f"{'Day':<6} | {'Hours':<8} | {'Actual Battery %':<18} | {'Predicted Battery %':<20}")
    print("-" * 65)
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    for d, h, act, pred in zip(days, instagram_hours.flatten(), y_true, y_pred):
        print(f"{d:<6} | {h:<8.1f} | {act:<18.1f} | {pred:<20.2f}")
    print("-" * 65)

    print("\nMODEL PERFORMANCE METRICS:")
    print("-" * 50)
    print(f"Mean Absolute Error (MAE)            : {mae:.4f}%")
    print(f"Mean Squared Error (MSE)             : {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE)       : {rmse:.4f}%")
    print(f"Coefficient of Determination (R^2)   : {r2:.4f}")
    print("-" * 50)

    print("\nANALYSIS & INTERPRETATION:")
    print(f"1. R^2 Score ({r2*100:.2f}%): Approximately {r2*100:.2f}% of the variance in phone battery percentage")
    print("   is directly explained by daily Instagram usage time.")
    print(f"2. MAE ({mae:.2f}%): On average, predictions deviate from actual battery percentage by only ~{mae:.2f}%.")
    print(f"3. RMSE ({rmse:.2f}%): Represents the standard deviation of residuals, giving higher penalty to larger errors.")
    print("=" * 70)

if __name__ == "__main__":
    main()
