"""
Session 7 - Task 1
Calculate MSE and RMSE for 10 Zomato food delivery orders.
"""

import numpy as np
from sklearn.metrics import mean_squared_error, root_mean_squared_error

def main():
    print("=" * 70)
    print("SESSION 7 - TASK 1: Zomato Delivery Time Error Metrics (MSE & RMSE)")
    print("=" * 70)

    # 10 Zomato delivery order time arrays (in minutes)
    actual_times = np.array([25, 30, 45, 20, 35, 50, 28, 40, 32, 22])
    predicted_times = np.array([28, 27, 48, 24, 32, 55, 30, 37, 35, 20])

    # Calculate MSE and RMSE
    mse = mean_squared_error(actual_times, predicted_times)
    rmse = root_mean_squared_error(actual_times, predicted_times)

    print("\nZOMATO ORDERS DATA TABLE:")
    print("-" * 65)
    print(f"{'Order #':<10} | {'Actual Time (min)':<18} | {'Predicted Time (min)':<20} | {'Error (y - y_hat)':<15}")
    print("-" * 65)
    for i, (act, pred) in enumerate(zip(actual_times, predicted_times), 1):
        err = act - pred
        print(f"Order #{i:<3} | {act:<18.1f} | {pred:<20.1f} | {err:<15.1f}")
    print("-" * 65)

    print("\nEVALUATION METRICS RESULTS:")
    print("-" * 50)
    print(f"Mean Squared Error (MSE)       : {mse:.4f} min^2")
    print(f"Root Mean Squared Error (RMSE) : {rmse:.4f} minutes")
    print("-" * 50)
    print(f"\nINTERPRETATION:")
    print(f"- The model's delivery time predictions deviate by an average standard deviation of {rmse:.2f} minutes.")
    print("=" * 70)

if __name__ == "__main__":
    main()
