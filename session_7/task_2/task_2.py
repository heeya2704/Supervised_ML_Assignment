"""
Session 7 - Task 2
Compute Mean Absolute Error (MAE) and R^2 score for Flipkart product price predictions.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

def evaluate_flipkart_predictions(y_actual, y_predicted):
    mae = mean_absolute_error(y_actual, y_predicted)
    r2 = r2_score(y_actual, y_predicted)
    return mae, r2

def main():
    print("=" * 70)
    print("SESSION 7 - TASK 2: Flipkart Product Price MAE & R^2 Evaluation")
    print("=" * 70)

    # Flipkart product actual vs predicted prices (in INR ₹)
    actual_prices = np.array([1299, 2499, 4999, 899, 15999, 29999, 749, 1899, 11999, 3499])
    predicted_prices = np.array([1250, 2550, 4800, 950, 16200, 29500, 700, 1950, 11800, 3600])

    mae, r2 = evaluate_flipkart_predictions(actual_prices, predicted_prices)

    print("\nFLIPKART PRODUCT PRICING DATA SAMPLES:")
    print("-" * 65)
    print(f"{'Product #':<12} | {'Actual Price (Rs)':<20} | {'Predicted Price (Rs)':<20}")
    print("-" * 65)
    for i, (act, pred) in enumerate(zip(actual_prices, predicted_prices), 1):
        print(f"Product #{i:<3} | Rs. {act:<16} | Rs. {pred:<16}")
    print("-" * 65)

    print("\nMODEL EVALUATION RESULTS:")
    print("-" * 50)
    print(f"Mean Absolute Error (MAE)          : Rs. {mae:.2f}")
    print(f"Coefficient of Determination (R^2) : {r2:.4f} ({r2*100:.2f}%)")
    print("-" * 50)
    print("\nINTERPRETATION:")
    print(f"1. MAE (Rs. {mae:.2f}): Flipkart product price predictions deviate by an absolute average of Rs. {mae:.2f}.")
    print(f"2. R^2 Score ({r2:.4f}): {r2*100:.2f}% of price variance is successfully captured by the regression model.")
    print("=" * 70)

if __name__ == "__main__":
    main()
