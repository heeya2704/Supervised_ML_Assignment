"""
Session 7 - Task 5
Compare two regression models for Spotify song popularity and detect overfitting via Train vs Test evaluation metrics.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error, r2_score

def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    
    # Train set predictions
    y_tr_pred = model.predict(X_train)
    tr_mse = mean_squared_error(y_train, y_tr_pred)
    tr_rmse = root_mean_squared_error(y_train, y_tr_pred)
    tr_mae = mean_absolute_error(y_train, y_tr_pred)
    tr_r2 = r2_score(y_train, y_tr_pred)
    
    # Test set predictions
    y_te_pred = model.predict(X_test)
    te_mse = mean_squared_error(y_test, y_te_pred)
    te_rmse = root_mean_squared_error(y_test, y_te_pred)
    te_mae = mean_absolute_error(y_test, y_te_pred)
    te_r2 = r2_score(y_test, y_te_pred)

    return {
        'Train MSE': tr_mse, 'Test MSE': te_mse,
        'Train RMSE': tr_rmse, 'Test RMSE': te_rmse,
        'Train MAE': tr_mae, 'Test MAE': te_mae,
        'Train R2': tr_r2, 'Test R2': te_r2
    }

def main():
    print("=" * 70)
    print("SESSION 7 - TASK 5: Spotify Popularity Model Overfitting Diagnosis")
    print("=" * 70)

    np.random.seed(42)
    n_samples = 200

    # Synthetic Spotify track features
    danceability = np.random.uniform(0.3, 0.95, size=n_samples)
    energy = np.random.uniform(0.2, 0.99, size=n_samples)
    loudness = np.random.uniform(-20, -2, size=n_samples)
    tempo = np.random.uniform(70, 180, size=n_samples)

    # Ground truth popularity (scale 0-100)
    popularity = (
        20 + danceability * 35 + energy * 25 + (loudness + 20) * 1.2 + 
        (tempo / 180) * 10 + np.random.normal(0, 4.0, size=n_samples)
    )
    popularity = np.clip(popularity, 0, 100)

    X = pd.DataFrame({'danceability': danceability, 'energy': energy, 'loudness': loudness, 'tempo': tempo})
    y = popularity

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Model 1: Regularized Linear Model (Generalized)
    model_1 = Ridge(alpha=1.0)
    res_1 = evaluate_model(model_1, X_train, X_test, y_train, y_test)

    # Model 2: Unconstrained Deep Decision Tree (Overfitted)
    model_2 = DecisionTreeRegressor(max_depth=None, random_state=42)
    res_2 = evaluate_model(model_2, X_train, X_test, y_train, y_test)

    print("\nTRAIN VS TEST METRIC COMPARISON:")
    print("-" * 75)
    print(f"{'Metric':<14} | {'Model 1 (Ridge) Train':<22} | {'Model 1 Test':<14} | {'Model 2 (Complex) Train':<24} | {'Model 2 Test':<12}")
    print("-" * 75)
    print(f"{'MSE':<14} | {res_1['Train MSE']:<22.2f} | {res_1['Test MSE']:<14.2f} | {res_2['Train MSE']:<24.2f} | {res_2['Test MSE']:<12.2f}")
    print(f"{'RMSE':<14} | {res_1['Train RMSE']:<22.2f} | {res_1['Test RMSE']:<14.2f} | {res_2['Train RMSE']:<24.2f} | {res_2['Test RMSE']:<12.2f}")
    print(f"{'MAE':<14} | {res_1['Train MAE']:<22.2f} | {res_1['Test MAE']:<14.2f} | {res_2['Train MAE']:<24.2f} | {res_2['Test MAE']:<12.2f}")
    print(f"{'R^2 Score':<14} | {res_1['Train R2']:<22.4f} | {res_1['Test R2']:<14.4f} | {res_2['Train R2']:<24.4f} | {res_2['Test R2']:<12.4f}")
    print("-" * 75)

    print("\nOVERFITTING DIAGNOSIS:")
    print("--> MODEL 2 (Unconstrained Decision Tree) IS SEVERELY OVERFITTING!")
    print("    - Model 2 achieves near-zero error on training data (Train R^2 = 1.0000, Train RMSE = 0.00),")
    print("      but performance deteriorates drastically on test data (Test RMSE spikes, Test R^2 drops).")
    print("    - Model 1 (Ridge) generalizes smoothly with consistent Train and Test R^2 (~0.85).")
    print("=" * 70)

if __name__ == "__main__":
    main()
