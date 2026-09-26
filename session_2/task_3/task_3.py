"""
Session 2 - Task 3
Simple linear regression predicting popularity from danceability using 5% train and 95% test split.
Observe performance metrics and analyze underfitting vs overfitting.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def main():
    print("=" * 70)
    print("SESSION 2 - TASK 3: Extreme 5% Train / 95% Test Split Experiment")
    print("=" * 70)

    csv_path = os.path.join(os.path.dirname(__file__), "..", "dataset", "spotify_top_50.csv")
    df = pd.read_csv(csv_path)

    # Feature: danceability, Target: popularity
    X = df[['danceability']]
    y = df['popularity']

    # 5% train (approx 2-3 rows out of 50), 95% test (47-48 rows)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.95, random_state=42
    )

    print(f"\nDataset Size : {len(df)} rows")
    print(f"Train Size   : {len(X_train)} rows (5%)")
    print(f"Test Size    : {len(X_test)} rows (95%)\n")

    # Fit Linear Regression Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print(f"Model Slope (Coefficient) : {model.coef_[0]:.4f}")
    print(f"Model Intercept           : {model.intercept_:.4f}\n")

    print("-" * 50)
    print(f"Train MSE : {train_mse:.4f}")
    print(f"Test MSE  : {test_mse:.4f}")
    print(f"Train R²  : {train_r2:.4f}")
    print(f"Test R²   : {test_r2:.4f}")
    print("-" * 50)

    print("\nEXPLANATION & ANALYSIS:")
    print("Training on only 5% of the data (just 2-3 samples) causes severe UNDERFITTING & HIGH VARIANCE in parameter estimation.")
    print("1. Severe Sample Deficiency: 2-3 data points fail to capture the true underlying distribution of song popularity.")
    print("2. Generalization Failure: The fitted line is highly sensitive to whichever 2 points were sampled, resulting in terrible performance (low/negative R² and high MSE) on the 95% test set.")
    print("3. Conclusion: Training with an insufficient sample size leads to severe Underfitting/Estimation Variance where the model fails to learn true patterns.")
    print("=" * 70)

if __name__ == "__main__":
    main()
