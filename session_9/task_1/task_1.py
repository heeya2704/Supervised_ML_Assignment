"""
Session 9 - Task 1: Dataset Loading and Train/Test Splitting for Regression
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    print("=" * 70)
    print("SESSION 9 - TASK 1: Dataset Preparation & Train-Test Split")
    print("=" * 70)

    csv_path = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'zomato_ratings.csv')
    df = pd.read_csv(csv_path)

    print(f"\n1. Loaded Dataset Summary:")
    print(f"   Total Samples: {df.shape[0]}")
    print(f"   Total Columns: {df.shape[1]}")
    print(f"   Target Variable: 'rating'")

    # Features (X) and Target (y)
    X = df.drop(columns=['rating'])
    y = df['rating']

    # Train/Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"\n2. Train-Test Split Results (80/20):")
    print(f"   Feature Matrix X Shape: {X.shape}")
    print(f"   Training Set (X_train): {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"   Testing Set (X_test):   {X_test.shape[0]} samples, {X_test.shape[1]} features")
    print(f"   Target Vector (y_train): {y_train.shape[0]} samples")
    print(f"   Target Vector (y_test):  {y_test.shape[0]} samples")

    print("\n3. Feature Names (13 Features):")
    for idx, col in enumerate(X.columns, 1):
        print(f"   [{idx:2d}] {col}")

    print("=" * 70)

if __name__ == "__main__":
    main()
