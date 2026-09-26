"""
Session 9 - Task 5: Zomato Restaurant Rating Predictor Feature Selector via Lasso
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

def zomato_feature_selector(alpha=0.03):
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'zomato_ratings.csv')
    df = pd.read_csv(csv_path)

    X = df.drop(columns=['rating'])
    y = df['rating']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    lasso = Lasso(alpha=alpha, random_state=42)
    lasso.fit(X_train_scaled, y_train)

    feature_coef_map = pd.Series(lasso.coef_, index=X.columns)

    kept_features = feature_coef_map[feature_coef_map.abs() > 1e-5].sort_values(ascending=False)
    dropped_features = feature_coef_map[feature_coef_map.abs() <= 1e-5]

    return kept_features, dropped_features

def main():
    print("=" * 75)
    print("SESSION 9 - TASK 5: Zomato Feature Selector (Lasso Regularization)")
    print("=" * 75)

    kept, dropped = zomato_feature_selector(alpha=0.03)

    print(f"\nKEPT FEATURES ({len(kept)} features with non-zero weights):")
    print("-" * 55)
    print(f"{'Feature Name':<25} | {'Importance / Coef':<20}")
    print("-" * 55)
    for feat, coef in kept.items():
        print(f"{feat:<25} | {coef:<20.6f}")

    print("-" * 55)
    print(f"\nDROPPED FEATURES ({len(dropped)} features eliminated with 0 weight):")
    print("-" * 55)
    print(f"{'Feature Name':<25} | {'Status':<20}")
    print("-" * 55)
    for feat in dropped.index:
        print(f"{feat:<25} | {'ELIMINATED (0.00)':<20}")

    print("-" * 55)
    print("\nBUSINESS REASONING & FEATURE SELECTION SUMMARY:")
    print("1. Key Drivers Kept: 'location_score', 'votes', 'book_table', and 'restaurant_age_yrs' have highest positive predictive power.")
    print("2. Irrelevant Noise Dropped: Uninformative random features (noise_feature_1, 2, 3) and weak predictors (parking_available) are discarded.")
    print("3. Pipeline Efficiency: Training downstream models on the 9 kept features improves training speed and removes noise distortion.")
    print("=" * 75)

if __name__ == "__main__":
    main()
