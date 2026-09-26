"""
Session 11 - Task 3: IRCTC Train Booking Confirmation Classifier (Gaussian Naive Bayes)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

def main():
    print("=" * 75)
    print("SESSION 11 - TASK 3: IRCTC Booking Confirmation Classifier (GaussianNB)")
    print("=" * 75)

    np.random.seed(42)
    n_samples = 120

    # Booking features
    booking_days_before = np.random.randint(1, 90, size=n_samples)
    train_popularity_score = np.random.uniform(1.0, 10.0, size=n_samples)
    travel_day_weekend = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])

    # Target: Confirmed (1) vs Waitlisted (0)
    # Higher booking days before + lower train popularity + weekday -> Confirmed
    score = (
        0.05 * booking_days_before - 
        0.35 * train_popularity_score - 
        0.8 * travel_day_weekend + 
        np.random.normal(0, 0.8, size=n_samples)
    )
    status_label = np.where(score > -1.5, "confirmed", "waitlisted")

    df = pd.DataFrame({
        'booking_days_before': booking_days_before,
        'train_popularity_score': train_popularity_score.round(2),
        'travel_day_weekend': travel_day_weekend,
        'status': status_label
    })

    X = df.drop(columns=['status'])
    y = df['status']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    # Train Gaussian Naive Bayes model
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)

    y_pred = gnb.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n1. Dataset Size: {len(df)} bookings ({len(X_train)} train / {len(X_test)} test)")
    print(f"2. Features: 'booking_days_before', 'train_popularity_score', 'travel_day_weekend'")
    print(f"3. Model Accuracy: {acc * 100:.2f}%\n")

    print("Sample IRCTC Booking Predictions:")
    print("-" * 75)
    print(f"{'Days Before':<12} | {'Popularity (1-10)':<18} | {'Weekend (0/1)':<14} | {'Actual':<12} | {'Predicted':<12}")
    print("-" * 75)
    for idx in range(6):
        row = X_test.iloc[idx]
        actual = y_test.iloc[idx]
        pred = y_pred[idx]
        print(f"{int(row['booking_days_before']):<12} | {row['train_popularity_score']:<18.2f} | {int(row['travel_day_weekend']):<14} | {actual:<12} | {pred:<12}")
    print("-" * 75)

    print("\nGAUSSIAN NAIVE BAYES FORMULA:")
    print("P(Confirmed | X) = [ P(Confirmed) * PROD( P(X_i | Confirmed) ) ] / P(X)")
    print("where P(X_i | Confirmed) is modeled as a continuous Gaussian (Normal) distribution.")
    print("=" * 75)

if __name__ == "__main__":
    main()
