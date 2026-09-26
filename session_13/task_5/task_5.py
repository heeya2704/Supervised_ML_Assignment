"""
Session 13 - Task 5: AI-Generated Spotify Hit Song Predictor (Random Forest)
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def create_spotify_hit_csv():
    np.random.seed(42)
    n = 120

    tempo = np.random.uniform(70, 180, size=n)
    danceability = np.random.uniform(0.3, 0.95, size=n)
    energy = np.random.uniform(0.3, 0.98, size=n)
    speechiness = np.random.uniform(0.03, 0.45, size=n)

    # Hit formula based on high danceability & energy
    hit_score = 2.5 * danceability + 2.0 * energy + 0.005 * tempo - 1.5 * speechiness + np.random.normal(0, 0.3, size=n)
    is_hit = (hit_score > 3.0).astype(int)

    df = pd.DataFrame({
        'tempo': tempo.round(1),
        'danceability': danceability.round(3),
        'energy': energy.round(3),
        'speechiness': speechiness.round(3),
        'is_hit': is_hit
    })

    dataset_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'session_13', 'dataset')
    os.makedirs(dataset_dir, exist_ok=True)
    csv_path = os.path.join(dataset_dir, 'spotify_hits.csv')
    df.to_csv(csv_path, index=False)
    return csv_path

def main():
    print("=" * 80)
    print("SESSION 13 - TASK 5: AI-Generated Spotify Hit Predictor (Random Forest)")
    print("=" * 80)

    csv_path = create_spotify_hit_csv()
    df = pd.read_csv(csv_path)

    X = df[['tempo', 'danceability', 'energy', 'speechiness']]
    y = df['is_hit']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    # AI solution snippet used oob_score=True without setting n_estimators large enough or bootstrap=True
    # Fix Applied: Set bootstrap=True explicitly and ensured n_estimators=100
    rf_hit = RandomForestClassifier(n_estimators=100, bootstrap=True, oob_score=True, random_state=42)
    rf_hit.fit(X_train, y_train)

    y_pred = rf_hit.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    oob_acc = rf_hit.oob_score_

    print(f"\n1. Spotify Dataset Loaded: {len(df)} songs from {csv_path}")
    print(f"2. Features: 'tempo', 'danceability', 'energy', 'speechiness'")
    print(f"3. Test Set Accuracy: {test_acc * 100:.2f}%")
    print(f"4. Out-of-Bag (OOB) Accuracy Score: {oob_acc * 100:.2f}%\n")

    print("FEATURE IMPORTANCES (SPOTIFY HIT PREDICTORS):")
    print("-" * 55)
    print(f"{'Feature Name':<20} | {'Importance Score':<18} | {'Contribution':<12}")
    print("-" * 55)
    for feat, score in zip(X.columns, rf_hit.feature_importances_):
        print(f"{feat:<20} | {score:<18.6f} | {score * 100:<10.2f}%")
    print("-" * 55)

    print("\nAI SOLUTION LEARNING & FIX:")
    print("1. Key Takeaway Learned: Out-of-Bag (OOB) scoring allows Random Forest to evaluate validation performance on unchosen bootstrap samples without needing a separate cross-validation split.")
    print("2. Fix Applied: Set bootstrap=True explicitly to enable oob_score evaluation.")
    print("=" * 80)

if __name__ == "__main__":
    main()
