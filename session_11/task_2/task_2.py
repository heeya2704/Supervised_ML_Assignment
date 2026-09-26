"""
Session 11 - Task 2: Spotify Playlist Classifier (Euclidean vs Manhattan KNN)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def main():
    print("=" * 75)
    print("SESSION 11 - TASK 2: Spotify Playlist KNN (Euclidean vs Manhattan)")
    print("=" * 75)

    np.random.seed(42)
    n_samples = 100

    # Workout Songs (High tempo, high danceability, high energy)
    tempo_workout = np.random.uniform(125, 175, size=50)
    dance_workout = np.random.uniform(0.70, 0.95, size=50)
    energy_workout = np.random.uniform(0.75, 0.98, size=50)
    labels_workout = ['workout'] * 50

    # Chill Songs (Low tempo, lower danceability, low energy)
    tempo_chill = np.random.uniform(60, 105, size=50)
    dance_chill = np.random.uniform(0.25, 0.60, size=50)
    energy_chill = np.random.uniform(0.15, 0.50, size=50)
    labels_chill = ['chill'] * 50

    X = pd.DataFrame({
        'tempo': np.concatenate([tempo_workout, tempo_chill]),
        'danceability': np.concatenate([dance_workout, dance_chill]),
        'energy': np.concatenate([energy_workout, energy_chill])
    })
    y = np.array(labels_workout + labels_chill)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 1. KNN with Euclidean Distance (L2 Norm)
    knn_euclidean = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
    knn_euclidean.fit(X_train_scaled, y_train)
    acc_euc = accuracy_score(y_test, knn_euclidean.predict(X_test_scaled))

    # 2. KNN with Manhattan Distance (L1 Norm)
    knn_manhattan = KNeighborsClassifier(n_neighbors=5, metric='manhattan')
    knn_manhattan.fit(X_train_scaled, y_train)
    acc_man = accuracy_score(y_test, knn_manhattan.predict(X_test_scaled))

    param_euc = "metric='euclidean'"
    param_man = "metric='manhattan'"

    print(f"\n1. Dataset Summary: 100 Songs (70 Train / 30 Test)")
    print(f"2. Features: 'tempo', 'danceability', 'energy'")
    print(f"3. Class Target: 'workout' vs 'chill'\n")

    print("DISTANCE METRIC PERFORMANCE COMPARISON:")
    print("-" * 65)
    print(f"{'Distance Metric':<25} | {'Scikit-Learn Parameter':<22} | {'Test Accuracy':<12}")
    print("-" * 65)
    print(f"{'Euclidean Distance (L2)':<25} | {param_euc:<22} | {acc_euc * 100:<10.2f}%")
    print(f"{'Manhattan Distance (L1)':<25} | {param_man:<22} | {acc_man * 100:<10.2f}%")
    print("-" * 65)

    print("\nMATHEMATICAL & IMPLICATION SUMMARY:")
    print("1. Euclidean Distance: d(p, q) = sqrt(sum((p_i - q_i)^2)). Measures straight-line distance in feature space.")
    print("2. Manhattan Distance: d(p, q) = sum(|p_i - q_i|). Measures grid-like distance along feature axes.")
    print("3. Both metrics achieve 100% accuracy on well-separated standardized Spotify audio features.")
    print("=" * 75)

if __name__ == "__main__":
    main()
