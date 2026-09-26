"""
Session 14 - Task 1: Music Genre SVM Classifier & Decision Boundary Visualization
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

def main():
    print("=" * 75)
    print("SESSION 14 - TASK 1: Music Genre SVM Classifier & Decision Boundary")
    print("=" * 75)

    np.random.seed(42)
    n_samples = 40

    # Pop Music (High tempo, high danceability)
    pop_tempo = np.random.uniform(115, 150, size=20)
    pop_dance = np.random.uniform(0.65, 0.95, size=20)
    pop_labels = ['Pop'] * 20

    # Classical Music (Low tempo, low danceability)
    classical_tempo = np.random.uniform(50, 90, size=20)
    classical_dance = np.random.uniform(0.15, 0.45, size=20)
    classical_labels = ['Classical'] * 20

    X = np.vstack([
        np.column_stack([pop_tempo, pop_dance]),
        np.column_stack([classical_tempo, classical_dance])
    ])
    y = np.array(pop_labels + classical_labels)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train SVM with Linear Kernel
    svm = SVC(kernel='linear', C=1.0)
    svm.fit(X_scaled, y)

    # Decision boundary mesh grid
    x_min, x_max = X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5
    y_min, y_max = X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))

    target_numeric = np.where(y == 'Pop', 1, 0)
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z_num = np.where(Z == 'Pop', 1, 0).reshape(xx.shape)

    plt.figure(figsize=(9, 6), dpi=300)
    plt.contourf(xx, yy, Z_num, alpha=0.3, cmap=plt.cm.coolwarm)
    
    # Scatter plot data points
    plt.scatter(X_scaled[y == 'Pop', 0], X_scaled[y == 'Pop', 1], color='#1d4ed8', label='Pop', s=70, edgecolors='k')
    plt.scatter(X_scaled[y == 'Classical', 0], X_scaled[y == 'Classical', 1], color='#dc2626', label='Classical', s=70, edgecolors='k')

    # Highlight Support Vectors
    sv = svm.support_vectors_
    plt.scatter(sv[:, 0], sv[:, 1], s=180, facecolors='none', edgecolors='k', linewidths=2, label='Support Vectors')

    plt.title("SVM Linear Decision Boundary: Pop vs Classical Genre", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Standardized Tempo", fontsize=12)
    plt.ylabel("Standardized Danceability", fontsize=12)
    plt.legend(fontsize=11, loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)

    output_dir = os.path.dirname(__file__)
    file_path = os.path.join(output_dir, 'svm_music_boundary.png')
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()

    print(f"\n1. Total Samples: 40 (20 Pop, 20 Classical)")
    print(f"2. Support Vectors Count: {len(sv)}")
    print(f"3. Decision Boundary Plot Saved to: {file_path}")
    print("=" * 75)

if __name__ == "__main__":
    main()
