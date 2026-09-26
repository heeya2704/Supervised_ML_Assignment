"""
Session 14 - Task 4: AI-Assisted SVM Margin & Support Vectors Visualization (Zomato Dataset)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

def main():
    print("=" * 80)
    print("SESSION 14 - TASK 4: Zomato Restaurant SVM Margin & Support Vector Plot")
    print("=" * 80)

    np.random.seed(42)
    n_samples = 40

    # Good Ratings (High location score, high votes)
    good_loc = np.random.uniform(6.5, 9.5, size=20)
    good_votes = np.random.uniform(800, 2500, size=20)

    # Bad Ratings (Low location score, low votes)
    bad_loc = np.random.uniform(1.5, 5.0, size=20)
    bad_votes = np.random.uniform(50, 600, size=20)

    X = np.vstack([
        np.column_stack([good_loc, good_votes]),
        np.column_stack([bad_loc, bad_votes])
    ])
    y = np.array(['Good'] * 20 + ['Bad'] * 20)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train Linear SVM Classifier
    svm = SVC(kernel='linear', C=1.0)
    svm.fit(X_scaled, y)

    # Plot Decision Boundary & Margins
    plt.figure(figsize=(9, 6), dpi=300)

    # Scatter data points
    plt.scatter(X_scaled[y == 'Good', 0], X_scaled[y == 'Good', 1], color='#16a34a', label='Good Rating', s=70, edgecolors='k')
    plt.scatter(X_scaled[y == 'Bad', 0], X_scaled[y == 'Bad', 1], color='#dc2626', label='Bad Rating', s=70, edgecolors='k')

    # Get hyperplane parameters: w0*x0 + w1*x1 + b = 0
    w = svm.coef_[0]
    b = svm.intercept_[0]

    x0_points = np.linspace(X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5, 200)

    # Hyperplane: w0*x0 + w1*x1 + b = 0  =>  x1 = (-w0*x0 - b) / w1
    decision_hyperplane = (-w[0] * x0_points - b) / w[1]
    
    # Margin hyperplanes: w0*x0 + w1*x1 + b = +1 and -1
    upper_margin = (-w[0] * x0_points - b + 1) / w[1]
    lower_margin = (-w[0] * x0_points - b - 1) / w[1]

    plt.plot(x0_points, decision_hyperplane, color='black', linewidth=2.5, label='Decision Boundary (w^T x + b = 0)')
    plt.plot(x0_points, upper_margin, color='gray', linestyle='--', linewidth=1.5, label='Upper Margin (w^T x + b = +1)')
    plt.plot(x0_points, lower_margin, color='gray', linestyle='--', linewidth=1.5, label='Lower Margin (w^T x + b = -1)')

    # Highlight Support Vectors
    sv = svm.support_vectors_
    plt.scatter(sv[:, 0], sv[:, 1], s=200, facecolors='none', edgecolors='#9333ea', linewidths=2.5, label='Support Vectors')

    plt.title("Zomato Restaurant Rating SVM: Maximum Margin & Support Vectors", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Standardized Location Score", fontsize=12)
    plt.ylabel("Standardized Customer Votes", fontsize=12)
    plt.legend(fontsize=10, loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)

    output_dir = os.path.dirname(__file__)
    file_path = os.path.join(output_dir, 'zomato_svm_margin.png')
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()

    print(f"\n1. Total Restaurant Samples: 40 (20 Good, 20 Bad)")
    print(f"2. Support Vectors Identified: {len(sv)} points")
    print(f"3. Margin Width (2 / ||w||): {2.0 / np.linalg.norm(w):.4f}")
    print(f"4. Margin plot saved to: {file_path}")
    print("=" * 80)

if __name__ == "__main__":
    main()
