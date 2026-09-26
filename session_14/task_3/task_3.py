"""
Session 14 - Task 3: Kernel Comparison (Polynomial vs RBF) on IPL Player Dataset
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def main():
    print("=" * 80)
    print("SESSION 14 - TASK 3: IPL Player Performance SVM (Polynomial vs RBF Kernel)")
    print("=" * 80)

    np.random.seed(42)
    n_samples = 120

    # IPL Player features
    batting_avg = np.random.uniform(15.0, 55.0, size=n_samples)
    strike_rate = np.random.uniform(105.0, 180.0, size=n_samples)
    wickets_taken = np.random.randint(0, 30, size=n_samples)

    # Non-linear decision boundary for IPL All-Star / Match Winner Status
    score = (
        0.08 * (batting_avg - 30)**2 + 
        0.005 * (strike_rate - 130)**2 + 
        0.15 * wickets_taken**1.5 - 
        np.random.normal(0, 3.0, size=n_samples)
    )
    is_star = np.where(score > 12.0, "Star_Player", "Regular_Player")

    df = pd.DataFrame({
        'batting_avg': batting_avg.round(1),
        'strike_rate': strike_rate.round(1),
        'wickets_taken': wickets_taken,
        'status': is_star
    })

    X = df[['batting_avg', 'strike_rate', 'wickets_taken']]
    y = df['status']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 1. Polynomial Kernel SVM (degree=3)
    svm_poly = SVC(kernel='poly', degree=3, C=1.0, random_state=42)
    svm_poly.fit(X_train_scaled, y_train)
    acc_poly = accuracy_score(y_test, svm_poly.predict(X_test_scaled))

    # 2. Radial Basis Function (RBF) Kernel SVM
    svm_rbf = SVC(kernel='rbf', gamma='scale', C=1.0, random_state=42)
    svm_rbf.fit(X_train_scaled, y_train)
    acc_rbf = accuracy_score(y_test, svm_rbf.predict(X_test_scaled))

    poly_param = "kernel='poly', degree=3"
    rbf_param = "kernel='rbf', gamma='scale'"

    print(f"\n1. Dataset Size: {len(df)} IPL Players ({len(X_train)} train / {len(X_test)} test)")
    print(f"2. Features: 'batting_avg', 'strike_rate', 'wickets_taken'")

    print("\nKERNEL ACCURACY BENCHMARK:")
    print("-" * 70)
    print(f"{'SVM Kernel Type':<25} | {'Scikit-Learn Call':<25} | {'Test Accuracy':<12}")
    print("-" * 70)
    print(f"{'Polynomial Kernel (d=3)':<25} | {poly_param:<25} | {acc_poly * 100:<10.2f}%")
    print(f"{'RBF Kernel (Gaussian)':<25} | {rbf_param:<25} | {acc_rbf * 100:<10.2f}%")
    print("-" * 70)

    print("\nTHEORETICAL EXPLANATION:")
    print("1. RBF (Radial Basis Function) Kernel: Projects data into infinite-dimensional Hilbert feature space using exp(-gamma * ||x - y||^2).")
    print("2. Polynomial Kernel: Computes interactions up to fixed degree d = (x^T y + c)^d. Can suffer from gradient explosion/instability for high degree values.")
    print("3. Conclusion: RBF kernel provides superior flexibility for smooth non-linear decision boundaries in complex sports performance data.")
    print("=" * 80)

if __name__ == "__main__":
    main()
