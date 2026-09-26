"""
Session 13 - Task 4: Bagging Demonstration - Single Decision Tree vs Random Forest on Noisy Data
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def main():
    print("=" * 80)
    print("SESSION 13 - TASK 4: Overfitting Mitigation: Decision Tree vs Random Forest")
    print("=" * 80)

    # Generate noisy classification dataset (20% flip noise)
    X, y = make_classification(
        n_samples=500,
        n_features=20,
        n_informative=8,
        n_redundant=8,
        n_clusters_per_class=2,
        flip_y=0.20,  # 20% random label noise to induce overfitting in single tree
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 1. Single Unpruned Decision Tree (Prone to Overfitting on Noise)
    dt_clf = DecisionTreeClassifier(random_state=42)
    dt_clf.fit(X_train, y_train)
    dt_train_acc = accuracy_score(y_train, dt_clf.predict(X_train))
    dt_test_acc = accuracy_score(y_test, dt_clf.predict(X_test))

    # 2. Random Forest Classifier (Bagging + Feature Subsampling reduces variance)
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_clf.fit(X_train, y_train)
    rf_train_acc = accuracy_score(y_train, rf_clf.predict(X_train))
    rf_test_acc = accuracy_score(y_test, rf_clf.predict(X_test))

    print("\nMODEL OVERFITTING COMPARISON MATRIX (ON NOISY DATA):")
    print("-" * 75)
    print(f"{'Classifier Model':<28} | {'Train Accuracy':<16} | {'Test Accuracy':<16} | {'Overfit Gap':<12}")
    print("-" * 75)
    print(f"{'Single Decision Tree':<28} | {dt_train_acc * 100:<14.2f}% | {dt_test_acc * 100:<14.2f}% | {(dt_train_acc - dt_test_acc) * 100:<10.2f}%")
    print(f"{'Random Forest (100 Trees)':<28} | {rf_train_acc * 100:<14.2f}% | {rf_test_acc * 100:<14.2f}% | {(rf_train_acc - rf_test_acc) * 100:<10.2f}%")
    print("-" * 75)

    print("\nHOW BAGGING REDUCES OVERFITTING & VARIANCE:")
    print("1. Bootstrap Aggregation (Bagging): Each tree is trained on an independent bootstrap sample of data (sample with replacement).")
    print("2. Random Feature Subsampling: At each node split, only sqrt(d) random features are considered, decorrelating individual trees.")
    print("3. Majority Voting (Ensemble): Averaging predictions across decorrelated trees cancels out random sample noise variance, boosting test accuracy.")
    print("=" * 80)

if __name__ == "__main__":
    main()
