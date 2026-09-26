"""
Session 17 - Task 1: Train Logistic Regression on Imbalanced Credit Card Fraud Dataset
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

def main():
    print("=" * 75)
    print("SESSION 17 - TASK 1: Logistic Regression on Imbalanced Fraud Dataset")
    print("=" * 75)

    # 1. Generate synthetic Credit Card Fraud dataset (5% fraud / positive class)
    X, y = make_classification(
        n_samples=2000,
        n_features=12,
        n_informative=8,
        n_redundant=4,
        weights=[0.95, 0.05],  # 95% Legitimate (0), 5% Fraudulent (1)
        random_state=42
    )

    feature_names = [f"V{i+1}" for i in range(12)]
    df = pd.DataFrame(X, columns=feature_names)
    df['Class'] = y

    print(f"\n1. DATASET DISTRIBUTION:")
    print(f"   • Total Transactions : {len(df)}")
    print(f"   • Legitimate (Class 0): {(y == 0).sum()} ({((y == 0).mean()*100):.2f}%)")
    print(f"   • Fraudulent (Class 1): {(y == 1).sum()} ({((y == 1).mean()*100):.2f}%)")

    # 2. Split dataset into Train and Test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"\n2. TRAIN / TEST SPLIT (Stratified 70/30):")
    print(f"   • Training Samples: {X_train.shape[0]} (Fraud: {sum(y_train)})")
    print(f"   • Testing Samples : {X_test.shape[0]} (Fraud: {sum(y_test)})")

    # 3. Train Logistic Regression Classifier
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)

    # 4. Predictions & Baseline Evaluation
    y_pred = model.predict(X_test)

    print("\n3. CLASSIFICATION REPORT (Baseline Logistic Regression):")
    print("-" * 65)
    print(classification_report(y_test, y_pred, target_names=["Legitimate (0)", "Fraud (1)"]))
    print("-" * 65)

    cm = confusion_matrix(y_test, y_pred)
    print("4. CONFUSION MATRIX:")
    print(f"   • True Negatives  (TN) = {cm[0,0]}")
    print(f"   • False Positives (FP) = {cm[0,1]}")
    print(f"   • False Negatives (FN) = {cm[1,0]}")
    print(f"   • True Positives  (TP) = {cm[1,1]}")
    print("=" * 75)

if __name__ == "__main__":
    main()
