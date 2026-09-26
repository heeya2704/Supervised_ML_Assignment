"""
Session 17 - Task 2: Plot ROC Curve and Display AUC Score for Fraud Classifier
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score

def main():
    print("=" * 75)
    print("SESSION 17 - TASK 2: ROC Curve & AUC Score Visualization")
    print("=" * 75)

    # 1. Dataset Generation & Model Fitting
    X, y = make_classification(
        n_samples=2000, n_features=12, n_informative=8, n_redundant=4,
        weights=[0.95, 0.05], random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)

    # 2. Get predicted probabilities for positive class (Fraud)
    y_probs = model.predict_proba(X_test)[:, 1]

    # 3. Compute ROC Curve metrics and AUC Score
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    auc_score = roc_auc_score(y_test, y_probs)

    print(f"\n1. COMPUTED ROC-AUC SCORE:")
    print("-" * 65)
    print(f"   • Area Under ROC Curve (AUC) = {auc_score:.4f} ({auc_score*100:.2f}%)")
    print("-" * 65)

    # 4. Plot ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='#1f77b4', lw=2.5, label=f'Logistic Regression (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='#d62728', lw=1.5, linestyle='--', label='Random Baseline (AUC = 0.5000)')

    plt.title('ROC Curve - Credit Card Fraud Classifier', fontsize=14, fontweight='bold')
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR / Recall)', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower right', fontsize=11)
    plt.tight_layout()

    output_img = "session_17/task_2/roc_curve.png"
    plt.savefig(output_img, dpi=300)
    plt.close()

    print(f"\n2. ROC Plot saved successfully to: {output_img}")
    print("=" * 75)

if __name__ == "__main__":
    main()
