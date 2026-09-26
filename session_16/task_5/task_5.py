"""
Session 16 - Task 5: Receiver Operating Characteristic (ROC) Curve and AUC Score Evaluation
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score

def main():
    print("=" * 75)
    print("SESSION 16 - TASK 5: ROC Curve & Area Under Curve (AUC) Evaluation")
    print("=" * 75)

    # 1. Generate imbalanced synthetic dataset (e.g. Fraud Detection scenario)
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_classes=2,
        weights=[0.85, 0.15],
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 2. Train Logistic Regression Classifier
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    # 3. Predict class probabilities
    y_probs = model.predict_proba(X_test)[:, 1]

    # 4. Compute ROC Curve metrics and AUC score
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    auc_score = roc_auc_score(y_test, y_probs)

    print(f"\n1. DATASET & MODEL DETAILS:")
    print(f"   • Total Test Samples : {len(y_test)}")
    print(f"   • Positive Class (Fraud) Count : {sum(y_test)}")
    print(f"   • Negative Class (Legit) Count : {len(y_test) - sum(y_test)}")

    print(f"\n2. ROC-AUC PERFORMANCE SCORE:")
    print("-" * 65)
    print(f"   • Computed Area Under Curve (AUC) Score = {auc_score:.4f} ({auc_score*100:.2f}%)")
    print("-" * 65)

    # Find optimal threshold using Youden's J statistic (TPR - FPR)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]
    best_fpr = fpr[best_idx]
    best_tpr = tpr[best_idx]

    print(f"\n3. OPTIMAL DECISION THRESHOLD ANALYSIS (Youden's J Statistic):")
    print(f"   • Optimal Probability Threshold = {best_threshold:.4f}")
    print(f"   • True Positive Rate (TPR / Recall) at Optimal Threshold = {best_tpr:.4f}")
    print(f"   • False Positive Rate (FPR) at Optimal Threshold       = {best_fpr:.4f}")

    # 5. Plot ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='#1f77b4', lw=2.5, label=f'Logistic Regression (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], color='#d62728', lw=1.5, linestyle='--', label='Random Classifier Baseline (AUC = 0.500)')
    
    # Highlight optimal threshold point
    plt.scatter(best_fpr, best_tpr, color='#2ca02c', s=100, zorder=5, 
                label=f'Optimal Threshold ({best_threshold:.2f})')

    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve - Fraud Detection', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()

    # Save plot
    output_img = "session_16/task_5/roc_curve.png"
    plt.savefig(output_img, dpi=300)
    plt.close()
    print(f"\n4. Plot saved successfully to: {output_img}")
    print("=" * 75)

if __name__ == "__main__":
    main()
