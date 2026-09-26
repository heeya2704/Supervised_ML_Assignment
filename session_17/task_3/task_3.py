"""
Session 17 - Task 3: Precision-Recall Curve Plotting & Imbalanced Utility Analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, average_precision_score

def main():
    print("=" * 75)
    print("SESSION 17 - TASK 3: Precision-Recall Curve Visualization")
    print("=" * 75)

    # 1. Dataset Generation & Model Training
    X, y = make_classification(
        n_samples=2000, n_features=12, n_informative=8, n_redundant=4,
        weights=[0.95, 0.05], random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)

    # 2. Compute Precision-Recall Curve and Average Precision (AP)
    y_probs = model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
    ap_score = average_precision_score(y_test, y_probs)

    print(f"\n1. COMPUTED METRICS:")
    print("-" * 65)
    print(f"   • Average Precision (AP / PR-AUC Score) = {ap_score:.4f} ({ap_score*100:.2f}%)")
    print("-" * 65)

    # 3. Plot Precision-Recall Curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='#2ca02c', lw=2.5, label=f'Precision-Recall Curve (AP = {ap_score:.4f})')
    
    # Baseline for PR curve is the ratio of positive class (fraud)
    baseline_ratio = sum(y_test) / len(y_test)
    plt.axhline(y=baseline_ratio, color='#d62728', linestyle='--', label=f'No-Skill Baseline (Ratio = {baseline_ratio:.2f})')

    plt.title('Precision-Recall Curve - Credit Card Fraud Classifier', fontsize=14, fontweight='bold')
    plt.xlabel('Recall (Sensitivity / True Positive Rate)', fontsize=12)
    plt.ylabel('Precision (Positive Predictive Value)', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower left', fontsize=11)
    plt.tight_layout()

    output_img = "session_17/task_3/precision_recall_curve.png"
    plt.savefig(output_img, dpi=300)
    plt.close()

    print(f"\n2. PR Plot saved successfully to: {output_img}\n")

    print("ONE-LINE EXPLANATION OF PRECISION-RECALL CURVE UTILITY:")
    print("The Precision-Recall curve is essential for imbalanced datasets because it ignores True Negatives (TN) and focuses exclusively on performance relative to the rare positive class (fraud), preventing overly optimistic evaluations caused by large majority class counts.")
    print("=" * 75)

if __name__ == "__main__":
    main()
