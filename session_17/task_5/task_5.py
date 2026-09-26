"""
Session 17 - Task 5: High Imbalance (2% Positive Class) Metric Comparison
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, classification_report

def main():
    print("=" * 75)
    print("SESSION 17 - TASK 5: High Imbalance (2% Positive Class) Metric Evaluation")
    print("=" * 75)

    # 1. Generate dataset with extreme class imbalance (98% Legitimate, 2% Fraud)
    X, y = make_classification(
        n_samples=3000, n_features=12, n_informative=8, n_redundant=4,
        weights=[0.98, 0.02], random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Train Logistic Regression
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)

    # Predict classes and probabilities
    y_preds = model.predict(X_test)
    y_probs = model.predict_proba(X_test)

    # 2. Calculate Metrics
    acc = accuracy_score(y_test, y_preds)
    auc = roc_auc_score(y_test, y_probs[:, 1])
    loss = log_loss(y_test, y_probs)

    print(f"\n1. EXTREME IMBALANCE TEST SET DETAILS:")
    print(f"   • Total Test Samples : {len(y_test)}")
    print(f"   • Class 0 (Legit)    : {(y_test == 0).sum()} ({(y_test == 0).mean()*100:.1f}%)")
    print(f"   • Class 1 (Fraud)    : {(y_test == 1).sum()} ({(y_test == 1).mean()*100:.1f}%)")

    print(f"\n2. RE-CALCULATED EVALUATION METRICS:")
    print("-" * 65)
    print(f"   • Accuracy  : {acc:.4f} ({acc*100:.2f}%)")
    print(f"   • ROC-AUC   : {auc:.4f} ({auc*100:.2f}%)")
    print(f"   • Log Loss  : {loss:.4f}")
    print("-" * 65)

    print("\n3. CLASSIFICATION REPORT:")
    print(classification_report(y_test, y_preds, target_names=["Legit (0)", "Fraud (1)"]))

    print("\nTWO-LINE EXPLANATION OF MISLEADING METRIC:")
    print("Accuracy is the most misleading metric under extreme class imbalance because a naive model predicting 100% legitimate transactions achieves 98% accuracy while completely failing to detect any fraudulent transactions.")
    print("In contrast, ROC-AUC and Log Loss evaluate ranked probability predictions and minority class recall, providing a far more realistic measure of model effectiveness.")
    print("=" * 75)

if __name__ == "__main__":
    main()
