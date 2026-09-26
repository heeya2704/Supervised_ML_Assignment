"""
Session 17 - Task 4: Log Loss (Cross-Entropy Loss) Computation
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

def main():
    print("=" * 75)
    print("SESSION 17 - TASK 4: Log Loss (Cross-Entropy Loss) Calculation")
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

    # 2. Predict probabilities for test set
    y_probs = model.predict_proba(X_test)

    # 3. Calculate Log Loss
    test_log_loss = log_loss(y_test, y_probs)

    print(f"\n1. LOG LOSS CALCULATION RESULTS:")
    print("-" * 65)
    print(f"   • Test Set Log Loss (Cross-Entropy Loss) = {test_log_loss:.4f}")
    print("-" * 65)

    print("\n2. INTERPRETATION OF LOG LOSS:")
    print("Log loss quantifies the performance of a classification model whose output is a probability value between 0 and 1.")
    print("A lower log loss indicates predictions that are closer to the true binary labels (0 or 1), heavily penalizing confident incorrect predictions.")
    print("=" * 75)

if __name__ == "__main__":
    main()
