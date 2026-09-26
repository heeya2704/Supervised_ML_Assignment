"""
Session 19 - Task 2: RandomizedSearchCV Tuning on Instagram Post Engagement Dataset
"""

import numpy as np
import pandas as pd
from scipy.stats import loguniform
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

def main():
    print("=" * 75)
    print("SESSION 19 - TASK 2: RandomizedSearchCV Tuning for Instagram Engagement")
    print("=" * 75)

    # 1. Create Instagram Post Engagement Dataset (caption_len, posting_hour, hashtags, followers)
    X, y = make_classification(
        n_samples=1200, n_features=5, n_informative=4, n_redundant=1,
        weights=[0.6, 0.4], random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"\n1. DATASET OVERVIEW (Instagram Engagement Prediction):")
    print(f"   • Training Samples: {X_train.shape[0]}")
    print(f"   • Testing Samples : {X_test.shape[0]}")
    print(f"   • Target Classes  : Low Engagement (0), High Engagement (1)")

    # 2. Define Continuous Hyperparameter Distributions for C and gamma
    param_distributions = {
        'C': loguniform(1e-2, 1e2),
        'gamma': loguniform(1e-3, 1e1),
        'kernel': ['rbf', 'sigmoid']
    }

    # 3. Instantiate SVM and RandomizedSearchCV (n_iter=10)
    svc = SVC(random_state=42)
    random_search = RandomizedSearchCV(
        estimator=svc,
        param_distributions=param_distributions,
        n_iter=10,  # Constraint: Limit to 10 iterations for fast search
        cv=5,
        scoring='accuracy',
        random_state=42,
        n_jobs=-1
    )

    print(f"\n2. EXECUTING RANDOMIZEDSEARCHCV (n_iter=10)...")
    random_search.fit(X_train, y_train)

    print(f"\n3. RANDOMSEARCHCV BEST HYPERPARAMETERS:")
    print("-" * 65)
    print(f"   • Best C Parameter     : {random_search.best_params_['C']:.4f}")
    print(f"   • Best Gamma Parameter : {random_search.best_params_['gamma']:.4f}")
    print(f"   • Best Kernel          : {random_search.best_params_['kernel']}")
    print(f"   • Best 5-Fold CV Score : {random_search.best_score_:.4f} ({random_search.best_score_*100:.2f}%)")
    print("-" * 65)

    # 4. Evaluate on Test Set
    best_svc = random_search.best_estimator_
    y_pred = best_svc.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)

    print(f"\n4. TEST SET EVALUATION:")
    print(f"   • Test Accuracy = {test_acc:.4f} ({test_acc*100:.2f}%)\n")
    print(classification_report(y_test, y_pred, target_names=["Low Engagement (0)", "High Engagement (1)"]))
    print("=" * 75)

if __name__ == "__main__":
    main()
