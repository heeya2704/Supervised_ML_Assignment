"""
Session 19 - Task 3: SMOTE Oversampling + GridSearchCV Hyperparameter Tuning
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, recall_score
from imblearn.over_sampling import SMOTE

def main():
    print("=" * 75)
    print("SESSION 19 - TASK 3: SMOTE Oversampling + GridSearchCV Tuning")
    print("=" * 75)

    # 1. Create Imbalanced Rare Event Dataset (90% Majority, 10% Minority Class)
    X, y = make_classification(
        n_samples=2000, n_features=10, n_informative=8, n_redundant=2,
        weights=[0.90, 0.10], random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"\n1. ORIGINAL TRAIN SET CLASS DISTRIBUTION:")
    print(f"   • Majority Class (0) : {(y_train == 0).sum()} ({(y_train == 0).mean()*100:.1f}%)")
    print(f"   • Minority Class (1) : {(y_train == 1).sum()} ({(y_train == 1).mean()*100:.1f}%)")

    # 2. Apply SMOTE Oversampling on Training Data ONLY
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print(f"\n2. RESAMPLED TRAIN SET CLASS DISTRIBUTION (AFTER SMOTE):")
    print(f"   • Majority Class (0) : {(y_train_resampled == 0).sum()}")
    print(f"   • Minority Class (1) : {(y_train_resampled == 1).sum()} (100% Balanced!)")

    # 3. Define GridSearchCV Parameter Grid
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [5, 10],
        'min_samples_split': [2, 5]
    }

    # 4. Fit GridSearchCV on SMOTE-resampled Training Data
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1
    )

    print(f"\n3. RUNNING GRIDSEARCHCV ON RESAMPLED DATA (Scoring = 'f1')...")
    grid_search.fit(X_train_resampled, y_train_resampled)

    print(f"\n4. BEST HYPERPARAMETERS & PERFORMANCE:")
    print("-" * 65)
    print(f"   • Best Hyperparameters : {grid_search.best_params_}")
    print(f"   • Best Cross-Val F1    : {grid_search.best_score_:.4f}")
    print("-" * 65)

    # 5. Evaluate on Original Imbalanced Test Set
    best_rf = grid_search.best_estimator_
    y_pred = best_rf.predict(X_test)

    test_f1 = f1_score(y_test, y_pred)
    test_recall = recall_score(y_test, y_pred)

    print(f"\n5. TEST SET EVALUATION (Original Imbalanced Test Data):")
    print(f"   • Minority Class Recall = {test_recall:.4f} ({test_recall*100:.2f}%)")
    print(f"   • Minority Class F1     = {test_f1:.4f} ({test_f1*100:.2f}%)\n")
    print(classification_report(y_test, y_pred, target_names=["Majority (0)", "Minority (1)"]))
    print("=" * 75)

if __name__ == "__main__":
    main()
