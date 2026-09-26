"""
Session 19 - Task 5: AI-Suggested Parameter Grid Tuning for RandomForestClassifier
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score

def main():
    print("=" * 75)
    print("SESSION 19 - TASK 5: Comprehensive Parameter Grid Tuning for Random Forest")
    print("=" * 75)

    # 1. Load Breast Cancer Dataset
    data = load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    print(f"\n1. DATASET OVERVIEW (Breast Cancer Diagnostic Dataset):")
    print(f"   • Training Samples: {X_train.shape[0]}")
    print(f"   • Testing Samples : {X_test.shape[0]}")
    print(f"   • Feature Count   : {X.shape[1]}")

    # 2. Comprehensive AI-Suggested Parameter Grid
    # Parameter Rationale:
    # - n_estimators: [50, 100, 200] tests model capacity vs computational speed.
    # - max_depth: [3, 6, 10, None] controls tree growth & prevents overfitting.
    # - min_samples_split: [2, 5, 10] controls minimum samples required to split an internal node.
    # - min_samples_leaf: [1, 2, 4] ensures leaf nodes have sufficient statistical support.
    # - criterion: ['gini', 'entropy'] tests node splitting impurity functions.
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 6, 10, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'criterion': ['gini', 'entropy']
    }

    print(f"\n2. PARAMETER GRID DEFINITION:")
    for param, values in param_grid.items():
        print(f"   • {param:<18}: {values}")

    total_combinations = np.prod([len(v) for v in param_grid.values()])
    print(f"\n   Total Hyperparameter Combinations to Evaluate: {total_combinations}")
    print(f"   Total Folds (5-Fold CV): {total_combinations * 5} fits")

    # 3. Fit GridSearchCV
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1
    )

    print(f"\n3. RUNNING GRIDSEARCHCV...")
    grid_search.fit(X_train, y_train)

    print(f"\n4. BEST HYPERPARAMETERS FOUND:")
    print("-" * 65)
    for param, val in grid_search.best_params_.items():
        print(f"   • {param:<18} = {val}")
    print(f"   • Best 5-Fold Cross-Val F1 = {grid_search.best_score_:.4f} ({grid_search.best_score_*100:.2f}%)")
    print("-" * 65)

    # 4. Evaluate on Test Set
    best_rf = grid_search.best_estimator_
    y_pred = best_rf.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred)

    print(f"\n5. TEST SET EVALUATION:")
    print(f"   • Test Accuracy = {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"   • Test F1-Score = {test_f1:.4f} ({test_f1*100:.2f}%)\n")
    print(classification_report(y_test, y_pred, target_names=data.target_names))
    print("=" * 75)

if __name__ == "__main__":
    main()
