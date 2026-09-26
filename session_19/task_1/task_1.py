"""
Session 19 - Task 1: GridSearchCV Hyperparameter Tuning for Product Reviews Classifier
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def main():
    print("=" * 75)
    print("SESSION 19 - TASK 1: GridSearchCV Tuning on Product Review Sentiment")
    print("=" * 75)

    # 1. Create synthetic Product Review Dataset (Features: review_length, word_count, rating_score, polarity)
    X, y = make_classification(
        n_samples=1500, n_features=6, n_informative=4, n_redundant=2,
        weights=[0.5, 0.5], random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"\n1. DATASET OVERVIEW (Product Reviews Sentiment):")
    print(f"   • Training Samples: {X_train.shape[0]}")
    print(f"   • Testing Samples : {X_test.shape[0]}")
    print(f"   • Target Classes  : Negative Review (0), Positive Review (1)")

    # 2. Define Parameter Grid for GridSearchCV
    param_grid = {
        'max_depth': [3, 5, 10, None],
        'n_estimators': [50, 100, 200]
    }

    # 3. Instantiate RandomForest and GridSearchCV
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )

    print(f"\n2. EXECUTING GRIDSEARCHCV (Tuning max_depth & n_estimators)...")
    grid_search.fit(X_train, y_train)

    print(f"\n3. GRIDSEARCHCV BEST RESULTS:")
    print("-" * 65)
    print(f"   • Best Hyperparameters : {grid_search.best_params_}")
    print(f"   • Best Cross-Val Score : {grid_search.best_score_:.4f} ({grid_search.best_score_*100:.2f}%)")
    print("-" * 65)

    # 4. Evaluate on Test Set
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)

    print(f"\n4. TEST SET PERFORMANCE:")
    print(f"   • Test Accuracy = {test_acc:.4f} ({test_acc*100:.2f}%)\n")
    print(classification_report(y_test, y_pred, target_names=["Negative Review (0)", "Positive Review (1)"]))
    print("=" * 75)

if __name__ == "__main__":
    main()
