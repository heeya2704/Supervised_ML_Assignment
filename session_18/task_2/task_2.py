"""
Session 18 - Task 2: 10-Fold Cross-Validation with DecisionTreeClassifier on Wine Dataset
"""

import numpy as np
from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

def main():
    print("=" * 75)
    print("SESSION 18 - TASK 2: 10-Fold Cross-Validation with Decision Tree")
    print("=" * 75)

    # 1. Load Wine dataset
    wine = load_wine()
    X, y = wine.data, wine.target

    print(f"\n1. DATASET OVERVIEW (Wine Classification):")
    print(f"   • Total Samples  : {X.shape[0]}")
    print(f"   • Features Count : {X.shape[1]}")
    print(f"   • Target Classes : {list(wine.target_names)}")

    # 2. Instantiate Decision Tree Classifier
    dt_clf = DecisionTreeClassifier(random_state=42)

    # 3. Perform 10-Fold Cross-Validation
    cv_scores = cross_val_score(dt_clf, X, y, cv=10, scoring='accuracy')

    print(f"\n2. 10-FOLD CROSS-VALIDATION INDIVIDUAL SCORES:")
    print("-" * 65)
    for fold_idx, score in enumerate(cv_scores, 1):
        print(f"   • Fold {fold_idx:2d}: Accuracy = {score:.4f} ({score*100:.2f}%)")
    print("-" * 65)
    print(f"   • Overall Mean Accuracy = {cv_scores.mean():.4f} ({cv_scores.mean()*100:.2f}%)")
    print(f"   • Standard Deviation    = ±{cv_scores.std():.4f}")
    print("=" * 75)

if __name__ == "__main__":
    main()
