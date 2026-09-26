"""
Session 18 - Task 1: 5-Fold Cross-Validation on Iris Dataset with KNN
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

def main():
    print("=" * 75)
    print("SESSION 18 - TASK 1: 5-Fold Cross-Validation with KNN on Iris Dataset")
    print("=" * 75)

    # 1. Load Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    print(f"\n1. DATASET OVERVIEW:")
    print(f"   • Total Samples  : {X.shape[0]}")
    print(f"   • Features Count : {X.shape[1]}")
    print(f"   • Target Classes : {list(iris.target_names)}")

    # 2. Instantiate KNN Classifier (k=5)
    knn = KNeighborsClassifier(n_neighbors=5)

    # 3. Perform 5-Fold Cross-Validation
    cv_scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')

    print(f"\n2. 5-FOLD CROSS-VALIDATION RESULTS:")
    print("-" * 65)
    for fold, score in enumerate(cv_scores, 1):
        print(f"   • Fold {fold}: Accuracy = {score:.4f} ({score*100:.2f}%)")
    print("-" * 65)
    print(f"   • Mean CV Accuracy = {cv_scores.mean():.4f} ({cv_scores.mean()*100:.2f}%)")
    print(f"   • Std Deviation    = ±{cv_scores.std():.4f}")
    print("=" * 75)

if __name__ == "__main__":
    main()
