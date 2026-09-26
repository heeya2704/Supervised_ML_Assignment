"""
Session 12 - Task 4: Iris Feature Importance Analysis via Decision Tree
"""

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

def main():
    print("=" * 75)
    print("SESSION 12 - TASK 4: Feature Importance Ranking (Decision Tree)")
    print("=" * 75)

    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.3, random_state=42, stratify=iris.target
    )

    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # Feature Importance analysis
    importances = clf.feature_importances_
    feat_imp = pd.Series(importances, index=iris.feature_names).sort_values(ascending=False)

    print("\nSORTED FEATURE IMPORTANCES (MOST TO LEAST IMPORTANT):")
    print("-" * 65)
    print(f"{'Rank':<6} | {'Feature Name':<25} | {'Importance Score':<18} | {'Percentage':<12}")
    print("-" * 65)

    for rank, (feat, score) in enumerate(feat_imp.items(), 1):
        print(f"#{rank:<5} | {feat:<25} | {score:<18.6f} | {score * 100:<10.2f}%")

    print("-" * 65)
    print("\nMATHEMATICAL RATIONALE:")
    print("Gini Importance (Mean Decrease Impurity) measures total reduction of Gini impurity brought by a feature across all tree splits.")
    print("Petal metrics (length and width) provide over 95% of total predictive information for Iris species classification.")
    print("=" * 75)

if __name__ == "__main__":
    main()
