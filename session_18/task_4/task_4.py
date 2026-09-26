"""
Session 18 - Task 4: Stability Analysis across CV Folds (cv = 3, 5, 10)
"""

import numpy as np
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

def main():
    print("=" * 75)
    print("SESSION 18 - TASK 4: RandomForest Cross-Validation Stability Analysis")
    print("=" * 75)

    # 1. Load Wine dataset
    wine = load_wine()
    X, y = wine.data, wine.target

    # 2. Instantiate Random Forest Classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)

    # 3. Test cv = 3, 5, 10
    cv_values = [3, 5, 10]
    results = {}

    print(f"\n1. EXPERIMENTAL COMPARISON Across CV Fold Sizes:")
    print("-" * 68)
    print(f"{'CV Folds (k)':<15} | {'Mean Accuracy':<18} | {'Std Dev (Variance)':<20} | {'Stability Rank':<15}")
    print("-" * 68)

    for cv in cv_values:
        scores = cross_val_score(rf, X, y, cv=cv, scoring='accuracy')
        mean_acc = scores.mean()
        std_acc = scores.std()
        results[cv] = (mean_acc, std_acc, scores)
        print(f"k = {cv:<11} | {mean_acc*100:.2f}% ({mean_acc:.4f})  | ±{std_acc:.4f}             | ")

    print("-" * 68)

    # Determine most stable cv value
    most_stable_cv = min(results.keys(), key=lambda k: results[k][1])
    
    print(f"\n2. STABILITY CONCLUSION:")
    print(f"   • Most Stable (Least Variable) Fold Setting: cv = {most_stable_cv}")
    print(f"   • Standard Deviation for cv={most_stable_cv}: ±{results[most_stable_cv][1]:.4f}")
    print("   • Reason: A lower standard deviation across fold iterations indicates higher score stability and lower evaluation variance.")
    print("=" * 75)

if __name__ == "__main__":
    main()
