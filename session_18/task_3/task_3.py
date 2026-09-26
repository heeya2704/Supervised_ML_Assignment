"""
Session 18 - Task 3: Target Class Distribution Verification with StratifiedKFold
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold, KFold

def main():
    print("=" * 75)
    print("SESSION 18 - TASK 3: StratifiedKFold vs Standard KFold Verification")
    print("=" * 75)

    # 1. Load Iris Dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    print("\n1. STRATIFIED K-FOLD CLASS DISTRIBUTION (5 Folds):")
    print("-" * 65)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        test_y = y[test_idx]
        counts = np.bincount(test_y, minlength=3)
        print(f"   • Fold {fold} Test Set Class Counts -> Class 0: {counts[0]}, Class 1: {counts[1]}, Class 2: {counts[2]} (Total: {len(test_y)})")

    print("-" * 65)
    print("\n2. EXPLANATION OF STRATIFIED SAMPLING:")
    print("StratifiedKFold ensures that each fold maintains the exact same class distribution percentage as the complete dataset (10 samples of Class 0, 10 of Class 1, and 10 of Class 2 per fold).")
    print("Standard non-stratified KFold can produce skewed folds with missing or over-represented classes, leading to biased performance evaluation.")
    print("=" * 75)

if __name__ == "__main__":
    main()
