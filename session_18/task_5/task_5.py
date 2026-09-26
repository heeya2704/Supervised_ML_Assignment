"""
Session 18 - Task 5: AI-Assisted Generic Cross-Validation Evaluation Function
"""

import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

def evaluate_classifier_cv(model, X, y, cv=5, scoring='accuracy'):
    """
    Generic AI-assisted utility function to evaluate any scikit-learn classifier
    using k-fold cross-validation.
    
    Parameters:
        model: Estimator object implementing 'fit'
        X (array-like): Feature matrix
        y (array-like): Target array
        cv (int): Number of cross-validation folds (default=5)
        scoring (str): Evaluation metric (default='accuracy')
        
    Returns:
        tuple: (mean_score, std_score, raw_scores_array)
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    return mean_score, std_score, scores

def main():
    print("=" * 75)
    print("SESSION 18 - TASK 5: Generic AI-Enhanced CV Evaluation Utility")
    print("=" * 75)

    # Test function on multiple models and datasets
    benchmarks = [
        ("Iris Dataset", load_iris(), LogisticRegression(max_iter=500, random_state=42)),
        ("Wine Dataset", load_wine(), RandomForestClassifier(n_estimators=50, random_state=42)),
        ("Breast Cancer", load_breast_cancer(), GradientBoostingClassifier(random_state=42))
    ]

    print("\n1. BENCHMARK EXECUTION RESULTS:")
    print("-" * 75)
    print(f"{'Dataset':<18} | {'Classifier':<28} | {'Mean Accuracy':<15} | {'Std Dev':<10}")
    print("-" * 75)

    for name, data, clf in benchmarks:
        mean_acc, std_acc, _ = evaluate_classifier_cv(clf, data.data, data.target, cv=5)
        clf_name = clf.__class__.__name__
        print(f"{name:<18} | {clf_name:<28} | {mean_acc*100:.2f}% ({mean_acc:.4f}) | ±{std_acc:.4f}")

    print("-" * 75)
    print("\n2. HOW THE AI TOOL IMPROVED THE SOLUTION:")
    print("• Flexibility: Added parameterized metric selection ('scoring') allowing seamless evaluation of F1, ROC-AUC, or Accuracy.")
    print("• Robust Output: Returned both aggregated statistics (mean, std) and raw fold scores for diagnostic plotting.")
    print("• Type Compatibility: Accepted any scikit-learn compatible estimator and NumPy array/DataFrame seamlessly.")
    print("=" * 75)

if __name__ == "__main__":
    main()
