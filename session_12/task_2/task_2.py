"""
Session 12 - Task 2: Splitting Criterion Comparison (Gini vs Entropy)
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def main():
    print("=" * 75)
    print("SESSION 12 - TASK 2: Splitting Criterion (Gini vs Entropy)")
    print("=" * 75)

    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.3, random_state=42, stratify=iris.target
    )

    # 1. Gini Impurity Criterion
    clf_gini = DecisionTreeClassifier(criterion='gini', random_state=42)
    clf_gini.fit(X_train, y_train)
    acc_gini = accuracy_score(y_test, clf_gini.predict(X_test))

    # 2. Entropy / Information Gain Criterion
    clf_entropy = DecisionTreeClassifier(criterion='entropy', random_state=42)
    clf_entropy.fit(X_train, y_train)
    acc_entropy = accuracy_score(y_test, clf_entropy.predict(X_test))

    print("\nPERFORMANCE MATRIX:")
    print("-" * 65)
    print(f"{'Splitting Criterion':<25} | {'Mathematical Formula':<25} | {'Test Accuracy':<12}")
    print("-" * 65)
    print(f"{'Gini Impurity':<25} | {'1 - SUM(p_i^2)':<25} | {acc_gini * 100:<10.2f}%")
    print(f"{'Entropy (Information Gain)':<25} | {'- SUM(p_i * log2(p_i))':<25} | {acc_entropy * 100:<10.2f}%")
    print("-" * 65)

    print("\nTHEORETICAL COMPARISON:")
    print("1. Gini Impurity: Measures probability of misclassifying a randomly chosen element. Computationally faster (no logarithmic calculations).")
    print("2. Entropy (Information Gain): Measures disorder/information gain in bit units. Computationally slightly slower due to log2().")
    print("3. Practice: Gini and Entropy yield identical or nearly identical decision boundaries in ~95% of classification tasks.")
    print("=" * 75)

if __name__ == "__main__":
    main()
