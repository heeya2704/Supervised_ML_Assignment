"""
Session 13 - Task 3: Hyperparameter Tuning (n_estimators vs max_depth) on Iris
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def main():
    print("=" * 80)
    print("SESSION 13 - TASK 3: Hyperparameter Grid Tuning for RandomForestClassifier")
    print("=" * 80)

    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.3, random_state=42, stratify=iris.target
    )

    n_estimators_list = [10, 50, 100]
    max_depth_list = [2, 4, 6]

    best_acc = 0.0
    best_params = None

    print("\nGRID SEARCH RESULTS:")
    print("-" * 65)
    print(f"{'n_estimators':<15} | {'max_depth':<15} | {'Test Accuracy':<18}")
    print("-" * 65)

    for n in n_estimators_list:
        for depth in max_depth_list:
            rf = RandomForestClassifier(n_estimators=n, max_depth=depth, random_state=42)
            rf.fit(X_train, y_train)
            acc = accuracy_score(y_test, rf.predict(X_test))
            
            print(f"{n:<15} | {depth:<15} | {acc * 100:<10.2f}%")

            if acc > best_acc:
                best_acc = acc
                best_params = (n, depth)

    print("-" * 65)
    print(f"\nOPTIMAL COMBINATION FOUND:")
    print(f"-> n_estimators = {best_params[0]}")
    print(f"-> max_depth    = {best_params[1]}")
    print(f"-> Best Accuracy = {best_acc * 100:.2f}%")
    print("=" * 80)

if __name__ == "__main__":
    main()
