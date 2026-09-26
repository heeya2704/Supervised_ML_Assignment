"""
Session 13 - Task 1: Default RandomForestClassifier on Iris Dataset
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def main():
    print("=" * 75)
    print("SESSION 13 - TASK 1: Default RandomForestClassifier on Iris Dataset")
    print("=" * 75)

    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.25, random_state=42, stratify=iris.target
    )

    # Train RandomForestClassifier with default hyperparameters
    rf_clf = RandomForestClassifier(random_state=42)
    rf_clf.fit(X_train, y_train)

    y_pred = rf_clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n1. Training Samples: {len(X_train)} | Test Samples: {len(X_test)}")
    print(f"2. Number of Trees (n_estimators): {rf_clf.n_estimators}")
    print(f"3. Criterion: {rf_clf.criterion}")
    print(f"4. Model Test Accuracy: {acc * 100:.2f}%\n")
    print("=" * 75)

if __name__ == "__main__":
    main()
