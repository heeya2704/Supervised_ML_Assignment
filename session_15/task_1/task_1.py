"""
Session 15 - Task 1: GradientBoostingClassifier on Iris Dataset
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

def main():
    print("=" * 75)
    print("SESSION 15 - TASK 1: GradientBoostingClassifier on Iris Dataset")
    print("=" * 75)

    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.3, random_state=42, stratify=iris.target
    )

    # Train GradientBoostingClassifier with default settings
    gbc = GradientBoostingClassifier(random_state=42)
    gbc.fit(X_train, y_train)

    y_pred = gbc.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n1. Dataset Split: {len(X_train)} Train / {len(X_test)} Test")
    print(f"2. Number of Boosting Stages (n_estimators): {gbc.n_estimators}")
    print(f"3. Learning Rate: {gbc.learning_rate}")
    print(f"4. Model Test Accuracy: {acc * 100:.2f}%\n")
    print("=" * 75)

if __name__ == "__main__":
    main()
