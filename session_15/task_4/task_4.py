"""
Session 15 - Task 4: Tuning n_estimators and learning_rate in GradientBoostingClassifier
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

def main():
    print("=" * 80)
    print("SESSION 15 - TASK 4: Tuning GradientBoosting Hyperparameters")
    print("=" * 80)

    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.3, random_state=42, stratify=iris.target
    )

    configurations = [
        {'n_estimators': 10, 'learning_rate': 0.01},
        {'n_estimators': 50, 'learning_rate': 0.1},
        {'n_estimators': 100, 'learning_rate': 0.1},
        {'n_estimators': 200, 'learning_rate': 0.05}
    ]

    best_acc = 0.0
    best_config = None

    print("\nHYPERPARAMETER BENCHMARK RESULTS:")
    print("-" * 65)
    print(f"{'n_estimators':<15} | {'learning_rate':<15} | {'Test Accuracy':<18}")
    print("-" * 65)

    for config in configurations:
        gbc = GradientBoostingClassifier(
            n_estimators=config['n_estimators'],
            learning_rate=config['learning_rate'],
            random_state=42
        )
        gbc.fit(X_train, y_train)
        acc = accuracy_score(y_test, gbc.predict(X_test))
        
        print(f"{config['n_estimators']:<15} | {config['learning_rate']:<15.2f} | {acc * 100:<10.2f}%")

        if acc > best_acc:
            best_acc = acc
            best_config = config

    print("-" * 65)
    print(f"\nBEST COMBINATION FOUND:")
    print(f"-> n_estimators = {best_config['n_estimators']}")
    print(f"-> learning_rate = {best_config['learning_rate']}")
    print(f"-> Best Accuracy = {best_acc * 100:.2f}%")
    print("=" * 80)

if __name__ == "__main__":
    main()
