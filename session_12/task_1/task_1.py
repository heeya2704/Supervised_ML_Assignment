"""
Session 12 - Task 1: Decision Tree Classifier on Iris Dataset & Visualization
"""

import os
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

def main():
    print("=" * 75)
    print("SESSION 12 - TASK 1: Iris Decision Tree Classifier & Plotting")
    print("=" * 75)

    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Train full-depth DecisionTreeClassifier
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n1. Training Samples: {len(X_train)} | Test Samples: {len(X_test)}")
    print(f"2. Features: {list(feature_names)}")
    print(f"3. Model Accuracy: {acc * 100:.2f}%\n")

    # Plot Decision Tree
    plt.figure(figsize=(14, 10), dpi=300)
    plot_tree(
        clf,
        feature_names=feature_names,
        class_names=target_names,
        filled=True,
        rounded=True,
        fontsize=10
    )
    plt.title("Iris Full-Depth Decision Tree Classifier (Default Gini)", fontsize=14, fontweight='bold', pad=15)

    output_dir = os.path.dirname(__file__)
    file_path = os.path.join(output_dir, 'iris_decision_tree.png')
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()

    print(f"Decision tree plot successfully generated and saved to: {file_path}")
    print("=" * 75)

if __name__ == "__main__":
    main()
