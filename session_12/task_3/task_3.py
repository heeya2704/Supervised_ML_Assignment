"""
Session 12 - Task 3: Max-Depth Pruned Decision Tree (max_depth=2)
"""

import os
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

def main():
    print("=" * 75)
    print("SESSION 12 - TASK 3: Pruned Decision Tree (max_depth=2)")
    print("=" * 75)

    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.3, random_state=42, stratify=iris.target
    )

    # Train max_depth=2 DecisionTree
    clf_pruned = DecisionTreeClassifier(max_depth=2, random_state=42)
    clf_pruned.fit(X_train, y_train)

    acc = accuracy_score(y_test, clf_pruned.predict(X_test))

    print(f"\n1. Max Depth Setting: max_depth=2")
    print(f"2. Total Tree Nodes: {clf_pruned.tree_.node_count}")
    print(f"3. Pruned Tree Test Accuracy: {acc * 100:.2f}%\n")

    # Plot Pruned Decision Tree
    plt.figure(figsize=(10, 6), dpi=300)
    plot_tree(
        clf_pruned,
        feature_names=iris.feature_names,
        class_names=iris.target_names,
        filled=True,
        rounded=True,
        fontsize=11
    )
    plt.title("Iris Pruned Decision Tree (max_depth=2)", fontsize=14, fontweight='bold', pad=15)

    output_dir = os.path.dirname(__file__)
    file_path = os.path.join(output_dir, 'iris_tree_maxdepth2.png')
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()

    print(f"Pruned decision tree plot saved to: {file_path}")
    print("=" * 75)

if __name__ == "__main__":
    main()
