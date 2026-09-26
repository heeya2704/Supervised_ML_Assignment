# Session 12 - Task 3: Decision Tree Depth Limitation (`max_depth=2`)

## Task Overview
Train a pre-pruned `DecisionTreeClassifier` with `max_depth=2` on the Iris dataset and analyze how restricting tree depth alters structure and decision rules.

---

## Pruned Decision Tree Structure
![Pruned Decision Tree (max_depth=2)](file:///c:/Users/heeya/OneDrive/Documents/TOPS/Machine%20Learning/supervised_ML_Assignment/session_12/task_3/iris_tree_maxdepth2.png)

---

## Structural Comparison Matrix

| Structural Attribute | Full-Depth Decision Tree (`max_depth=None`) | Pruned Decision Tree (`max_depth=2`) |
|---|---|---|
| **Total Nodes** | 9–13 nodes | **5 nodes** |
| **Leaf Nodes** | 5–7 leaf nodes | **3 leaf nodes** |
| **Max Depth Level** | Depth 4–5 | **Depth 2** |
| **Test Accuracy** | 93.33% | **88.89%** |
| **Risk of Overfitting** | High (captures noise in training data) | **Low (generalized regularized model)** |

---

## Observed Decision Changes
1. **Simplified Splitting Hierarchy**:
   - Root split isolates `Setosa` perfectly using `petal length (cm) <= 2.45`.
   - Level 1 split separates `Versicolor` vs `Virginica` using `petal width (cm) <= 1.75`.
2. **Early Stopping & Regularization**:
   - The tree halts further splitting at depth 2 even though one leaf contains a small mixture of 3 versicolor samples alongside virginica samples.
   - Pre-pruning prevents the model from creating hyper-specific single-sample leaves.
