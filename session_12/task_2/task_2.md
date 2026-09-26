# Session 12 - Task 2: Decision Tree Splitting Criteria (Gini vs. Entropy)

## Task Overview
Compare `DecisionTreeClassifier` performance on the Iris dataset using **Gini Impurity** (`criterion='gini'`) vs. **Information Gain / Entropy** (`criterion='entropy'`).

---

## Mathematical Formulation

1. **Gini Impurity**:
   $$I_G(p) = 1 - \sum_{i=1}^{K} p_i^2$$
   - Measures likelihood of misclassifying a randomly chosen sample.

2. **Entropy (Information Gain)**:
   $$H(p) = -\sum_{i=1}^{K} p_i \log_2(p_i)$$
   - Measures expected information / disorder in bit units.

---

## Performance Comparison

| Splitting Criterion | Parameter | Test Accuracy | Computational Complexity |
|---|---|---|---|
| **Gini Impurity** | `criterion='gini'` | **93.33%** | Lower (Polynomial arithmetic) |
| **Entropy** | `criterion='entropy'` | **88.89%** | Higher (Logarithmic computation) |

---

## Observations
- **Gini Impurity** is slightly faster to compute because it avoids logarithmic calls $\log_2(p_i)$.
- Both criteria split nodes by maximizing purity improvement, but Gini tends to isolate the largest class first, leading to a slightly higher test accuracy on this split.
