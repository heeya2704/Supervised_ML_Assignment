# Session 15 - Task 2: AdaBoost Sentiment Classifier on Flipkart Reviews

## Task Overview
Train an **AdaBoost Classifier (`AdaBoostClassifier`)** using 1-level Decision Stumps (`DecisionTreeClassifier(max_depth=1)`) on Flipkart product review text vectors.

---

## Model Setup & Evaluation

- **Algorithm**: `AdaBoostClassifier(algorithm='SAMME')`
- **Base Estimator**: Decision Stump (`max_depth=1`)
- **Number of Estimators**: 50 weak learners
- **Overall Test Accuracy**: **66.67%**

---

## Detailed Classification Report Table

| Class Label | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| **`negative`** | **1.00** | 0.33 | 0.50 | 3 |
| **`positive`** | 0.60 | **1.00** | **0.75** | 3 |
| **Macro Average** | **0.80** | **0.67** | **0.62** | 6 |
| **Weighted Average** | **0.80** | **0.67** | **0.62** | 6 |

---

## AdaBoost Adaptive Weighting Formula
After fitting each weak learner $h_m(\mathbf{x})$, sample weights $w_i$ are updated for the next iteration:

$$w_i^{(m+1)} = w_i^{(m)} \exp \left( \alpha_m \mathbb{I}(y_i \neq h_m(\mathbf{x}_i)) \right)$$

Misclassified reviews receive higher instance weights $w_i$, forcing subsequent decision stumps to focus on difficult boundary cases.
