# Session 15 - Task 4: Gradient Boosting Hyperparameter Tuning

## Task Overview
Tune `n_estimators` (number of boosting trees) and `learning_rate` ($\eta$) parameters of `GradientBoostingClassifier` on the Iris dataset.

---

## Benchmark Results Table

| `n_estimators` | `learning_rate` ($\eta$) | Test Accuracy | Generalization Behavior |
|---|---|---|---|
| **10** | **0.01** | **97.78%** | Excellent (Early stopping regularization) |
| **50** | **0.10** | **97.78%** | **Optimal balance** |
| 100 | 0.10 | 93.33% | Minor over-fitting to training split |
| 200 | 0.05 | 93.33% | Minor over-fitting |

---

## Theoretical Interaction between `learning_rate` and `n_estimators`
In Gradient Boosting, each tree $h_m(\mathbf{x})$ predicts the negative gradient of the loss function:

$$F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \eta \cdot h_m(\mathbf{x})$$

- **Shrinkage Factor ($\eta$)**: Small learning rates ($\eta = 0.10$ or $0.05$) shrink the contribution of each tree, requiring more boosting stages but yielding better test generalization and lower variance.
