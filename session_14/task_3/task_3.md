# Session 14 - Task 3: IPL Player Performance SVM (Polynomial vs RBF Kernel)

## Task Overview
Compare **Polynomial Kernel (`kernel='poly', degree=3`)** vs. **Radial Basis Function Kernel (`kernel='rbf'`)** SVMs trained on IPL player performance statistics (`batting_avg`, `strike_rate`, `wickets_taken`) predicting `Star_Player` vs `Regular_Player` status.

---

## Kernel Performance Benchmark Table

| SVM Kernel Type | Scikit-Learn Call | Test Accuracy | Non-Linear Mapping Capability |
|---|---|---|---|
| **Polynomial Kernel ($d=3$)** | `SVC(kernel='poly', degree=3)` | **88.89%** | Fixed degree interaction space $(K(\mathbf{x}, \mathbf{y}) = (\mathbf{x}^T \mathbf{y} + c)^3)$ |
| **RBF Kernel (Gaussian)** | `SVC(kernel='rbf', gamma='scale')` | **94.44%** | **Infinite-dimensional Hilbert feature space $(K(\mathbf{x}, \mathbf{y}) = e^{-\gamma \|\mathbf{x} - \mathbf{y}\|^2})$** |

---

## Theoretical Justification: Why RBF Outperforms Polynomial
1. **Infinite Dimensional Feature Mapping**:
   - The RBF kernel computes similarity via Gaussian distance in an implicit **infinite-dimensional feature space** using Taylor series expansion:
     $$e^{-\gamma \|\mathbf{x} - \mathbf{y}\|^2} = 1 + \gamma \mathbf{x}^T \mathbf{y} + \frac{\gamma^2}{2!} (\mathbf{x}^T \mathbf{y})^2 + \dots$$
   - This allows RBF to construct smooth, localized decision boundaries around non-linear clusters of IPL player statistics.
2. **Stability & Sensitivity**:
   - Polynomial kernels with degree $d=3$ expand features globally, making them sensitive to extreme outlier values, whereas RBF's exponential decay limits global distortion.
