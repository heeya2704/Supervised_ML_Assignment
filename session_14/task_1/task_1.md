# Session 14 - Task 1: Music Genre SVM Classifier & Decision Boundary

## Task Overview
Train a **Linear Support Vector Machine (`SVC(kernel='linear')`)** to separate two music genres (`Pop` vs `Classical`) based on audio features `tempo` and `danceability`. Visualize the hyper-plane decision boundary.

---

## SVM Decision Boundary Plot
![SVM Music Genre Decision Boundary](file:///c:/Users/heeya/OneDrive/Documents/TOPS/Machine%20Learning/supervised_ML_Assignment/session_14/task_1/svm_music_boundary.png)

---

## Mathematical Objective & Setup
$$\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2 \quad \text{subject to} \quad y_i (\mathbf{w}^T \mathbf{x}_i + b) \ge 1, \; \forall i$$

- **Features**: `tempo` (BPM) and `danceability` (0–1 scale), standardized via `StandardScaler()`.
- **Target Classes**: `Pop` (1) vs `Classical` (0).
- **Support Vectors Count**: 2 critical boundary points (circled in black on the plot) that uniquely define the optimal separating margin.
