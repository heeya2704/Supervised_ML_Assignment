# Session 13 - Task 1: Default RandomForestClassifier on Iris Dataset

## Task Overview
Train an ensemble `RandomForestClassifier` with default parameters on the Iris dataset and measure test set classification accuracy.

---

## Model Hyperparameters & Setup
- **Algorithm**: `sklearn.ensemble.RandomForestClassifier`
- **Number of Decision Trees (`n_estimators`)**: `100` (default)
- **Splitting Criterion (`criterion`)**: `'gini'` (default)
- **Dataset Split**: 75% Training (112 samples) / 25% Testing (38 samples)

---

## Results
- **Test Set Accuracy**: **92.11%**
- **Ensemble Advantage**: Aggregates predictions across 100 individual decision trees trained on bootstrap sub-samples to minimize variance.
