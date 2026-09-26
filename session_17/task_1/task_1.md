# Session 17 - Task 1: Train Logistic Regression on Imbalanced Fraud Dataset

## Task Overview
Download/generate an imbalanced Credit Card Fraud dataset (95% Legitimate transactions, 5% Fraudulent transactions), perform stratified train/test split, and train a baseline `LogisticRegression` classifier.

---

## Dataset & Model Configuration

- **Dataset Size**: 2,000 transactions (12 features)
- **Class Distribution**:
  - Class 0 (Legitimate): 1,900 samples (95.0%)
  - Class 1 (Fraud): 100 samples (5.0%)
- **Train/Test Split**: 70% Train (1,400 samples) / 30% Test (600 samples) stratified by target class.
- **Model**: `LogisticRegression(max_iter=1000, random_state=42)`

---

## Model Evaluation Baseline

```text
               precision    recall  f1-score   support

Legitimate (0)       0.97      0.99      0.98       570
     Fraud (1)       0.77      0.57      0.65        30

      accuracy                           0.97       600
     macro avg       0.87      0.78      0.82       600
  weighted avg       0.96      0.97      0.96       600
```

### Insights
- **High Accuracy (97.0%)**: Driven primarily by predicting the majority legitimate class ($TN$).
- **Minority Class Recall (57.0%)**: Misses 43% of fraudulent transactions ($FN=13$), highlighting the need for dedicated imbalanced metric evaluation.
