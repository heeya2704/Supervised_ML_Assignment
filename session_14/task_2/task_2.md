# Session 14 - Task 2: Linear SVM Classifier on Flipkart Reviews

## Task Overview
Classify Flipkart product review sentiment (`positive` vs `negative`) using a **Linear Support Vector Classifier (`SVC(kernel='linear')`)** and `CountVectorizer`.

---

## Linear SVM Formulation
Linear kernel decision boundary:

$$f(\mathbf{x}) = \mathbf{w}^T \mathbf{x} + b = \sum_{i=1}^{N} \alpha_i y_i (\mathbf{x}_i^T \mathbf{x}) + b$$

- **Feature Vector**: Bag-of-Words word frequencies (64 features)
- **Model Hyperparameter**: `kernel='linear'`, $C = 1.0$
- **Test Accuracy**: **83.33%**

---

## Prediction Results Table

| Review Snippet | Actual Label | Predicted Label | Evaluation |
|---|---|---|---|
| *"Top notch quality, super impressed..."* | positive | positive | Correct |
| *"Brilliant product, best battery life..."* | positive | positive | Correct |
| *"Very good phone, highly recommended..."* | positive | positive | Correct |
| *"Horrible build quality, complete waste..."* | negative | negative | Correct |
| *"Bad product, poor performance..."* | negative | negative | Correct |
| *"Poor screen display and frequent battery..."* | negative | positive | Misclassification |
