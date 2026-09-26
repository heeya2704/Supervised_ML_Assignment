# Session 10 / Session 11 - Task 1: KNN Flipkart Review Sentiment Classifier

## Task Overview
Build a **K-Nearest Neighbors (KNN)** sentiment classifier to predict whether a Flipkart product review is `positive` or `negative`. Convert raw review strings into numerical frequency vectors using `CountVectorizer`.

---

## Workflow & Pipeline
1. **Text Vectorization**: Convert text to Bag-of-Words (BoW) feature matrix using `sklearn.feature_extraction.text.CountVectorizer(stop_words='english')`.
2. **Model Training**: Train `KNeighborsClassifier(n_neighbors=1, metric='euclidean')` on text vectors.
3. **Evaluation**: Predict on unseen test reviews.

---

## Model Evaluation Results

- **Dataset Size**: 20 product reviews (14 training, 6 testing)
- **Vocabulary Size**: 64 unique words (features)
- **Test Accuracy**: **83.33%**

| Review Snippet | Actual Label | Predicted Label | Status |
|---|---|---|---|
| *"Brilliant product, best battery life..."* | positive | positive | Correct |
| *"Top notch quality, super impressed..."* | positive | positive | Correct |
| *"Very good phone, highly recommended..."* | positive | positive | Correct |
| *"Horrible build quality, complete waste..."* | negative | negative | Correct |
| *"Bad product, poor performance..."* | negative | negative | Correct |
| *"Poor screen display and frequent battery..."* | negative | positive | Error |
