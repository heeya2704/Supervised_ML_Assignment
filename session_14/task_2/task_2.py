"""
Session 14 - Task 2: Linear SVM Classifier on Flipkart Product Reviews
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def main():
    print("=" * 75)
    print("SESSION 14 - TASK 2: Linear SVM Flipkart Review Sentiment Classifier")
    print("=" * 75)

    reviews = [
        "Great quality product, fast delivery and excellent build quality!",
        "Awesome battery life and smooth display screen, love it",
        "Very good phone, highly recommended positive buy for gaming",
        "Superb performance, excellent value for money purchase",
        "Loved the packaging and premium camera quality overall",
        "Fantastic experience, great sound clarity and heavy bass",
        "Brilliant product, best battery life and super fast charger",
        "Awesome purchase, highly satisfied with Flipkart speed",
        "Excellent product, works perfectly and high quality",
        "Top notch quality, super impressed with performance",
        "Worst product ever, defective piece received",
        "Extremely slow delivery and terrible customer support experience",
        "Useless item, broke within two days of usage worst build",
        "Horrible build quality, complete waste of money terrible",
        "Poor screen display and frequent battery heating issues bad",
        "Very disappointing experience, damaged box on arrival terrible",
        "Bad product, poor performance and awful customer support",
        "Defective piece, waste of money terrible service overall",
        "Worst battery backup, cheap plastic quality highly disappointed",
        "Pathetic item, useless product and horrible seller service"
    ]
    labels = [
        "positive", "positive", "positive", "positive", "positive",
        "positive", "positive", "positive", "positive", "positive",
        "negative", "negative", "negative", "negative", "negative",
        "negative", "negative", "negative", "negative", "negative"
    ]

    df = pd.DataFrame({'review': reviews, 'label': labels})

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df['review'], df['label'], test_size=0.3, random_state=42, stratify=df['label']
    )

    vectorizer = CountVectorizer(stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train_raw)
    X_test_vec = vectorizer.transform(X_test_raw)

    # Linear SVM Classifier
    svm_linear = SVC(kernel='linear', C=1.0, random_state=42)
    svm_linear.fit(X_train_vec, y_train)

    y_pred = svm_linear.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n1. Dataset Size: {len(df)} reviews ({len(X_train_raw)} train / {len(X_test_raw)} test)")
    print(f"2. Vectorization: CountVectorizer (Vocabulary size: {len(vectorizer.get_feature_names_out())})")
    print(f"3. Linear SVM Kernel Accuracy: {acc * 100:.2f}%\n")

    print("Test Set Prediction Breakdown:")
    print("-" * 75)
    print(f"{'Review Snippet':<45} | {'Actual':<10} | {'Predicted':<10}")
    print("-" * 75)
    for text, actual, pred in zip(X_test_raw, y_test, y_pred):
        snippet = text[:43] + ("..." if len(text) > 43 else "")
        print(f"{snippet:<45} | {actual:<10} | {pred:<10}")
    print("-" * 75)
    print("=" * 75)

if __name__ == "__main__":
    main()
