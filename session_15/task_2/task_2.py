"""
Session 15 - Task 2: AdaBoost Classifier on Flipkart Review Sentiment Data
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def main():
    print("=" * 80)
    print("SESSION 15 - TASK 2: AdaBoost Sentiment Classifier on Flipkart Reviews")
    print("=" * 80)

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

    # Train AdaBoost Classifier (using SAMME algorithm)
    ada = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=50,
        learning_rate=1.0,
        algorithm='SAMME',
        random_state=42
    )
    ada.fit(X_train_vec, y_train)

    y_pred = ada.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n1. Total Reviews: {len(df)} ({len(X_train_raw)} train / {len(X_test_raw)} test)")
    print(f"2. AdaBoost Base Estimator: DecisionTreeClassifier(max_depth=1)")
    print(f"3. Model Accuracy: {acc * 100:.2f}%\n")

    print("DETAILED CLASSIFICATION REPORT (PRECISION, RECALL, F1-SCORE):")
    print("-" * 65)
    print(classification_report(y_test, y_pred))
    print("=" * 80)

if __name__ == "__main__":
    main()
