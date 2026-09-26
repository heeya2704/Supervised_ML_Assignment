"""
Session 11 - Task 4: WhatsApp Message Category Classifier (MultinomialNB & Confusion Matrix)
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def main():
    print("=" * 75)
    print("SESSION 11 - TASK 4: WhatsApp Chat Category Classifier (MultinomialNB)")
    print("=" * 75)

    # WhatsApp Chat Messages Dataset
    messages = [
        # Personal
        "Hey bro, are we meeting for coffee today at 5?",
        "Call me when you reach home Mom is asking",
        "Happy birthday! Hope you have an amazing day ahead",
        "Can you send me the lecture notes from yesterday's class?",
        "Let's catch up this weekend over dinner",
        "Thanks for helping me with the homework bro",
        
        # Group
        "Guys submit the assignment link before 11:59 PM today",
        "Project team meeting scheduled on Google Meet tomorrow at 10 AM",
        "Please share your attendance confirmation for the college trip",
        "Who is managing the Google Drive folder for the group presentation?",
        "Everyone complete the survey form pinned in the group description",
        "Reminder team code review meeting starts in 15 minutes",

        # Spam
        "CONGRATULATIONS! You won a cash prize of Rs 50,000! Click link to claim now",
        "URGENT: Your account will be blocked. Update your KYC immediately here",
        "Get 90% discount on iPhone 15 Pro Max! Limited time offer click link",
        "Earn Rs 5000 per day working from home part time job apply now",
        "FREE instant loan approved up to Rs 2 Lakhs without document verification",
        "Win guaranteed rewards and gift vouchers! Click to claim free prize"
    ]
    labels = [
        "personal", "personal", "personal", "personal", "personal", "personal",
        "group", "group", "group", "group", "group", "group",
        "spam", "spam", "spam", "spam", "spam", "spam"
    ]

    df = pd.DataFrame({'message': messages, 'label': labels})

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df['message'], df['label'], test_size=0.33, random_state=42, stratify=df['label']
    )

    vectorizer = CountVectorizer(stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train_raw)
    X_test_vec = vectorizer.transform(X_test_raw)

    # Multinomial Naive Bayes Classifier
    mnb = MultinomialNB()
    mnb.fit(X_train_vec, y_train)

    y_pred = mnb.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    classes = list(mnb.classes_)
    cm = confusion_matrix(y_test, y_pred, labels=classes)

    print(f"\n1. Total WhatsApp Messages: {len(df)}")
    print(f"2. Classes: {classes}")
    print(f"3. Model Accuracy: {acc * 100:.2f}%\n")

    header_title = "Actual \\ Predicted"
    print("CONFUSION MATRIX:")
    print("-" * 55)
    cols_str = " | ".join([f"{c:<10}" for c in classes])
    print(f"{header_title:<18} | {cols_str}")
    print("-" * 55)
    for idx, cls in enumerate(classes):
        row_str = " | ".join([f"{count:<10}" for count in cm[idx]])
        print(f"{cls:<18} | {row_str}")
    print("-" * 55)

    print("\nCLASSIFICATION REPORT:")
    print(classification_report(y_test, y_pred, target_names=classes))
    print("=" * 75)

if __name__ == "__main__":
    main()
