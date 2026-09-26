"""
Session 13 - Task 2: Flipkart Product Category Classifier via Random Forest
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def main():
    print("=" * 75)
    print("SESSION 13 - TASK 2: Flipkart Product Category Classifier & Importance")
    print("=" * 75)

    # DataFrame with 20 Flipkart products
    data = {
        'product_name': [
            "Laptop", "Smartphone", "Headphones", "Smartwatch", "Tablet", "LED TV", "Gaming Console",
            "T-Shirt", "Jeans", "Jacket", "Sneakers", "Dress", "Watch",
            "Sofa set", "Dining Table", "Bedsheet", "Microwave", "Curtains", "Cookware Set", "Wall Clock"
        ],
        'price_rs': [
            55000, 25000, 3500, 6000, 18000, 42000, 38000,
            800, 1500, 2500, 3000, 1800, 4500,
            22000, 15000, 700, 8500, 1200, 3200, 900
        ],
        'rating': [
            4.5, 4.3, 4.1, 4.0, 4.2, 4.6, 4.7,
            3.9, 4.0, 4.2, 4.4, 4.1, 4.3,
            4.4, 4.5, 3.8, 4.3, 3.9, 4.1, 3.7
        ],
        'brand_tier': [
            3, 3, 2, 2, 3, 3, 3,  # 3: Premium, 2: Mid, 1: Basic
            1, 2, 2, 2, 1, 3,
            3, 3, 1, 2, 1, 2, 1
        ],
        'category': [
            "Electronics", "Electronics", "Electronics", "Electronics", "Electronics", "Electronics", "Electronics",
            "Fashion", "Fashion", "Fashion", "Fashion", "Fashion", "Fashion",
            "Home", "Home", "Home", "Home", "Home", "Home", "Home"
        ]
    }

    df = pd.DataFrame(data)

    print("\nFlipkart 20-Product Sample Dataset:")
    print("-" * 65)
    print(f"{'Product Name':<18} | {'Price (Rs)':<12} | {'Rating':<8} | {'Tier':<6} | {'Category':<12}")
    print("-" * 65)
    for _, row in df.iterrows():
        print(f"{row['product_name']:<18} | Rs. {row['price_rs']:<8} | {row['rating']:<8} | {row['brand_tier']:<6} | {row['category']:<12}")
    print("-" * 65)

    X = df[['price_rs', 'rating', 'brand_tier']]
    y = df['category']

    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(X, y)

    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

    print("\nRANDOM FOREST FEATURE IMPORTANCE VALUES:")
    print("-" * 55)
    print(f"{'Feature Name':<20} | {'Importance Score':<18} | {'Contribution':<12}")
    print("-" * 55)
    for feat, score in importances.items():
        print(f"{feat:<20} | {score:<18.6f} | {score * 100:<10.2f}%")
    print("-" * 55)
    print("=" * 75)

if __name__ == "__main__":
    main()
