"""
Session 10 - Task 2: Logistic Regression Model on 10 Instagram Posts
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def main():
    print("=" * 70)
    print("SESSION 10 - TASK 2: Logistic Regression on Instagram Virality Data")
    print("=" * 70)

    # Dataset of 10 Instagram posts
    data = {
        'likes': [150, 1200, 300, 4500, 80, 2500, 6000, 400, 3200, 9500],
        'has_caption': [0, 1, 0, 1, 0, 1, 1, 0, 1, 1],
        'viral': [0, 0, 0, 1, 0, 1, 1, 0, 1, 1]
    }
    df = pd.DataFrame(data)

    print("\nInstagram Dataset (10 Posts):")
    print("-" * 45)
    print(f"{'Post #':<8} | {'Likes':<10} | {'Has Caption':<12} | {'Viral (y)':<10}")
    print("-" * 45)
    for idx, row in df.iterrows():
        print(f"Post {idx+1:<3} | {row['likes']:<10} | {row['has_caption']:<12} | {row['viral']:<10}")
    print("-" * 45)

    X = df[['likes', 'has_caption']]
    y = df['viral']

    # 1. Fit Raw Logistic Regression Model
    model_raw = LogisticRegression(random_state=42)
    model_raw.fit(X, y)

    # 2. Fit Scaled Logistic Regression Model (StandardScaler)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model_scaled = LogisticRegression(random_state=42)
    model_scaled.fit(X_scaled, y)

    print("\nMODEL COEFFICIENTS & INTERCEPT (UNSCALED VS SCALED):")
    print("-" * 65)
    print(f"{'Parameter / Feature':<25} | {'Unscaled Model':<18} | {'Standardized Model':<18}")
    print("-" * 65)
    print(f"{'Intercept (beta_0)':<25} | {model_raw.intercept_[0]:<18.6f} | {model_scaled.intercept_[0]:<18.6f}")
    print(f"{'likes (beta_1)':<25} | {model_raw.coef_[0][0]:<18.6f} | {model_scaled.coef_[0][0]:<18.6f}")
    print(f"{'has_caption (beta_2)':<25} | {model_raw.coef_[0][1]:<18.6f} | {model_scaled.coef_[0][1]:<18.6f}")
    print("-" * 65)

    print("\nLOGISTIC REGRESSION DECISION EQUATION (Standardized):")
    print(f"z = {model_scaled.intercept_[0]:.4f} + ({model_scaled.coef_[0][0]:.4f} * likes_std) + ({model_scaled.coef_[0][1]:.4f} * caption_std)")
    print("P(Viral = 1) = 1 / (1 + e^(-z))")
    print("=" * 70)

if __name__ == "__main__":
    main()
