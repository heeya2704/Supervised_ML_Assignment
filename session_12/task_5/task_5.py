"""
Session 12 - Task 5: Swiggy Delivery ETA Decision Tree & Split Interpretation
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def main():
    print("=" * 75)
    print("SESSION 12 - TASK 5: Swiggy Delivery ETA Decision Tree & Split Analysis")
    print("=" * 75)

    np.random.seed(42)
    n_samples = 150

    # Swiggy Food Delivery Features
    distance_km = np.random.uniform(1.0, 15.0, size=n_samples)
    prep_time_min = np.random.randint(10, 45, size=n_samples)
    rain_intensity = np.random.choice([0, 1, 2], size=n_samples, p=[0.6, 0.25, 0.15]) # 0: None, 1: Moderate, 2: Heavy
    traffic_density = np.random.uniform(1.0, 10.0, size=n_samples)

    # Delivery ETA Target: 'Fast' (<=30 min), 'Moderate' (31-50 min), 'Delayed' (>50 min)
    est_time = 1.8 * distance_km + 1.1 * prep_time_min + 7.0 * rain_intensity + 2.0 * traffic_density
    eta_category = []
    for t in est_time:
        if t <= 35:
            eta_category.append("Fast")
        elif t <= 55:
            eta_category.append("Moderate")
        else:
            eta_category.append("Delayed")

    df = pd.DataFrame({
        'distance_km': distance_km.round(1),
        'prep_time_min': prep_time_min,
        'rain_intensity': rain_intensity,
        'traffic_density': traffic_density.round(1),
        'delivery_speed': eta_category
    })

    X = df.drop(columns=['delivery_speed'])
    y = df['delivery_speed']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X_train, y_train)

    acc = accuracy_score(y_test, clf.predict(X_test))

    print(f"\n1. Dataset Size: {len(df)} deliveries (112 train / 38 test)")
    print(f"2. Classes: {list(clf.classes_)}")
    print(f"3. Model Accuracy: {acc * 100:.2f}%\n")

    # Plot Decision Tree
    plt.figure(figsize=(14, 8), dpi=300)
    plot_tree(
        clf,
        feature_names=X.columns,
        class_names=clf.classes_,
        filled=True,
        rounded=True,
        fontsize=10
    )
    plt.title("Swiggy Food Delivery Speed Classifier (Max Depth = 3)", fontsize=14, fontweight='bold', pad=15)

    output_dir = os.path.dirname(__file__)
    file_path = os.path.join(output_dir, 'swiggy_tree.png')
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()

    print(f"Swiggy decision tree plot saved to: {file_path}")
    print("=" * 75)

if __name__ == "__main__":
    main()
