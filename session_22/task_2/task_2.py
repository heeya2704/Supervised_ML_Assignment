"""
Session 22 - Task 2: Missing Value Imputation & Outlier Handling via IQR Method
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("=" * 75)
    print("SESSION 22 - TASK 2: Missing Value Imputation & IQR Outlier Handling")
    print("=" * 75)

    # 1. Load dataset
    df = pd.read_csv("session_22/dataset/zomato_restaurants.csv")

    # 2. Missing Value Check & Imputation
    print("\n1. MISSING VALUES ANALYSIS:")
    null_summary = df[["cost", "rating"]].isnull().sum()
    print(f"   • Missing 'cost' values   : {null_summary['cost']}")
    print(f"   • Missing 'rating' values : {null_summary['rating']}")

    cost_median = df["cost"].median()
    rating_median = df["rating"].median()

    df["cost_imputed"] = df["cost"].fillna(cost_median)
    df["rating_imputed"] = df["rating"].fillna(rating_median)

    print(f"\n2. MEDIAN IMPUTATION:")
    print(f"   • Imputed 'cost' NaNs with median: INR {cost_median:.2f}")
    print(f"   • Imputed 'rating' NaNs with median: {rating_median:.2f}")

    # 3. IQR Outlier Detection for 'cost'
    Q1 = df["cost_imputed"].quantile(0.25)
    Q3 = df["cost_imputed"].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df["cost_imputed"] < lower_bound) | (df["cost_imputed"] > upper_bound)]
    print(f"\n3. IQR OUTLIER DETECTION FOR 'cost':")
    print(f"   • Q1 (25th percentile) : INR {Q1:.2f}")
    print(f"   • Q3 (75th percentile) : INR {Q3:.2f}")
    print(f"   • IQR (Q3 - Q1)        : INR {IQR:.2f}")
    print(f"   • Upper Bound          : INR {upper_bound:.2f}")
    print(f"   • Outliers Detected    : {len(outliers)} restaurants (Max cost: INR {df['cost_imputed'].max():.2f})")

    # Handle outliers by capping at upper bound
    df["cost_capped"] = np.clip(df["cost_imputed"], lower_bound, upper_bound)
    print(f"\n4. OUTLIER HANDLING (CAPPING):")
    print(f"   • Capped cost values above upper bound at INR {upper_bound:.2f}")
    print(f"   • Max cost after capping: INR {df['cost_capped'].max():.2f}")

    # 4. Generate Boxplot Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(y=df["cost_imputed"], ax=axes[0], color="coral")
    axes[0].set_title("Before Outlier Handling (Raw / Imputed Cost)")
    axes[0].set_ylabel("Cost for Two (INR)")

    sns.boxplot(y=df["cost_capped"], ax=axes[1], color="teal")
    axes[1].set_title(f"After IQR Capping (Capped at Upper Bound INR {upper_bound:.1f})")
    axes[1].set_ylabel("Cost for Two (INR)")

    plt.suptitle("Session 22 Task 2: Zomato Cost Outlier Detection & Capping", fontsize=14)
    plt.tight_layout()
    
    save_path = "session_22/task_2/outliers_boxplot.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"\n5. Saved visualization plot to: '{save_path}'")
    print("=" * 75)

if __name__ == "__main__":
    main()
