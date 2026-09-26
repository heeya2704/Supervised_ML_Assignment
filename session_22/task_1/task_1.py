"""
Session 22 - Task 1: Zomato Dataset Loading & Exploratory Data Analysis (EDA)
"""

import pandas as pd

def main():
    print("=" * 75)
    print("SESSION 22 - TASK 1: Zomato Restaurant Dataset EDA & Inspection")
    print("=" * 75)

    # 1. Load dataset into pandas DataFrame
    dataset_path = "session_22/dataset/zomato_restaurants.csv"
    df = pd.read_csv(dataset_path)

    print(f"\n1. DATASET OVERVIEW:")
    print(f"   • Path: '{dataset_path}'")
    print(f"   • Shape: {df.shape[0]} rows x {df.shape[1]} columns")

    print("\n2. FIRST 5 ROWS:")
    print("-" * 75)
    print(df.head())
    print("-" * 75)

    print("\n3. DATASET INFORMATION (info()):")
    print("-" * 75)
    df.info()
    print("-" * 75)

    print("\n4. STATISTICAL SUMMARY (describe()):")
    print("-" * 75)
    print(df.describe())
    print("-" * 75)
    print("=" * 75)

if __name__ == "__main__":
    main()
