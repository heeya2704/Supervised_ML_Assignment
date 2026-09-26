"""
Session 21 - Task 2: DateTime Feature Extraction on Flipkart Sales Dataset
"""

import pandas as pd

def main():
    print("=" * 75)
    print("SESSION 21 - TASK 2: DateTime Feature Extraction (Day, Month, Weekend)")
    print("=" * 75)

    # 1. Load Flipkart Sales Dataset
    df = pd.read_csv("session_21/dataset/flipkart_sales.csv")
    df["order_date"] = pd.to_datetime(df["order_date"])

    # 2. Extract Calendar & Weekend Features
    df["day_of_week"] = df["order_date"].dt.day_name()
    df["month"] = df["order_date"].dt.month_name()
    df["is_weekend"] = df["order_date"].dt.dayofweek.isin([5, 6]).astype(int)

    print(f"\n1. DATASET OVERVIEW (Flipkart Sales Orders):")
    print(f"   • Total Orders Processed : {len(df)}")
    print(f"   • Date Range             : {df['order_date'].min().strftime('%Y-%m-%d')} to {df['order_date'].max().strftime('%Y-%m-%d')}")
    print(f"   • Total Weekend Orders   : {df['is_weekend'].sum()} ({df['is_weekend'].mean()*100:.1f}%)")

    print(f"\n2. SAMPLE EXTRACTED DATETIME FEATURES (First 10 Rows):")
    print("-" * 75)
    cols = ["order_id", "order_date", "day_of_week", "month", "is_weekend", "sales_amount"]
    print(df[cols].head(10).to_string(index=False))
    print("-" * 75)
    print("=" * 75)

if __name__ == "__main__":
    main()
