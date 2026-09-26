"""
Session 2 - Task 2
Split Spotify dataset into 80% train and 20% test using train_test_split. Print row counts.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    print("=" * 70)
    print("SESSION 2 - TASK 2: Train/Test Split (80% Train, 20% Test)")
    print("=" * 70)

    csv_path = os.path.join(os.path.dirname(__file__), "..", "dataset", "spotify_top_50.csv")
    df = pd.read_csv(csv_path)

    label_col = 'popularity'
    drop_cols = ['track_id', 'track_name', 'artist_name', label_col]
    X = df.drop(columns=drop_cols)
    y = df[label_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    print(f"\nTotal Dataset Row Count : {len(df)} rows")
    print(f"Training Set Row Count  : {len(X_train)} rows ({(len(X_train)/len(df))*100:.1f}%)")
    print(f"Test Set Row Count      : {len(X_test)} rows ({(len(X_test)/len(df))*100:.1f}%)")
    print("=" * 70)
    print("Shape Verification:")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape : {X_test.shape}, y_test shape : {y_test.shape}")
    print("=" * 70)

if __name__ == "__main__":
    main()
