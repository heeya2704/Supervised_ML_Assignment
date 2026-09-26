"""
Session 2 - Task 1
Load 'Spotify Top 50 Songs' dataset, identify features and label for predicting popularity score.
"""

import os
import pandas as pd

def main():
    print("=" * 70)
    print("SESSION 2 - TASK 1: Spotify Dataset Loading & Feature/Label Identification")
    print("=" * 70)
    
    csv_path = os.path.join(os.path.dirname(__file__), "..", "dataset", "spotify_top_50.csv")
    df = pd.read_csv(csv_path)
    
    print(f"\nSuccessfully loaded dataset: {df.shape[0]} rows x {df.shape[1]} columns.\n")
    print("Dataset Columns & Data Types:")
    print(df.dtypes)
    
    # Label identification
    label_column = 'popularity'
    
    # Non-predictive identifier columns
    identifiers = ['track_id', 'track_name', 'artist_name']
    
    # Feature columns
    feature_columns = [col for col in df.columns if col not in identifiers + [label_column]]
    
    print("\n" + "=" * 70)
    print(f"TARGET LABEL COLUMN: '{label_column}' (Continuous variable representing popularity score)")
    print("=" * 70)
    print(f"IDENTIFIER COLUMNS (Excluded from modeling): {identifiers}")
    print("\nINPUT FEATURE COLUMNS:")
    for idx, feature in enumerate(feature_columns, 1):
        print(f"  {idx}. {feature} ({df[feature].dtype})")
    print("=" * 70)

if __name__ == "__main__":
    main()
