"""
Session 4 - Task 3
Correlation analysis on Spotify songs dataset to select top 2 features related to 'popularity'.
"""

import os
import pandas as pd

def main():
    print("=" * 70)
    print("SESSION 4 - TASK 3: Spotify Feature Selection via Correlation Analysis")
    print("=" * 70)

    csv_path = os.path.join(os.path.dirname(__file__), "..", "dataset", "spotify_songs.csv")
    df = pd.read_csv(csv_path)

    numeric_df = df[['danceability', 'energy', 'tempo', 'popularity']]

    # Calculate Pearson Correlation Matrix
    corr_matrix = numeric_df.corr()

    print("\nPearson Correlation Matrix:")
    print("-" * 50)
    print(corr_matrix)
    print("-" * 50)

    # Extract correlations with target 'popularity', sorted by absolute correlation
    pop_corr = corr_matrix['popularity'].drop('popularity').abs().sort_values(ascending=False)
    raw_pop_corr = corr_matrix['popularity'].drop('popularity')

    print("\nCorrelation with 'popularity' (Ranked by Absolute Magnitude):")
    for feat, abs_val in pop_corr.items():
        print(f"  {feat:<15} : {raw_pop_corr[feat]:+.4f} (Abs: {abs_val:.4f})")

    top_2_features = pop_corr.head(2).index.tolist()

    print("\n" + "=" * 70)
    print(f"TOP 2 SELECTED FEATURES FOR PREDICTING POPULARITY: {top_2_features}")
    print("=" * 70)
    print(f"1. {top_2_features[0].capitalize()}: Shows the strongest linear correlation ({raw_pop_corr[top_2_features[0]]:+.4f}) with song popularity.")
    print(f"2. {top_2_features[1].capitalize()}: Exhibits the second highest correlation strength ({raw_pop_corr[top_2_features[1]]:+.4f}) with popularity scores.")
    print("=" * 70)

if __name__ == "__main__":
    main()
