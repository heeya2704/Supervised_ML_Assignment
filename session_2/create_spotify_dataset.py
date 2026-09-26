"""
Generate realistic Spotify Top 50 dataset for Session 2 tasks.
"""

import pandas as pd
import numpy as np

def generate_spotify_data():
    np.random.seed(42)
    n_samples = 50

    track_names = [f"Hit Song {i+1}" for i in range(n_samples)]
    artist_names = [f"Artist {np.random.randint(1, 15)}" for _ in range(n_samples)]
    
    danceability = np.round(np.random.uniform(0.40, 0.95, n_samples), 3)
    energy = np.round(np.random.uniform(0.35, 0.98, n_samples), 3)
    key = np.random.randint(0, 12, n_samples)
    loudness = np.round(np.random.uniform(-12.0, -2.0, n_samples), 2)
    mode = np.random.choice([0, 1], n_samples)
    speechiness = np.round(np.random.uniform(0.03, 0.35, n_samples), 3)
    acousticness = np.round(np.random.uniform(0.01, 0.85, n_samples), 3)
    instrumentalness = np.round(np.random.uniform(0.00, 0.30, n_samples), 4)
    liveness = np.round(np.random.uniform(0.05, 0.40, n_samples), 3)
    valence = np.round(np.random.uniform(0.20, 0.95, n_samples), 3)
    tempo = np.round(np.random.uniform(85.0, 175.0, n_samples), 1)
    duration_ms = np.random.randint(150000, 270000, n_samples)
    
    # Popularity formula with some random noise
    popularity = np.round(
        35 + 40 * danceability + 20 * energy - 15 * acousticness + np.random.normal(0, 5, n_samples)
    ).astype(int)
    popularity = np.clip(popularity, 10, 100)

    df = pd.DataFrame({
        'track_id': [f"spot_{1000+i}" for i in range(n_samples)],
        'track_name': track_names,
        'artist_name': artist_names,
        'danceability': danceability,
        'energy': energy,
        'key': key,
        'loudness': loudness,
        'mode': mode,
        'speechiness': speechiness,
        'acousticness': acousticness,
        'instrumentalness': instrumentalness,
        'liveness': liveness,
        'valence': valence,
        'tempo': tempo,
        'duration_ms': duration_ms,
        'popularity': popularity
    })

    return df

if __name__ == "__main__":
    df = generate_spotify_data()
    file_path = "session_2/dataset/spotify_top_50.csv"
    import os
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=False)
    print(f"Generated Spotify Top 50 dataset saved to {file_path}")
    print(df.head())
