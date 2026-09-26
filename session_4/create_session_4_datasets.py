"""
Create realistic datasets for all Session 4 tasks.
"""

import os
import pandas as pd
import numpy as np

def generate_datasets():
    os.makedirs("session_4/dataset", exist_ok=True)
    np.random.seed(42)
    
    # 1. IPL Stats Dataset
    ipl_df = pd.DataFrame({
        'player': [f"Player_{i}" for i in range(1, 21)],
        'runs': np.random.randint(200, 6500, 20),
        'wickets': np.random.randint(0, 170, 20),
        'matches': np.random.randint(15, 230, 20),
        'strike_rate': np.round(np.random.uniform(95.0, 165.0, 20), 2)
    })
    ipl_df.to_csv("session_4/dataset/ipl_stats.csv", index=False)
    
    # 2. Flipkart Products Dataset
    flipkart_df = pd.DataFrame({
        'product_id': [f"FK_{100+i}" for i in range(20)],
        'price': np.random.randint(299, 89999, 20),
        'rating': np.round(np.random.uniform(2.1, 4.9, 20), 1),
        'number_of_reviews': np.random.randint(10, 15000, 20),
        'discount': np.random.randint(5, 75, 20)
    })
    flipkart_df.to_csv("session_4/dataset/flipkart_products.csv", index=False)
    
    # 3. Spotify Songs Dataset
    dance = np.round(np.random.uniform(0.3, 0.9, 30), 2)
    energy = np.round(np.random.uniform(0.4, 0.95, 30), 2)
    tempo = np.round(np.random.uniform(80.0, 180.0, 30), 1)
    pop = np.round(40 * dance + 25 * energy + 0.1 * tempo + np.random.normal(0, 3, 30)).astype(int)
    pop = np.clip(pop, 10, 99)
    spotify_df = pd.DataFrame({
        'song_id': [f"S_{i}" for i in range(1, 31)],
        'danceability': dance,
        'energy': energy,
        'tempo': tempo,
        'popularity': pop
    })
    spotify_df.to_csv("session_4/dataset/spotify_songs.csv", index=False)
    
    # 4. Zomato Restaurants Dataset
    rating = np.round(np.random.uniform(2.5, 4.9, 30), 1)
    votes = np.random.randint(50, 5000, 30)
    table_booking = np.random.choice([0, 1], 30)
    online_order = np.random.choice([0, 1], 30)
    cuisine_variety = np.random.randint(1, 10, 30)
    location_score = np.round(np.random.uniform(1.0, 10.0, 30), 1)
    cost = np.round(200 + 300 * rating + 0.15 * votes + 400 * table_booking + 100 * cuisine_variety + np.random.normal(0, 50, 30)).astype(int)
    
    zomato_df = pd.DataFrame({
        'rating': rating,
        'votes': votes,
        'table_booking': table_booking,
        'online_order': online_order,
        'cuisine_variety': cuisine_variety,
        'location_score': location_score,
        'average_cost_for_two': cost
    })
    zomato_df.to_csv("session_4/dataset/zomato_restaurants.csv", index=False)
    
    # 5. BookMyShow Ratings Dataset with Outliers
    normal_ratings = np.round(np.random.normal(7.5, 1.0, 25), 1)
    normal_ratings = np.clip(normal_ratings, 5.0, 9.8)
    outliers = [14.5, -2.5, 18.0, 0.1, 16.0]  # Intentionally corrupted outlier ratings
    all_ratings = np.concatenate([normal_ratings, outliers])
    np.random.shuffle(all_ratings)
    
    bms_df = pd.DataFrame({
        'movie_id': [f"MOV_{100+i}" for i in range(len(all_ratings))],
        'movie_name': [f"Movie {i+1}" for i in range(len(all_ratings))],
        'user_ratings': all_ratings
    })
    bms_df.to_csv("session_4/dataset/bookmyshow_ratings.csv", index=False)
    
    print("[+] All 5 datasets for Session 4 created successfully!")

if __name__ == "__main__":
    generate_datasets()
