import numpy as np
import pandas as pd
import os

def generate_zomato_dataset():
    np.random.seed(42)
    n_samples = 300

    # Meaningful features
    online_order = np.random.choice([0, 1], size=n_samples, p=[0.3, 0.7])
    book_table = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])
    votes = np.random.randint(20, 3000, size=n_samples)
    approx_cost_for_two = np.random.randint(200, 2500, size=n_samples)
    location_score = np.random.uniform(1.0, 10.0, size=n_samples)
    cuisine_count = np.random.randint(1, 8, size=n_samples)
    avg_dish_price = approx_cost_for_two / np.random.uniform(1.8, 2.5, size=n_samples)
    delivery_time_min = np.random.randint(20, 60, size=n_samples)
    restaurant_age_yrs = np.random.uniform(0.5, 15.0, size=n_samples)
    parking_available = np.random.choice([0, 1], size=n_samples, p=[0.5, 0.5])

    # Redundant / Noisy features (for Lasso feature selection demonstration)
    noise_feature_1 = np.random.normal(0, 1, size=n_samples)
    noise_feature_2 = np.random.uniform(10, 100, size=n_samples)
    noise_feature_3 = np.random.choice([0, 1], size=n_samples)

    # True Target generation: Rating (scale 1.0 to 5.0)
    # Signal depends strongly on location_score, votes, book_table, online_order, approx_cost_for_two, restaurant_age_yrs
    target_rating = (
        2.5 +
        0.15 * location_score +
        0.0003 * votes +
        0.35 * book_table +
        0.20 * online_order +
        0.0002 * approx_cost_for_two +
        0.04 * restaurant_age_yrs -
        0.005 * delivery_time_min +
        np.random.normal(0, 0.15, size=n_samples)
    )
    target_rating = np.clip(target_rating, 1.0, 5.0).round(2)

    df = pd.DataFrame({
        'online_order': online_order,
        'book_table': book_table,
        'votes': votes,
        'approx_cost_for_two': approx_cost_for_two,
        'location_score': location_score.round(2),
        'cuisine_count': cuisine_count,
        'avg_dish_price': avg_dish_price.round(2),
        'delivery_time_min': delivery_time_min,
        'restaurant_age_yrs': restaurant_age_yrs.round(1),
        'parking_available': parking_available,
        'noise_feature_1': noise_feature_1.round(3),
        'noise_feature_2': noise_feature_2.round(2),
        'noise_feature_3': noise_feature_3,
        'rating': target_rating
    })

    dataset_dir = os.path.join(os.path.dirname(__file__), 'dataset')
    os.makedirs(dataset_dir, exist_ok=True)
    file_path = os.path.join(dataset_dir, 'zomato_ratings.csv')
    df.to_csv(file_path, index=False)
    print(f"Dataset successfully created at: {file_path}")
    print(f"Shape: {df.shape}")

if __name__ == "__main__":
    generate_zomato_dataset()
