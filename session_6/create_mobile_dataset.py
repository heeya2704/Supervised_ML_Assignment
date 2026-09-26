"""
Session 6 Dataset Generator: Mobile Phone Prices Dataset
Generates mobile_prices.csv with high collinearity between features (RAM & Storage, Battery & Screen Size).
"""

import os
import pandas as pd
import numpy as np

def generate_mobile_dataset(output_path):
    np.random.seed(42)
    n_samples = 100

    ram = np.random.choice([4, 6, 8, 12, 16], size=n_samples, p=[0.2, 0.3, 0.3, 0.15, 0.05])
    # Storage is almost a direct scalar multiple of RAM (high correlation / multicollinearity)
    storage = ram * 32 + np.random.normal(0, 2, size=n_samples)
    storage = np.round(storage)

    battery = np.random.choice([4000, 4500, 5000, 6000], size=n_samples)
    camera = np.random.choice([48, 50, 64, 108, 200], size=n_samples)
    
    # Screen size highly correlated with battery capacity
    screen_size = 5.5 + (battery / 4000.0) * 0.8 + np.random.normal(0, 0.02, size=n_samples)
    screen_size = np.round(screen_size, 2)

    # Base price calculation + noise
    price = (
        8000 + 
        ram * 3500 + 
        storage * 120 + 
        battery * 4 + 
        camera * 180 + 
        screen_size * 2500 + 
        np.random.normal(0, 2500, size=n_samples)
    )
    price = np.round(price, -2) # Round to nearest 100

    df = pd.DataFrame({
        'ram_gb': ram,
        'storage_gb': storage,
        'battery_mah': battery,
        'camera_mp': camera,
        'screen_size_inch': screen_size,
        'price_inr': price
    })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[+] Mobile phone dataset saved to: {output_path} ({len(df)} records)")

if __name__ == "__main__":
    dataset_file = os.path.join(os.path.dirname(__file__), "dataset", "mobile_prices.csv")
    generate_mobile_dataset(dataset_file)
