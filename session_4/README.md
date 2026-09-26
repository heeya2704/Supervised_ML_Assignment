# Session 4: Feature Scaling, Selection & Outlier Detection

This folder contains dataset scripts, code files, scaled dataset outputs, and markdown reports for all tasks in **Session 4**.

## Directory Structure
- [`dataset/`](dataset/): Source CSV files (`ipl_stats.csv`, `flipkart_products.csv`, `spotify_songs.csv`, `zomato_restaurants.csv`, `bookmyshow_ratings.csv`).
- [`task_1/`](task_1/): `StandardScaler` on IPL player stats ($\mu=0, \sigma=1$) & [`ipl_scaled.csv`](task_1/ipl_scaled.csv).
- [`task_2/`](task_2/): `MinMaxScaler` on Flipkart product ratings ([0, 1] range scaling with Min/Max comparison).
- [`task_3/`](task_3/): Correlation analysis on Spotify songs dataset to select top 2 features (`danceability`, `tempo`).
- [`task_4/`](task_4/): Recursive Feature Elimination (RFE) with `DecisionTreeRegressor` selecting top 3 Zomato features.
- [`task_5/`](task_5/): Outlier detection and removal on BookMyShow user ratings using the IQR method.

## How to Run Python Scripts
```bash
python session_4/task_1/task_1.py
python session_4/task_2/task_2.py
python session_4/task_3/task_3.py
python session_4/task_4/task_4.py
python session_4/task_5/task_5.py
```
