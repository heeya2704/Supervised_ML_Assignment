# Session 2 - Task 1: Spotify Top 50 Dataset Feature & Label Identification

## Task Overview
Download the 'Spotify Top 50 Songs' dataset from Kaggle, load it into a pandas DataFrame, and identify which columns should be used as features and which as the label if you want to predict a song's popularity score.

---

## Dataset Summary & Column Classification

### 1. Target Label Column
* **`popularity`**: Continuous integer score (0–100) representing how popular a track is based on total play counts and recency. This is the **dependent target variable ($y$)**.

### 2. Feature Columns (Predictors / $X$)
The numerical audio attributes that describe song characteristics serve as input features:
* **`danceability`**: Describes how suitable a track is for dancing (0.0 to 1.0).
* **`energy`**: Perceptual measure of intensity and activity (0.0 to 1.0).
* **`key`**: Estimated overall key of the track (integer mapping).
* **`loudness`**: Overall loudness of a track in decibels (dB).
* **`mode`**: Modality (major = 1, minor = 0).
* **`speechiness`**: Detects presence of spoken words (0.0 to 1.0).
* **`acousticness`**: Confidence measure of acoustic sound (0.0 to 1.0).
* **`instrumentalness`**: Predicts whether a track contains no vocals (0.0 to 1.0).
* **`liveness`**: Detects presence of an audience in the recording (0.0 to 1.0).
* **`valence`**: Musical positiveness conveyed by a track (0.0 to 1.0).
* **`tempo`**: Overall estimated tempo in beats per minute (BPM).
* **`duration_ms`**: Duration of the track in milliseconds.

### 3. Excluded Identifier Columns
* **`track_id`**, **`track_name`**, **`artist_name`**: Dropped prior to modeling because unique string metadata does not generalize across new songs and would cause spurious correlation / leakage.
