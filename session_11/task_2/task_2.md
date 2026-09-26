# Session 11 - Task 2: Spotify Playlist Classifier (Euclidean vs Manhattan KNN)

## Task Overview
Classify Spotify songs as either `'workout'` or `'chill'` playlist tracks based on audio features (`tempo`, `danceability`, `energy`) using K-Nearest Neighbors (KNN). Compare classification performance using **Euclidean distance ($L_2$)** vs **Manhattan distance ($L_1$)**.

---

## Distance Metrics Formulation

1. **Euclidean Distance ($L_2$ Norm)**:
   $$d_{\text{Euclidean}}(\mathbf{x}, \mathbf{y}) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}$$
   - Scikit-Learn parameter: `metric='euclidean'`

2. **Manhattan Distance ($L_1$ Norm)**:
   $$d_{\text{Manhattan}}(\mathbf{x}, \mathbf{y}) = \sum_{i=1}^{n} |x_i - y_i|$$
   - Scikit-Learn parameter: `metric='manhattan'`

---

## Experimental Setup & Performance

- **Dataset Size**: 100 Spotify songs (70 training, 30 testing)
- **Features**: `tempo` (BPM), `danceability` (0-1 scale), `energy` (0-1 scale)
- **Preprocessing**: `StandardScaler()` applied to eliminate scale disparities (e.g. tempo 150 vs energy 0.8)

| Distance Metric | Scikit-Learn Parameter | Test Accuracy Score |
|---|---|---|
| **Euclidean Distance ($L_2$)** | `metric='euclidean'` | **100.00%** |
| **Manhattan Distance ($L_1$)** | `metric='manhattan'` | **100.00%** |

---

## Insights
- Feature standardization ensures that `tempo` (range 60–175) does not dominate `danceability` or `energy` (range 0–1).
- When clusters are well-separated in feature space, both Euclidean and Manhattan distance metrics achieve robust, high-accuracy classifications.
