# Session 13 - Task 5: AI-Generated Spotify Hit Song Predictor

## Task Overview
Train an AI-assisted `RandomForestClassifier` on Spotify audio features (`tempo`, `danceability`, `energy`, `speechiness`) from [`spotify_hits.csv`](file:///c:/Users/heeya/OneDrive/Documents/TOPS/Machine%20Learning/supervised_ML_Assignment/session_13/dataset/spotify_hits.csv) to predict whether a song becomes a hit. Analyze **Out-of-Bag (OOB)** score evaluation.

---

## Model Evaluation & Performance
- **Dataset Size**: 120 Spotify songs (90 training, 30 testing)
- **Features**: `danceability` (36.03%), `energy` (30.17%), `tempo` (17.10%), `speechiness` (16.70%)
- **Test Accuracy**: **80.00%**
- **Out-of-Bag (OOB) Score**: **75.56%**

---

## AI-Generated Code Review & Fixes Applied

### 1. Code Snippet & Error:
- **Original AI Prompt Error**: Initial AI-generated snippet set `oob_score=True` but left `bootstrap=False` (or did not set `bootstrap=True`), triggering `ValueError: Out-of-bag estimation only available if bootstrap=True`.

### 2. Fix Applied:
```python
# Fixed instantiation:
rf_model = RandomForestClassifier(
    n_estimators=100,
    bootstrap=True,  # Mandatory for OOB sampling
    oob_score=True,  # Enables Out-of-Bag evaluation
    random_state=42
)
```

### 3. Key Learning:
- **Out-of-Bag (OOB) Error Estimation**: Because bagging samples data with replacement ($N$ samples with probability $1 - 1/N$), approx. **36.8% of samples are left out** of each tree's training set. Evaluating predictions on these left-out ("out-of-bag") samples yields an unbiased validation accuracy score (**75.56%**) without requiring a separate validation split.
