# Session 18 - Task 4: Cross-Validation Stability Benchmark ($k = 3, 5, 10$)

## Task Overview
Benchmark accuracy score variance (standard deviation) across different fold counts ($cv = 3, 5, 10$) using `RandomForestClassifier` on the Wine recognition dataset.

---

## Experimental Benchmark Results

- **Model**: `RandomForestClassifier(n_estimators=100, random_state=42)`
- **Dataset**: Wine Dataset ($N=178$ samples)

| $k$-Fold Setting ($cv$) | Mean Accuracy | Standard Deviation ($\sigma$) | Evaluation Stability |
| :---: | :---: | :---: | :---: |
| **$cv = 3$** | **97.74% (0.9774)** | **$\pm 0.0160$** | **Highest Stability** |
| **$cv = 5$** | **97.76% (0.9776)** | **$\pm 0.0274$** | Moderate Stability |
| **$cv = 10$** | **98.33% (0.9833)** | **$\pm 0.0272$** | Moderate Stability |

---

## Findings & Analysis
- **Most Stable Setting**: **$cv = 3$** yields the lowest score variance ($\sigma = \pm 0.0160$).
- **Variance vs Bias Trade-off**: Increasing $k$ to 10 slightly increases evaluation variance due to smaller validation fold sizes (~17 samples/fold), while $cv=3$ and $cv=5$ balance sample volume per fold and variance stability.
