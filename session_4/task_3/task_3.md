# Session 4 - Task 3: Correlation Analysis Feature Selection (Spotify Popularity)

## Task Overview
Given a Spotify songs dataset (features: `danceability`, `energy`, `tempo`, `popularity`), use correlation analysis to select the top 2 features most related to `'popularity'`. List the selected features and explain your choice in one line each.

---

## Correlation Matrix Output

$$r_{xy} = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2 \sum (y_i - \bar{y})^2}}$$

| Feature Name | Pearson Correlation Coefficient ($r$) with `popularity` | Correlation Ranking | Selection Status |
|---|---|---|---|
| **`danceability`** | **`+0.8095`** | **1st** | **SELECTED (Top 1)** |
| **`tempo`** | **`+0.4081`** | **2nd** | **SELECTED (Top 2)** |
| `energy` | `+0.3902` | 3rd | Excluded |

---

## Selected Top 2 Features & One-Line Explanations

1. **`danceability`**
   > *Selected because it demonstrates the strongest positive linear correlation ($r = +0.8095$) with song popularity scores, indicating higher danceability strongly aligns with higher listener popularity.*

2. **`tempo`**
   > *Selected because it exhibits the second highest correlation coefficient ($r = +0.4081$) with popularity, showing a consistent positive relationship with listener preferences.*
