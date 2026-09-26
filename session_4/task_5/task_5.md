# Session 4 - Task 5: IQR Outlier Detection (BookMyShow User Ratings)

## Task Overview
Given a BookMyShow movie ratings dataset, detect outliers in the `'user_ratings'` column using the IQR method. Remove the outliers and display how many rows were dropped.  
*(Hint: Calculate Q1, Q3, and IQR; filter out rows outside $[Q1 - 1.5 \times IQR, Q3 + 1.5 \times IQR]$).*

---

## Mathematical Formula & Threshold Calculation

1. **First Quartile ($Q1$):** `6.8500` (25th percentile)
2. **Third Quartile ($Q3$):** `7.9750` (75th percentile)
3. **Interquartile Range ($IQR$):** $Q3 - Q1 = 7.9750 - 6.8500 =$ **`1.1250`**

### Acceptable Bounds:
* **Lower Bound:** $Q1 - 1.5 \times IQR = 6.8500 - 1.5(1.1250) =$ **`5.1625`**
* **Upper Bound:** $Q3 + 1.5 \times IQR = 7.9750 + 1.5(1.1250) =$ **`9.6625`**

---

## Identified & Dropped Outlier Rows

| Index | Movie ID | Movie Title | Corrupted Rating | Outlier Reason |
|---|---|---|---|---|
| 0 | MOV_100 | Movie 1 | **`-2.5`** | Below Lower Bound ($< 5.1625$) |
| 7 | MOV_107 | Movie 8 | **`18.0`** | Above Upper Bound ($> 9.6625$) |
| 21 | MOV_121 | Movie 22 | **`0.1`** | Below Lower Bound ($< 5.1625$) |
| 26 | MOV_126 | Movie 27 | **`14.5`** | Above Upper Bound ($> 9.6625$) |
| 27 | MOV_127 | Movie 28 | **`16.0`** | Above Upper Bound ($> 9.6625$) |

---

## Final Outlier Removal Summary

* **Initial Dataset Rows:** 30 rows
* **Total Outlier Rows Dropped:** **`5` rows**
* **Clean Dataset Rows Remaining:** **`25` rows**
