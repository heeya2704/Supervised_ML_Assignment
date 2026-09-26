# Session 2 - Task 3: Insufficient Training Data Experiment (5% Train / 95% Test)

## Task Overview
Given a simple linear regression model predicting song popularity from danceability, intentionally use only **5% of the data for training** and **95% for testing**. Observe the model's performance and explain whether this is likely to cause underfitting or overfitting.  
*(Hint: Look at the accuracy or error on both train and test sets to support your answer).*

---

## Experimental Results

| Split Metric | Training Set (5%) | Testing Set (95%) |
|---|---|---|
| **Row Count** | **2 rows** | **48 rows** |
| **Mean Squared Error (MSE)** | **0.0000** | **98.7997** |
| **$R^2$ Score** | **1.0000 (100%)** | **0.1104 (11.0%)** |

---

## Performance Observation & Analysis

### 1. Training Set Behavior
* The model achieves a **perfect $R^2 = 1.0000$ and zero error ($\text{MSE} = 0.0000$)** on the training set because a 1D linear regression line fits any 2 points with 100% exactness.

### 2. Testing Set Behavior
* On the 95% test set (48 unseen songs), the test error explodes to **$\text{MSE} = 98.80$** and **$R^2$ drops sharply to $0.1104$**.

---

## Conclusion: Underfitting vs. Overfitting Diagnosis

This setup causes **Extreme Estimation Variance & Underfitting of the True Distribution**:

1. **Why it happens:** With only 2 samples, the model is deprived of sufficient data to learn the true population relationship between danceability and song popularity.
2. **False Fit on Train vs Generalization Breakdown:** Although the line superficially "fits" the 2 training points perfectly (appearing like micro-overfitting to 2 samples), the model fails to learn the general underlying pattern of the domain, leading to high general error (underfitting the real data manifold).
3. **Key Takeaway:** Supervised models require adequate training sample size ($N$) relative to model capacity to prevent high variance and achieve stable generalization.
