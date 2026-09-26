# Session 7 - Task 5: Spotify Popularity Model Overfitting Diagnosis

## Task Overview
Compare two regression models (Model 1: Ridge Linear Regression vs. Model 2: Complex Unconstrained Decision Tree) on Spotify song popularity predictions across Train and Test datasets to diagnose overfitting.

---

## Train vs. Test Performance Comparison Table

| Metric | Model 1 (Ridge) Train | Model 1 Test | Model 2 (Tree) Train | Model 2 Test | Generalization Status |
|---|---|---|---|---|---|
| **MSE** | 17.97 | 14.63 | **0.00** | **46.54** | Model 2 test error spikes |
| **RMSE** | 4.24 | 3.83 | **0.00** | **6.82** | Model 2 test error jumps |
| **MAE** | 3.46 | 3.23 | **0.00** | **5.57** | Model 2 train error is 0 |
| **$R^2$ Score** | **0.8513** | **0.8827** | **1.0000** | **0.6269** | Model 2 drops from 1.00 $\rightarrow$ 0.62 |

---

## Overfitting Identification & Reasoning

**Overfitting Model Identified:** **Model 2 (Unconstrained Decision Tree)**

**Reasoning:**
- **Model 2** demonstrates classic overfitting signature: perfect training fit ($\text{Train } R^2 = 1.0000$, $\text{Train RMSE} = 0.00$) paired with a steep drop in test performance ($\text{Test } R^2 = 0.6269$, $\text{Test RMSE} = 6.82$).
- In contrast, **Model 1 (Ridge)** exhibits strong generalization with consistent Train ($R^2 = 0.8513$) and Test ($R^2 = 0.8827$) metrics, proving robust on unseen test audio tracks.
