# Session 7 - Task 4: Model Selection Trade-Off Analysis (Swiggy Delivery Times)

## Task Overview
Evaluate and select the superior model for Swiggy food delivery time estimation between **Model A** ($\text{RMSE} = 8\text{ mins}$, $R^2 = 0.65$) and **Model B** ($\text{RMSE} = 6\text{ mins}$, $R^2 = 0.55$).

---

## Model Metrics Comparison Table

| Model Name | RMSE (Minutes) | $R^2$ Score | Primary Metric Focus | Recommended Choice |
|---|---|---|---|---|
| **Model A** | 8.0 minutes | **0.65 (65%)** | Higher Variance Explanation | ❌ |
| **Model B** | **6.0 minutes** | 0.55 (55%) | **Lower Prediction Error Margin** | **✅ Recommended** |

---

## Selection Decision & Detailed Reasoning

**Recommended Choice:** **Model B**

**Reasoning:**
1. **Direct Customer Experience Impact (RMSE):** In food delivery operations (Swiggy), minimizing real-time delivery ETA errors is the primary objective. Model B provides a **25% reduction in error magnitude** ($\text{RMSE} = 6\text{ minutes}$ compared to 8 minutes for Model A), resulting in tighter, more reliable customer delivery windows.
2. **$R^2$ vs. RMSE Metric Nuance:** While Model A explains a slightly higher proportion of dataset variance ($R^2 = 0.65$ vs $0.55$), $R^2$ is sensitive to overall sample variance. For end-user delivery promises, an actual error margin of $\pm 6\text{ mins}$ provides vastly superior operational reliability than $\pm 8\text{ mins}$.
