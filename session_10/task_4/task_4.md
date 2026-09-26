# Session 10 - Task 4: Customizable Probability Classifier

## Task Overview
Define a custom classification function `classify_virality(probability, threshold=0.5)` that maps continuous logistic regression probabilities $p \in [0, 1]$ into binary predictions (`'Viral'` vs `'Not Viral'`) using a configurable threshold cutoff.

$$\hat{y} = \begin{cases} \text{'Viral'}, & \text{if } P(\text{viral}) \ge \text{threshold} \\ \text{'Not Viral'}, & \text{if } P(\text{viral}) < \text{threshold} \end{cases}$$

---

## Test Results Matrix

### 1. Default Decision Threshold ($\tau = 0.5$)
| Input Probability ($p$) | Threshold ($\tau$) | Condition ($p \ge \tau$) | Classification Output |
|---|---|---|---|
| **0.20** | 0.50 | $0.20 < 0.50$ | **Not Viral** |
| **0.60** | 0.50 | $0.60 \ge 0.50$ | **Viral** |
| **0.80** | 0.50 | $0.80 \ge 0.50$ | **Viral** |

---

### 2. Strict Decision Threshold ($\tau = 0.7$)
| Input Probability ($p$) | Threshold ($\tau$) | Condition ($p \ge \tau$) | Classification Output |
|---|---|---|---|
| **0.20** | 0.70 | $0.20 < 0.70$ | **Not Viral** |
| **0.60** | 0.70 | $0.60 < 0.70$ | **Not Viral** *(Shifted from Viral)* |
| **0.80** | 0.70 | $0.80 \ge 0.70$ | **Viral** |

---

## Key Takeaways & Practical Trade-offs
1. **Sensitivity to Threshold ($\tau$)**:
   - At $\tau = 0.5$, a post with $60\%$ predicted virality probability is classified as **Viral**.
   - Increasing the threshold to $\tau = 0.7$ flips the classification of $p = 0.60$ to **Not Viral**, requiring a higher confidence level for positive classification.
2. **Business Application**:
   - In high-stakes applications (e.g., ad allocation for viral posts), raising the threshold increases **Precision** (fewer false positives). Lowering the threshold increases **Recall** (captures more potential viral posts).
