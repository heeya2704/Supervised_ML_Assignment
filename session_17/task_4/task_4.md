# Session 17 - Task 4: Log Loss (Cross-Entropy Loss) Computation

## Task Overview
Calculate the Log Loss (Binary Cross-Entropy Loss) on test set predictions using `sklearn.metrics.log_loss`.

---

## Mathematical Formula

$$\text{Log Loss} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right]$$

Where:
- $N$ is the number of test samples.
- $y_i \in \{0, 1\}$ is the actual binary class label.
- $p_i \in (0, 1)$ is the predicted probability for the positive class.

---

## Result & Explanation

- **Calculated Test Log Loss**: **0.0812**
- **Interpretation**: Log Loss penalizes wrong predictions based on the confidence level of the probability prediction. A value close to 0 indicates high calibration accuracy and strong prediction certainty.
