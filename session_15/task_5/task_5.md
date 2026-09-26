# Session 15 - Task 5: Real-World Use Case of Boosting in Indian Mobile Apps

## Real-World Case Study: Real-Time Fraud Detection in Paytm UPI Payments

### Description & Performance Impact (3–4 lines):
1. **Sequential Error Correction**: Boosting models (e.g. XGBoost / Gradient Boosting) train sequential decision trees where each new tree specifically targets fraudulent payment patterns misclassified by earlier trees.
2. **Multi-Signal Synthesis**: Paytm evaluates non-linear interactions across hundreds of weak indicators—including device ID switches, transaction velocity spikes, IP geofencing anomalies, and nocturnal high-value transfers.
3. **High-Precision Real-Time Defense**: Boosting converts weak individual rules into a high-precision strong classifier, processing millions of UPI transactions per second with ultra-low latency while preventing false payment declines for genuine customers.
