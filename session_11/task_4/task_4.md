# Session 11 - Task 4: WhatsApp Message Classifier (MultinomialNB & Confusion Matrix)

## Task Overview
Categorize WhatsApp chat messages into 3 target classes (`'personal'`, `'group'`, or `'spam'`) using **Multinomial Naive Bayes (`MultinomialNB`)** and `CountVectorizer`.

---

## Model Pipeline
1. **CountVectorizer**: Transform text tokens into word frequency counts:
   $$P(w_i \mid c) = \frac{N_{c, i} + \alpha}{N_c + \alpha |V|}$$
2. **MultinomialNB**: Compute posterior probability:
   $$P(c \mid \mathbf{x}) \propto P(c) \prod_{i=1}^{|V|} P(w_i \mid c)^{x_i}$$

---

## Model Performance & Confusion Matrix

- **Total Dataset Size**: 18 WhatsApp messages (12 train, 6 test)
- **Overall Test Accuracy**: **83.33%**

### Confusion Matrix Table

| Actual \ Predicted | `group` | `personal` | `spam` | Total Support |
|---|---|---|---|---|
| **`group`** | **2** | 0 | 0 | 2 |
| **`personal`** | 1 | **1** | 0 | 2 |
| **`spam`** | 0 | 0 | **2** | 2 |

---

## Detailed Classification Metrics

| Class Category | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| **`group`** | 0.67 | **1.00** | 0.80 | 2 |
| **`personal`** | **1.00** | 0.50 | 0.67 | 2 |
| **`spam`** | **1.00** | **1.00** | **1.00** | 2 |
