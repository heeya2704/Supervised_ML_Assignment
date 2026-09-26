# Session 8 - Task 1: Polynomial Features Transformation (Degree = 2)

## Task Overview
Use `scikit-learn`'s `PolynomialFeatures(degree=2)` transformer on a mobile phone dataset containing features `ram_gb` and `storage_gb` to generate non-linear interactive features up to degree 2.

---

## Original Feature Matrix

| Sample # | `ram_gb` | `storage_gb` |
|---|---|---|
| Sample 1 | 4 GB | 64 GB |
| Sample 2 | 6 GB | 128 GB |
| Sample 3 | 8 GB | 128 GB |
| Sample 4 | 12 GB | 256 GB |
| Sample 5 | 16 GB | 512 GB |

---

## Degree-2 Polynomial Transformed Feature Matrix

| Sample # | Bias (`1`) | `ram_gb` | `storage_gb` | `ram_gb^2` | `ram_gb * storage_gb` | `storage_gb^2` |
|---|---|---|---|---|---|---|
| Sample 1 | 1.0 | 4.0 | 64.0 | 16.0 | 256.0 | 4,096.0 |
| Sample 2 | 1.0 | 6.0 | 128.0 | 36.0 | 768.0 | 16,384.0 |
| Sample 3 | 1.0 | 8.0 | 128.0 | 64.0 | 1,024.0 | 16,384.0 |
| Sample 4 | 1.0 | 12.0 | 256.0 | 144.0 | 3,072.0 | 65,536.0 |
| Sample 5 | 1.0 | 16.0 | 512.0 | 256.0 | 8,192.0 | 262,144.0 |

---

## Key Takeaways
- The original 2-feature matrix was expanded into a 6-feature polynomial space: `[1, ram_gb, storage_gb, ram_gb^2, ram_gb * storage_gb, storage_gb^2]`.
- The cross-product term (`ram_gb * storage_gb`) captures non-linear synergies between memory capacity and storage tiering.
