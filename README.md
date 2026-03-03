# Statistical Characterization of Embedding Geometry in Pretrained Language Models

![Covariance Spectrum Decay](d009b6fd-70af-4ea3-b5d3-2f03d55ac388.png)

---

## Experimental Summary

| Model | Effective Rank | Spectral Entropy | Anisotropy Ratio | Linear Probe Accuracy |
|--------|---------------|-----------------|------------------|-----------------------|
| all-MiniLM-L6-v2 | 167.84 | 5.12 | 19.72 | 0.8767 |
| paraphrase-MiniLM-L3-v2 | 143.28 | 4.96 | 16.42 | 0.8550 |
| all-mpnet-base-v2 | 205.36 | 5.32 | 34.35 | 0.8950 |

**Observed Correlations**

- Pearson(Effective Rank, Accuracy) = **0.9859**
- Pearson(Anisotropy, Accuracy) = **0.9218**

---

## Problem Statement

Modern pretrained language models produce high-dimensional embedding spaces that power downstream tasks such as classification, retrieval, and semantic similarity. While predictive performance is well studied, the statistical geometry of these embedding spaces remains under-analyzed.

This work investigates:

> How does the spectral structure of embedding covariance relate to downstream generalization performance?

---

## Mathematical Framework

Let  

$$
X = \{x_1, x_2, \dots, x_n\}, \quad x_i \in \mathbb{R}^d
$$

be L2-normalized sentence embeddings extracted from a pretrained transformer model.

### Covariance Estimation

We compute the empirical covariance matrix:

$$
\Sigma = \frac{1}{n} \sum_{i=1}^{n} (x_i - \mu)(x_i - \mu)^T
$$

where

$$
\mu = \frac{1}{n} \sum_{i=1}^{n} x_i
$$

Let the eigenvalues of $\Sigma$ be:

$$
\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_d \ge 0
$$

The eigenvalue decay profile characterizes variance concentration across principal directions.

---

## Spectral Geometry Metrics

### Effective Rank (Intrinsic Dimensionality)

Define normalized spectrum:

$$
p_i = \frac{\lambda_i}{\sum_j \lambda_j}
$$

Effective rank is defined as:

$$
\text{erank}(\Sigma) = \exp\left(-\sum_i p_i \log p_i\right)
$$

Higher values indicate reduced spectral collapse and greater dimensional utilization.

---

### Spectral Entropy

$$
H(\Sigma) = -\sum_i p_i \log p_i
$$

This quantifies dispersion of variance across principal components.

---

### Anisotropy Ratio

We measure geometric anisotropy as:

$$
\text{Anisotropy} = \frac{\lambda_{\max}}{\frac{1}{d} \sum_i \lambda_i}
$$

Higher anisotropy indicates variance concentration along dominant principal directions.

---

## Empirical Observations

### Intrinsic Dimensionality vs Accuracy

Effective rank shows a strong positive correlation with linear probe accuracy:

$$
\rho \approx 0.986
$$

Embedding spaces with higher intrinsic dimensionality exhibit improved linear separability.

---

### Anisotropy and Performance

Anisotropy correlates positively with performance:

$$
\rho \approx 0.922
$$

This suggests dominant subspace structure contributes to class separation.

---

## Downstream Evaluation

A logistic regression linear probe was trained on AG News classification using frozen embeddings.

Observed accuracy:

- all-mpnet-base-v2: 0.8950  
- all-MiniLM-L6-v2: 0.8767  
- paraphrase-MiniLM-L3-v2: 0.8550  

The ranking aligns with effective rank ordering.

---

## Statistical Reliability

Bootstrap resampling (200 iterations, 70% subsampling) was used to compute confidence intervals for spectral metrics, ensuring robustness under sample variability.

---

## Conclusion

Embedding covariance geometry is strongly predictive of linear separability performance.

Spectral metrics such as effective rank and anisotropy provide interpretable, statistically grounded indicators of representational quality in pretrained transformer models.

This work bridges empirical performance and statistical geometry in modern language representations.
