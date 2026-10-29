# Applying Probability and Statistics to Evaluate a Machine Learning Model

## Overview
This repository contains a complete, rigorous, and reproducible Python implementation for evaluating machine learning models using probability theory, sampling distributions, statistical estimation, and hypothesis testing.

### Core Question Investigated
> **"Does the improved model perform better, or could the observed difference be due to variation in the test sample?"**

---

## 1. Experimental Setup & Dataset

### Dataset: Wisconsin Diagnostic Breast Cancer (WDBC)
- **Source**: UCI Machine Learning Repository / Scikit-Learn (`load_breast_cancer`)
- **Total Instances**: 569 patient samples
- **Features**: 30 continuous clinical measurements computed from fine needle aspirate (FNA) images of breast masses (e.g., radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry).
- **Target Classes**: Binary classification — Malignant (212 samples, 37.3%) vs. Benign (357 samples, 62.7%).
- **Data Partitioning**: Stratified 75% training ($N_{\text{train}} = 426$) and 25% held-out test set ($N_{\text{test}} = 143$), preserving class ratios with `random_state=42`.

### Models Evaluated
1. **Baseline Model**: Restricted Decision Tree Classifier (`max_depth=3`, `random_state=42`).
   - Represents a fast, heuristic, and interpretable clinical decision benchmark.
2. **Improved Model**: Tuned Gradient Boosting Classifier Pipeline with Feature Standardization (`StandardScaler` + `GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, subsample=0.8, random_state=42)`).
   - Represents a modern non-linear ensemble with shrinkage and feature normalization.

---

## 2. Evaluation on the Identical Held-Out Test Set ($N = 143$)

Both models were evaluated on the exact same held-out test partition:

| Metric | Baseline (Decision Tree) | Improved (Gradient Boosting) | Difference ($\Delta = \text{Imp} - \text{Base}$) |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **0.9441** (94.41%) | **0.9580** (95.80%) | **+0.0140 (+1.40%)** |
| **Precision** | 0.9457 | 0.9468 | +0.0012 |
| **Recall (Sensitivity)** | 0.9667 | 0.9889 | +0.0222 |
| **Specificity** | 0.9057 | 0.9057 | +0.0000 |
| **F1-Score** | 0.9560 | 0.9674 | +0.0113 |
| **ROC-AUC** | 0.9463 | 0.9933 | +0.0470 |

### Observed Point Performance:
- **Baseline Correct**: 135 / 143 test cases
- **Improved Correct**: 137 / 143 test cases
- **Net Gain**: $+2$ test cases correctly classified ($+1.40\%$ accuracy).

---

## 3. Statistical Sampling & Confidence Intervals

Because evaluating on a single finite sample of size $N = 143$ is subject to sampling error, we applied **non-parametric paired bootstrap resampling** ($B = 2,000$ iterations) to estimate the true sampling distribution of $\Delta = \text{Acc}_{\text{improved}} - \text{Acc}_{\text{baseline}}$:

- **Bootstrap Mean Difference**: $+0.0144$
- **Standard Error ($SE_B(\Delta)$)**: $0.0168$
- **Bootstrap Bias**: $+0.00042$

### 95% Confidence Intervals for Difference in Accuracy:

| Method | Lower Bound (2.5%) | Upper Bound (97.5%) | Contains Zero ($0 \in \text{CI}$)? |
| :--- | :---: | :---: | :---: |
| **Percentile Bootstrap (95%)** | **-0.0210** (-2.10%) | **+0.0490** (+4.90%) | **YES** |
| **Normal Approximation Bootstrap (95%)** | **-0.0190** (-1.90%) | **+0.0470** (+4.70%) | **YES** |
| **Paired Wald Large-Sample (95%)** | **-0.0195** (-1.95%) | **+0.0475** (+4.75%) | **YES** |

> **Key Finding**: All 95% confidence intervals cross zero (ranging from $-2.10\%$ to $+4.90\%$). At the $\alpha = 0.05$ significance level, we cannot rule out that the true population performance difference is zero or even slightly negative.

---

## 4. Hypothesis Testing

### Hypotheses:
- **Null Hypothesis ($H_0$)**: $p_{\text{improved}} = p_{\text{baseline}}$ (No difference in accuracy between the two models).
- **Alternative Hypothesis ($H_1$)**: $p_{\text{improved}} \neq p_{\text{baseline}}$ (Two-sided test; difference in accuracy is non-zero).

### Contingency Table of Paired Predictions:
Evaluating both models on the identical test instances yields:
- Both Correct ($n_{00}$): **133**
- Baseline Correct, Improved Incorrect ($b = n_{01}$): **2**
- Baseline Incorrect, Improved Correct ($c = n_{10}$): **4**
- Both Incorrect ($n_{11}$): **4**
- **Total Discordant Pairs ($b + c$)**: **6**

### Test Results Summary:

| Statistical Test | Test Statistic | $p$-value | Decision ($\alpha = 0.05$) |
| :--- | :---: | :---: | :---: |
| **McNemar's Exact Binomial Test (Two-Sided)** | Discordant: $b=2, c=4$ | **0.6875** | **Fail to Reject $H_0$** |
| **McNemar's Chi-Squared (with Continuity Correction)** | $\chi^2 = 0.1667$ ($df = 1$) | **0.6831** | **Fail to Reject $H_0$** |
| **Paired Permutation / Randomization Test (10,000 runs)** | $\Delta = +0.0140$ | **0.6855** | **Fail to Reject $H_0$** |
| **Dietterich 5x2 Cross-Validation Paired $t$-Test** | $t = 3.0956$ ($df = 5$) | **0.0270** | **Reject $H_0$** |

---

## 5. Direct Scientific Answer to the Core Question

> ### **Does the improved model perform better, or could the observed difference be due to variation in the test sample?**
>
> **Definitive Answer:**
> **On this held-out test sample, we CANNOT conclude that the improved model performs reliably better. The observed $+1.40\%$ difference can plausibly be attributed entirely to random variation in the test sample draw.**

### Mathematical & Statistical Justification:
1. **Discordant Sample Scarcity**: Out of 143 test cases, the two models agree on 137 cases and disagree on only 6 cases ($b = 2$, $c = 4$). Under $H_0$, each discordant case has a 50% probability of favoring either model ($b \sim \text{Binomial}(6, 0.5)$). The probability of observing 4 or more successes purely by chance is **$68.75\%$** ($p = 0.6875 \gg 0.05$).
2. **Confidence Interval Spans Zero**: The 95% bootstrap confidence interval $[-0.0210, +0.0490]$ encompasses both positive and negative values.
3. **Single Test Set vs. Repeated Resampling**: While the single held-out test set lacks statistical power to confirm superiority due to sample variance, the Dietterich 5x2 cross-validation test across repeated folds on the full dataset reveals a statistically significant improvement ($p = 0.0270 < 0.05$). This illustrates why machine learning practitioners must never rely solely on a single small test set point estimate.

---

## 6. How to Run the Code

### Prerequisites:
```bash
pip install numpy pandas scipy scikit-learn matplotlib
```

### Execution:
```bash
python evaluate_model_statistics.py
```

### Output Artifacts:
- **Console Output**: Formatted tables displaying performance metrics, bootstrap estimates, confidence intervals, McNemar contingency table, and hypothesis testing verdicts.
- **Diagnostic Plot (`model_statistical_evaluation.png`)**: High-resolution 4-panel figure containing:
  1. ROC curves with AUC scores
  2. McNemar paired contingency heatmap
  3. Bootstrap sampling distribution of $\Delta$ with 95% CI bounds
  4. Permutation test null distribution vs observed difference

