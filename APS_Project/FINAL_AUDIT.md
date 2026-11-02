# Final Statistical & Mathematical Audit Report

## Applied Probability and Statistics (AME 5102)
**Project Title:** Sales Demand Forecasting & Statistical Model Evaluation  
**Candidate Name:** Rohan Doggalli  
**Registration Number:** `261100670012`  
**Degree Program:** Master of Engineering (M.E.) in Big Data Analytics  
**Department:** Manipal School of Information Sciences (MSIS), MAHE, Manipal  
**Date of Audit:** November 2026

---

## 1. Mathematical Formulation & Audit of Statistical Calculations

### 1.1 Paired Observation Design
Because both the Baseline (Standardized Linear Regression) and Improved (Tuned XGBoost Regressor) models were trained on the identical training set ($N_{\text{train}} = 9,270$, 309 days) and evaluated on the exact same held-out test partition ($N_{\text{test}} = 840$, 28 days), the evaluation is an inherently **paired experiment**.

Let $y_i$ denote the ground-truth unit sales on test record $i \in \{1, \dots, N\}$.  
Let $\hat{y}_i^{\text{base}}$ and $\hat{y}_i^{\text{imp}}$ denote the predictions from the baseline and improved models.

The absolute errors are:
$$e_i^{\text{base}} = |y_i - \hat{y}_i^{\text{base}}|$$
$$e_i^{\text{imp}} = |y_i - \hat{y}_i^{\text{imp}}|$$

The paired absolute error difference $d_i$ is defined as:
$$d_i = e_i^{\text{base}} - e_i^{\text{imp}}$$

Under this definition:
- $d_i > 0 \implies$ Improved model has lower absolute error (outperformed baseline).
- $d_i < 0 \implies$ Baseline model has lower absolute error.
- $d_i = 0 \implies$ Identical absolute error.

### 1.2 Sample Moments Audit
On the $N = 840$ held-out test observations:
- **Sample Mean Difference ($\bar{d}$):**
  $$\bar{d} = \frac{1}{N} \sum_{i=1}^{N} d_i = \mathbf{+0.0107} \text{ units/day}$$
  *(Baseline MAE: $1.8158$ units/day, Improved MAE: $1.8051$ units/day $\implies \Delta = 1.8158 - 1.8051 = +0.0107$ units/day, representing a nominal $+0.59\%$ relative error reduction).*
- **Sample Standard Deviation ($s_d$):**
  $$s_d = \sqrt{\frac{1}{N-1} \sum_{i=1}^N (d_i - \bar{d})^2} = \mathbf{0.7845} \text{ units/day}$$
- **Standard Error of the Mean ($SE_{\bar{d}}$):**
  $$SE_{\bar{d}} = \frac{s_d}{\sqrt{N}} = \frac{0.7845}{\sqrt{840}} = \mathbf{0.0271} \text{ units/day}$$
- **Standardized Effect Size (Cohen's $d$):**
  $$\text{Cohen's } d = \frac{\bar{d}}{s_d} = \frac{0.0107}{0.7845} = \mathbf{0.0136}$$
  *According to Cohen's (1988) taxonomy, $d < 0.20$ represents a negligible effect size. An effect size of $0.0136$ indicates an overlap exceeding $99.4\%$ between the two error distributions.*

---

## 2. Hypothesis Testing Battery Audit

### 2.1 Hypotheses Formulation
$$\text{Null Hypothesis } (H_0): \mu_d \le 0 \quad (\text{Improved model does not reduce forecast error})$$
$$\text{Alternative Hypothesis } (H_1): \mu_d > 0 \quad (\text{Improved model significantly reduces forecast error})$$
$$\text{Significance Level: } \alpha = 0.05$$

### 2.2 Paired Student's $t$-Test
- **Test Statistic Formula:**
  $$t = \frac{\bar{d} - 0}{s_d / \sqrt{N}} = \frac{0.0107}{0.0271} = \mathbf{0.3954}$$
- **Degrees of Freedom:** $df = N - 1 = 839$.
- **One-Sided $p$-Value:**
  $$p = 1 - F_{t(839)}(0.3954) = \mathbf{0.3463}$$
- **Decision:** Because $p = 0.3463 > 0.05$, **FAIL TO REJECT $H_0$**.
- **Normality Violation Warning:** Normality of $d_i$ is severely violated (Shapiro-Wilk $W = 0.9759, p = 1.44 \times 10^{-10}$; D'Agostino $K^2 = 44.00, p = 2.78 \times 10^{-10}$). Hence, parametric $t$-test inference is complemented by non-parametric and resampling tests.

### 2.3 Non-Parametric Wilcoxon Signed-Rank Test
- Conducted on non-zero paired differences ($N_{\text{non-zero}} = 840$):
  $$W = \sum_{i: d_i > 0} \text{Rank}(|d_i|) = \mathbf{178,868.0}$$
- **One-Sided $p$-Value:** $p = \mathbf{0.3741}$.
- **Decision:** Because $p = 0.3741 > 0.05$, **FAIL TO REJECT $H_0$**. The median error difference is not statistically distinguishable from zero.

### 2.4 Diebold-Mariano HAC Test with Harvey-Leybourne-Newbold Correction
- Designed specifically for time-series forecast comparisons under autocorrelation:
  $$\hat{V}_{\text{HAC}} = \hat{\gamma}_0 + 2 \sum_{k=1}^{h-1} \left(1 - \frac{k}{h}\right) \hat{\gamma}_k$$
  $$\text{HLN Factor} = \sqrt{\frac{N + 1 - 2h + h(h-1)/N}{N}}$$
  $$DM = \frac{\bar{d}}{\sqrt{\hat{V}_{\text{HAC}} / N}} \times \text{HLN Factor} = \mathbf{0.3973}$$
- **One-Sided $p$-Value:** $p = \mathbf{0.3456}$.
- **Decision:** Because $p = 0.3456 > 0.05$, **FAIL TO REJECT $H_0$**.

### 2.5 Moving Block Bootstrap (MBB, $L = 7$ Days, $B = 2,000$ Resamples)
- Resampling algorithm: Partitions the 28-day evaluation calendar into overlapping blocks of length $L = 7$ days (preserving 7-day weekly seasonality and autocorrelation).
- For each bootstrap replication $b \in \{1, \dots, 2000\}$, $k = \lceil 28 / 7 \rceil = 4$ blocks are sampled with replacement, concatenated, and the bootstrap mean $\bar{d}^{*(b)}$ is recorded.
- **Bootstrap Standard Error ($SE_{\text{boot}}$):** $0.0214$ units/day.
- **Empirical 95% Percentile Confidence Interval:**
  $$\text{CI}_{95\%} = [\text{Percentile}(2.5), \text{Percentile}(97.5)] = [\mathbf{-0.0297}, \mathbf{+0.0475}]$$
- **Zero Containment:** $[-0.0297, +0.0475]$ contains $0.0000$.
- **Empirical Bootstrap $p$-Value:**
  $$p_{\text{boot}} = \frac{1}{B} \sum_{b=1}^{B} \mathbb{I}(\bar{d}^{*(b)} \le 0) = \mathbf{0.2270}$$
- **Decision:** Because $p = 0.2270 > 0.05$, **FAIL TO REJECT $H_0$**. 22.7% of all bootstrap pseudo-worlds favored the baseline model.

---

## 3. Comprehensive Audit of the ₹103,160.17 Inventory Savings Claim

### 3.1 Origin of the Figure
In early exploratory brainstorming, a calculation was introduced attempting to estimate the monetary impact of demand forecast accuracy:
$$\text{Annual Units Saved per Item} = 0.0107 \text{ units/day} \times 365 \text{ days} = 3.9055 \text{ units/year}$$
$$\text{Dollar Value per Series} = 3.9055 \text{ units} \times \$4.67 \text{ (average product price)} = \$18.24$$
$$\text{Hypothetical Store Catalog (3,000 SKUs)} = \$18.24 \times 3,000 = \$54,716 \text{ (approx ₹103,160 across holding costs)}$$

### 3.2 Methodological Audit & Why It Is Rejected
1. **Statistical Insignificance:** The underlying point estimate $+0.0107$ has a $p$-value of $0.2270 - 0.3741$, meaning there is a $23\% - 37\%$ probability of observing this difference purely by chance when the true difference is zero or negative.
2. **Negative Confidence Interval Region:** The 95% confidence interval spans down to $-0.0297$. Under the same arithmetic, a lower-bound estimate would indicate an annual **loss** of $\$152,000$. Presenting only the positive point estimate without its confidence band misrepresents the empirical reality.
3. **Negligible Effect Size:** Cohen's $d = 0.0136$ indicates that random noise within each model's predictions accounts for over $99.98\%$ of the total variance.
4. **Supply Chain Disconnect:** Real-world inventory ordering follows discrete lot sizes (Economic Order Quantity / EOQ) and safety stock reorder thresholds; an unverified sub-fraction of a single unit per day does not translate to warehouse savings.

### 3.3 Audit Action Taken
- The ₹103,160.17 figure has been **completely eliminated** from all code, output logs, CSV files, and final summaries.
- The academic report specifically documents this audit in Section 27 as an educational demonstration of the dangers of extrapolating statistically non-significant machine learning metrics into hypothetical business revenue.

---

## 4. Verification Checklist of Deliverables

- [x] **Statistical Calculations:** All verified against executed code with exact 4-decimal precision.
- [x] **Hypothesis Testing Battery:** Paired t-test ($p=0.3463$), Wilcoxon ($p=0.3741$), Diebold-Mariano ($p=0.3456$), and MBB ($p=0.2270$) verified.
- [x] **95% Confidence Intervals:** Verified to span across zero ($[-0.0297, +0.0475]$).
- [x] **Unsupported Financial Extrapolations:** Audited and purged.
- [x] **Visualizations Catalog:** All 10 publication-quality diagnostic graphs generated at 300 DPI in `outputs/graphs/`.
- [x] **Academic Word Report:** Complete 28 sections with all 10 figures embedded in `Sales_Demand_Forecasting_Report.docx` (3.48 MB).
- [x] **Documentation:** `README.md`, `requirements.txt`, `IMPLEMENTATION_STATUS.md`, and `FINAL_AUDIT.md` complete and verified.
- [x] **Execution Verification:** Both `run_complete_evaluation.py` and `main.py` run cleanly to completion with exit code 0.
