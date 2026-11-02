# Implementation Status & Task Verification Checklist

## Applied Probability and Statistics (AME 5102)
**Student Name:** Rohan Doggalli | **Reg. No.:** 261100670012  
**Degree:** M.E. Big Data Analytics, MSIS, MAHE, Manipal  
**Date of Audit:** November 2026

---

## 1. Summary of Remaining Tasks & Verification Status

| Task # | Task Description | Status | Verification Evidence / File Location |
|---|---|---|---|
| **1** | Audit actual implementation of all statistical calculations | **VERIFIED & PASSED** | `src/statistical_analysis.py`, `run_complete_evaluation.py` |
| **2** | Verify all model results against executed code | **VERIFIED & PASSED** | Baseline MAE = 1.8158, Improved MAE = 1.8051 ($\Delta = +0.0107$) |
| **3** | Verify paired t-test, Wilcoxon, Diebold-Mariano, and MBB | **VERIFIED & PASSED** | All 4 tests executed; all $p > 0.05 \implies$ Fail to Reject $H_0$ |
| **4** | Verify 95% Confidence Interval and $p$-values | **VERIFIED & PASSED** | MBB 95% CI: $[-0.0297, +0.0475]$ (contains 0.0000; $p = 0.2270$) |
| **5** | Verify ₹103,160.17 inventory savings and remove if unsupported | **AUDITED & PURGED** | Purged as statistically unsupported extrapolation of noise |
| **6** | Generate all 10 diagnostic visualizations | **VERIFIED & PASSED** | All 10 plots generated at 300 DPI in `outputs/graphs/` |
| **7** | Check academic report against all 28 required sections | **VERIFIED & PASSED** | All 28 sections written in `Sales_Demand_Forecasting_Report.docx` |
| **8** | Verify every required graph is correctly embedded in DOCX | **VERIFIED & PASSED** | Exactly 10 embedded image shapes verified via python-docx |
| **9** | Create/complete README.md, requirements.txt, IMPLEMENTATION_STATUS.md, FINAL_AUDIT.md | **VERIFIED & PASSED** | Complete documentation suite compiled and synchronized |
| **10** | Verify all project files and folders | **VERIFIED & PASSED** | Both workspace root and `APS_Project` verified clean and runnable |
| **11** | Execute both `run_complete_evaluation.py` and `main.py` | **VERIFIED & PASSED** | Both executed successfully, exit code 0, matching terminal output |
| **12** | Fix errors and regenerate report if necessary | **VERIFIED & PASSED** | Word report generated at 3.48 MB with zero XML/styling errors |
| **13** | Verify final outputs match actual experimental results | **VERIFIED & PASSED** | Exact 4-decimal precision match across console, CSV, and report |

---

## 2. Model Evaluation Verification Matrix

| Model | MAE (Units/Day) | RMSE (Units/Day) | $R^2$ Score | sMAPE (%) | Source Function |
|---|---|---|---|---|---|
| **Baseline (Standardized Linear Regression)** | 1.8158 | 2.3380 | 0.4457 | 46.49% | `train_and_evaluate_models()` |
| **Improved (Tuned XGBoost Regressor)** | 1.8051 | 2.3256 | 0.4516 | 46.18% | `train_and_evaluate_models()` |
| **Absolute Improvement ($\Delta$)** | **+0.0107** | **+0.0124** | **+0.0059** | **+0.31%** | Positive indicates error reduction |
| **Relative Improvement (%)** | **+0.59%** | **+0.53%** | **+1.32%** | **+0.67%** | Relative gain over baseline |

---

## 3. Inferential Hypothesis Testing & Resampling Audit

| Statistical Test | Test Statistic Value | $p$-value (One-Sided) | $\alpha$ Threshold | Decision | Scientific Interpretation |
|---|---|---|---|---|---|
| **Paired Student's $t$-test** | $t = 0.3954$ ($df = 839$) | $p = 0.3463$ | 0.05 | **Fail to Reject $H_0$** | Difference is not statistically significant |
| **Wilcoxon Signed-Rank Test** | $W = 178,868.0$ | $p = 0.3741$ | 0.05 | **Fail to Reject $H_0$** | Median error reduction not significant |
| **Diebold-Mariano HAC Test** | $DM = 0.3973$ | $p = 0.3456$ | 0.05 | **Fail to Reject $H_0$** | Robust to autocorrelation; not significant |
| **Moving Block Bootstrap ($L=7$d, $B=2000$)** | $\Delta = +0.0107$ | $p = 0.2270$ | 0.05 | **Fail to Reject $H_0$** | 22.7% of resamples favored baseline |

### 95% Confidence Interval Verification
- **Moving Block Bootstrap (MBB, $L=7$ days, $B=2,000$):**
  - Point Estimate ($\bar{d}$): $+0.0107$ units/day
  - Standard Error (SE): $0.0214$ units/day
  - **95% Percentile Confidence Interval:** **$[-0.0297, +0.0475]$**
  - **Contains Zero? YES.** Because zero lies comfortably within the interval, we cannot reject the null hypothesis of zero true improvement.
- **Normal Approximation Bootstrap:**
  - **95% Normal CI:** **$[-0.0313, +0.0527]$**
  - **Contains Zero? YES.**

---

## 4. Assumption Diagnostic Test Results

| Diagnostic Test | Test Statistic | $p$-value | Threshold | Assumption Violated? | Methodological Consequence |
|---|---|---|---|---|---|
| **Shapiro-Wilk Normality** | $W = 0.9759$ | $p = 1.4400 \times 10^{-10}$ | $p < 0.05$ | **YES (Severely Violated)** | Parametric $t$-test cannot be trusted alone |
| **D'Agostino-Pearson Omnibus** | $K^2 = 44.0030$ | $p = 2.7853 \times 10^{-10}$ | $p < 0.05$ | **YES (Severely Violated)** | Non-Gaussian skewness (+0.51) & kurtosis (4.28) |
| **Durbin-Watson Autocorrelation** | $DW = 2.0179$ | Expected $\sim 2.0$ | $|DW - 2| > 0.4$ | **NO (Acceptable)** | Residual serial correlation is low |
| **Ljung-Box Test (Lag 7)** | $Q = 10.05$ | $p = 0.1849$ | $p < 0.05$ | **NO (Acceptable)** | Weekly lag correlation not statistically significant |

---

## 5. Audit & Elimination of Unsupported Inventory Cost Savings

- **Target Claim:** Earlier experimental drafts mentioned an estimated catalog-wide inventory saving of ₹103,160.17 (or $103,160.17).
- **Audit Findings:**
  1. The figure was calculated as $0.0107 \times 365 \times 4.67 \times 3000 = \$54,766$ (or ₹103,160.17 when scaled across two stores / holding cost assumptions).
  2. The underlying metric $\bar{d} = 0.0107$ has a $p$-value of $0.2270 - 0.3741$ and a 95% Confidence Interval of $[-0.0297, +0.0475]$.
  3. Standardized Effect Size: Cohen's $d = 0.0136$ (negligible).
  4. Extrapolating a difference that is indistinguishable from zero across 3,000 products compounds statistical noise and creates a mathematically unfounded claim of financial savings.
- **Action Taken:**
  - The ₹103,160.17 figure has been **completely removed** from `main.py`, `run_complete_evaluation.py`, `src/statistical_analysis.py`, and the academic report.
  - Section 27 of `Sales_Demand_Forecasting_Report.docx` contains an explicit audit discussion explaining why such extrapolations violate statistical integrity.

---

## 6. Diagnostic Visualizations Catalog (All 10 Graphs Generated at 300 DPI)

| # | File Name | Description | Verified File Size |
|---|---|---|---|
| **1** | `01_actual_vs_predicted_sales.png` | Aggregate 28-day daily demand & SKU-level trajectory | 691,888 bytes |
| **2** | `02_baseline_vs_improved_performance.png` | Bar chart of MAE, RMSE, $R^2$, and sMAPE metrics | 281,896 bytes |
| **3** | `03_distribution_paired_error_differences.png` | Histogram & KDE of paired error differences with zero line | 237,417 bytes |
| **4** | `04_bootstrap_confidence_interval_plot.png` | MBB ($B=2000$) sampling distribution showing 95% CI spanning 0 | 287,209 bytes |
| **5** | `05_residual_error_comparison.png` | Daily mean residuals over time & residuals vs fitted values | 1,027,559 bytes |
| **6** | `06_qq_plot_normality_diagnostic.png` | Normal Q-Q plot showing heavy-tailed non-Gaussian distribution | 264,576 bytes |
| **7** | `07_autocorrelation_acf_pacf.png` | ACF and PACF plots across lags 1–14 with 95% Bartlett bounds | 183,937 bytes |
| **8** | `08_cumulative_absolute_error_difference.png` | Cumulative daily absolute error reduction over 28-day test period | 303,917 bytes |
| **9** | `09_sku_level_error_reduction_breakdown.png` | Per-SKU error difference breakdown across all 30 series | 382,757 bytes |
| **10** | `10_bootstrap_block_size_sensitivity.png` | Sensitivity of bootstrap SE and 95% CI to block length $L$ | 218,859 bytes |

---

## 7. Execution Commands Verification

1. **Standalone All-in-One Execution:**
   - Command: `python run_complete_evaluation.py`
   - Exit Code: `0` (Success)
   - Runtime: ~6 seconds
   - Output: Formatted numerical tables, paired t-test assumption diagnostics, 95% bootstrap CIs, hypothesis test battery, and scientific verdict.

2. **Modular Main Execution:**
   - Command: `python main.py`
   - Exit Code: `0` (Success)
   - Runtime: ~6 seconds
   - Output: Formatted numerical tables and statistical decision.

3. **Graph Generation Execution:**
   - Command: `python generate_all_10_graphs.py`
   - Exit Code: `0` (Success)
   - Output: All 10 figures saved at 300 DPI.

4. **Report Compilation Execution:**
   - Command: `python build_complete_28_section_report.py`
   - Exit Code: `0` (Success)
   - Output: `Sales_Demand_Forecasting_Report.docx` (3.48 MB, 28 sections, 10 embedded figures, 6 formatted tables).
