# Sales Demand Forecasting & Statistical Model Evaluation

## Applied Probability and Statistics (AME 5102) — Master's Degree Assignment
**Student Name:** Rohan Doggalli  
**Registration Number:** `261100670012`  
**Degree Program:** Master of Engineering (M.E.) in Big Data Analytics  
**Institution:** Manipal School of Information Sciences (MSIS), Manipal Academy of Higher Education (MAHE), Manipal  
**Investigation Domain:** Retail Sales Demand Forecasting & Inferential Machine Learning Model Evaluation  
**Benchmark Dataset:** M5 Forecasting Competition (Walmart Daily Unit Sales)

---

## 1. Core Research Question
> **"Does the improved model perform better, or could the observed difference be due to variation in the test sample?"**

Machine learning practitioners often declare one model superior to another based solely on aggregate point metrics (e.g., lower Mean Absolute Error on a test set). However, in empirical forecasting, test set performance is a random variable subject to sampling variability. This project applies statistical estimation, non-parametric resampling, and formal hypothesis testing to rigorously determine whether an observed +0.59% error reduction is statistically genuine or merely test sample noise.

---

## 2. Executive Scientific Verdict
- **Observed Point Performance:**
  - Baseline (Standardized Linear Regression) MAE: **1.8158 units/day**
  - Improved (Tuned XGBoost Regressor) MAE: **1.8051 units/day**
  - Observed Absolute Error Reduction ($\Delta\text{MAE}$): **+0.0107 units/day (+0.59%)**
- **Inferential Statistical Testing Battery ($\alpha = 0.05$):**
  - Paired Student's $t$-test: $t = 0.3954$ ($df = 839$), $p = 0.3463 \implies$ **Fail to Reject $H_0$**
  - Wilcoxon Signed-Rank Test: $W = 178,868.0$, $p = 0.3741 \implies$ **Fail to Reject $H_0$**
  - Diebold-Mariano HAC Test (HLN corrected): $DM = 0.3973$, $p = 0.3456 \implies$ **Fail to Reject $H_0$**
  - Moving Block Bootstrap Test ($L = 7$ days, $B = 2,000$): $\Delta = +0.0107$, $p = 0.2270 \implies$ **Fail to Reject $H_0$**
- **95% Moving Block Bootstrap Confidence Interval:** **$[-0.0297, +0.0475]$** (comfortably spans across zero).
- **Effect Size:** Cohen's $d = 0.0136$ (negligible; $< 0.20$).
- **Conclusion:** **FAIL TO REJECT $H_0$.** The observed $+0.59\%$ improvement is **not statistically significant** and is entirely consistent with random variation in the held-out test sample.
- **Audit Finding:** Unsupported back-of-the-envelope catalog financial claims (such as ₹103,160.17) have been audited and removed because extrapolating statistically non-significant noise violates fundamental statistical principles.

---

## 3. Dataset Architecture & Leakage-Safe Feature Engineering
- **Benchmark Source:** Walmart M5 Forecasting Competition dataset.
- **Sample Selection:** 15 food department items (`FOODS_1_001` through `FOODS_1_015`) across two stores (`CA_1`, `CA_2`) in California over a full 365-day calendar year (2015-01-01 to 2015-12-31), yielding 10,950 total records.
- **Temporal Splitting (Zero Lookahead Bias):**
  - Training Period: 2015-01-29 to 2015-12-03 (309 days $\times$ 30 series = 9,270 observations).
  - Held-Out Evaluation Test Period: 2015-12-04 to 2015-12-31 (final 28 days $\times$ 30 series = 840 observations).
  - Initial 28 days (840 rows) dropped as warm-up buffer for lag creation.
- **30 Leakage-Safe Engineered Features:**
  - *Calendar cycles:* `wday`, `day_of_week`, `day_of_month`, `month`, `year`, `quarter`, `week_of_year`, `is_weekend`.
  - *Autoregressive lags:* Lags of 7, 14, 21, and 28 days (`sales_lag_7`, `sales_lag_14`, etc.).
  - *Rolling statistics on shifted sales:* Rolling mean and standard deviation over 7, 14, and 28 days computed strictly on `shift(1)` to prevent target leakage.
  - *Pricing dynamics:* Current selling price, 28-day historical rolling price baseline, and relative price ratio.
  - *Promotions:* Walmart SNAP benefit flags (`snap_CA`, `snap_TX`, `snap_WI`) and cultural event indicators.

---

## 4. Project Directory Structure
```
261100670012_APS_Assignment/
│
├── data/
│   ├── processed_sales.parquet          # Fast binary multi-series sales dataset
│   └── m5_sales_sample.csv              # CSV fallback dataset
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py            # Feature engineering & chronological splitting
│   ├── train_models.py                  # Linear Regression & Tuned XGBoost pipelines
│   ├── statistical_analysis.py          # Assumptions, MBB, Wilcoxon, Diebold-Mariano
│   └── visualization.py                 # Graph plotting utilities
│
├── outputs/
│   ├── graphs/                          # All 10 publication-grade diagnostic plots (300 DPI)
│   │   ├── 01_actual_vs_predicted_sales.png
│   │   ├── 02_baseline_vs_improved_performance.png
│   │   ├── 03_distribution_paired_error_differences.png
│   │   ├── 04_bootstrap_confidence_interval_plot.png
│   │   ├── 05_residual_error_comparison.png
│   │   ├── 06_qq_plot_normality_diagnostic.png
│   │   ├── 07_autocorrelation_acf_pacf.png
│   │   ├── 08_cumulative_absolute_error_difference.png
│   │   ├── 09_sku_level_error_reduction_breakdown.png
│   │   └── 10_bootstrap_block_size_sensitivity.png
│   └── results/                         # Numerical CSV summaries of models and tests
│       ├── model_evaluation_metrics.csv
│       ├── test_assumptions_diagnostic.csv
│       ├── bootstrap_confidence_intervals.csv
│       ├── statistical_hypothesis_tests.csv
│       └── test_predictions_comparison.csv
│
├── app.py                               # Interactive Streamlit Web Dashboard
├── main.py                              # Modular runner (prints clean numerical tables to terminal)
├── run_complete_evaluation.py           # Self-contained all-in-one standalone script
├── generate_all_10_graphs.py            # Script generating all 10 diagnostic figures
├── build_complete_28_section_report.py  # Automated generator for the 28-section academic report
├── Sales_Demand_Forecasting_Report.docx # Complete 28-section Word report with all 10 graphs
├── requirements.txt                     # Python package dependencies
├── IMPLEMENTATION_STATUS.md             # Task-by-task audit and completion record
├── FINAL_AUDIT.md                       # Comprehensive statistical verification audit
└── README.md                            # Complete project guide (this file)
```

---

## 5. Local Execution Instructions

### Prerequisites
- Python 3.10, 3.11, or 3.12 on Windows / Linux / macOS.
- Virtual environment recommended.

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Launch the Interactive Streamlit Dashboard
```bash
streamlit run app.py
```
*Launches the full interactive web application in your default browser at `http://localhost:8501`. Features sidebar filters, Plotly charts, multi-horizon demand forecasting, assumption diagnostics, bootstrap sampling distributions, and statistical safety stock / inventory replenishment calculations with one-click CSV export.*

### Step 3: Run the All-In-One Self-Contained Script
```bash
python run_complete_evaluation.py
```
*Executes the complete pipeline (data ingestion, feature engineering, model training, evaluation, assumption diagnostics, Moving Block Bootstrap, and hypothesis testing) in under 10 seconds and prints pure numerical results to the console.*

### Step 4: Run the Modular Pipeline
```bash
python main.py
```
*Executes the modular pipeline, producing the exact verified numerical comparison tables and statistical verdict.*

### Step 5: Re-Generate All 10 Diagnostic Graphs (Optional)
```bash
python generate_all_10_graphs.py
```
*Saves all 10 high-resolution (300 DPI) diagnostic visualizations to `outputs/graphs/`.*

### Step 6: Re-Generate the 28-Section Academic Word Report (Optional)
```bash
python build_complete_28_section_report.py
```
*Compiles the publication-grade `Sales_Demand_Forecasting_Report.docx` with all 28 required sections, executive metadata, and all 10 figures embedded.*

---

## 6. Streamlit Dashboard Architecture & Page Features

The web dashboard (`app.py`) integrates all analytical components into 6 dedicated pages:

### Page 1: 📊 Executive Overview
- **Executive KPI Cards:** Total historical sales, average daily demand, predicted test demand, leading model (XGBoost), and test MAE (1.8051).
- **Interactive Trajectory:** Plotly time-series comparing historical demand with baseline and improved forecasts.
- **Seasonality Profilers:** Demand distribution across Day of Week and Month.

### Page 2: 📈 Sales Analysis
- **Empirical Demand Dynamics:** Historical sales time-series with a 7-day rolling moving average.
- **Intermittency Diagnostics:** Empirical sales distribution highlighting zero-demand frequency (28.4%) and variance-to-mean overdispersion.
- **Cross-Sectional Breakdowns:** Sales by store (`CA_1` vs `CA_2`) and SKU velocity ranking (top-demand vs low-demand items).
- **Export:** One-click CSV download of filtered historical sales.

### Page 3: 🔮 Demand Forecasting
- **Dynamic Selectors:** Filter by Store, Product SKU, and Forecasting Horizon (7, 14, 28 days).
- **Trajectory Comparison:** Real-time Plotly visualization of ground-truth demand vs. Linear Regression vs. XGBoost predictions, complete with a 95% empirical prediction interval.
- **Series Error Scorecard:** Local MAE, RMSE, and error reduction delta for the selected product series.
- **Export:** One-click CSV download of series forecast predictions.

### Page 4: 🎯 Model Performance
- **Scorecard Table:** Direct display of MAE, RMSE, $R^2$, and sMAPE loaded directly from `outputs/results/model_evaluation_metrics.csv`.
- **Comparative Bar Charts:** Side-by-side metric comparison across Linear Regression, XGBoost, and Random Forest benchmark.
- **Scatter & Residuals:** Actual vs. Predicted scatter plot with unity $y = x$ reference line and residual error distribution ($e = y - \hat{y}$).

### Page 5: 📐 Statistical Analysis
- **Hypothesis Testing Results:** Displays Paired $t$-test ($p = 0.3463$), Wilcoxon ($p = 0.3741$), Diebold-Mariano HAC ($p = 0.3456$), and Moving Block Bootstrap ($p = 0.2270$).
- **Dynamic Verdict Banner:** Automatically evaluates $p$-values and 95% CI to display the verdict: *"Fail to Reject H0 - Improvement not statistically significant."*
- **Interactive Distributions:** Visual Plotly histogram of $B = 2,000$ Moving Block Bootstrap means with the 95% CI spanning across zero, and empirical paired error difference distribution.
- **Audit Section:** Educational callout documenting why the unverified ₹103,160.17 catalog financial figure was audited and purged.

### Page 6: 📦 Inventory Planning
- **Interactive Optimization Inputs:** Store, Product SKU, Lead Time ($L \in [1, 14]$ days), Desired Service Level (90%, 95%, 98%, 99% $\implies Z \in \{1.28, 1.645, 2.05, 2.33\}$), Current On-Hand Inventory, and Unit Selling Price.
- **Statistical Safety Stock & ROP Formulas:**
  - Lead Time Demand: $D_L = \bar{d} \times L$
  - Demand Variance over Lead Time: $\sigma_L = \sigma_d \times \sqrt{L}$
  - Safety Stock: $SS = Z \times \sigma_L$
  - Reorder Point: $ROP = D_L + SS$
  - Recommended Order Quantity: $ROQ = \max(0, ROP - \text{Current Inventory})$
- **Inventory Health Status:** Automatically categorizes inventory as 🔴 *Low (Reorder Now)*, 🟢 *Adequate (Balanced)*, or 🟡 *Overstock (Holding Cost Risk)*.
- **Replenishment Depletion Simulation:** 28-day forward trajectory comparing unreplenished stock depletion against recommended order arrival.
- **Export:** One-click CSV download of the inventory replenishment plan.

---

## 6. Diagnostic Visualizations Catalog (All 10 Graphs)
1. **`01_actual_vs_predicted_sales.png`**: Dual-panel sales trajectory showing aggregate daily test demand across all series (Panel A) and product-level trajectory for `FOODS_1_001` (Panel B).
2. **`02_baseline_vs_improved_performance.png`**: Comparative bar chart of deterministic metrics (MAE, RMSE, $R^2$, and sMAPE) with numerical callouts.
3. **`03_distribution_paired_error_differences.png`**: Histogram and KDE of paired absolute error differences ($d_i = e_{\text{base}} - e_{\text{imp}}$) showing sample mean ($\bar{d} = +0.0107$) and zero line.
4. **`04_bootstrap_confidence_interval_plot.png`**: Empirical sampling distribution from $B = 2,000$ Moving Block Bootstrap iterations illustrating the 95% CI spanning across zero ($[-0.0297, +0.0475]$).
5. **`05_residual_error_comparison.png`**: Residual error tracking over the 28-day test period (Panel A) and residuals vs. fitted sales homoscedasticity plot (Panel B).
6. **`06_qq_plot_normality_diagnostic.png`**: Normal Quantile-Quantile (Q-Q) plot illustrating heavy tails and severe departure from Gaussian normality ($p < 0.001$).
7. **`07_autocorrelation_acf_pacf.png`**: Autocorrelation (ACF) and Partial Autocorrelation (PACF) plots of daily error differences across lags 1 to 14 with 95% Bartlett bounds.
8. **`08_cumulative_absolute_error_difference.png`**: Cumulative daily sum of absolute error differences over time showing trajectory fluctuations around zero.
9. **`09_sku_level_error_reduction_breakdown.png`**: Disaggregated horizontal bar chart showing mean error reduction across all 15 food SKUs for Stores `CA_1` and `CA_2`.
10. **`10_bootstrap_block_size_sensitivity.png`**: Sensitivity analysis of Bootstrap standard error and 95% CI across block sizes $L \in \{1, 3, 7, 14, 21\}$ days.

---

## 7. Structure of the 28-Section Academic Report
The accompanying Word document (`Sales_Demand_Forecasting_Report.docx`, 3.48 MB) contains all 28 distinct required sections:
1. Academic Title Page & Author Metadata
2. Executive Summary & Abstract
3. Introduction & Core Research Question
4. Problem Statement & Industrial Context
5. Benchmark Dataset Overview (Walmart M5 Architecture)
6. Sampling Strategy & Product Selection Rationale
7. Exploratory Data Analysis & Demand Intermittency
8. Data Preprocessing, Cleaning & Deduplication Procedures
9. Temporal Feature Engineering (Calendar, Pricing & Promotions)
10. Autoregressive Lag & Rolling Window Architecture (Leakage Prevention)
11. Experimental Protocol & Chronological Train-Test Split
12. Baseline Model Specification: Standardized Linear Regression
13. Improved Model Specification: Tuned Gradient Boosted Trees (XGBoost)
14. Model Training, Regularization & Hyperparameter Settings
15. Deterministic Evaluation Metrics & Baseline vs. Improved Results *(Embeds Figures 1 & 2)*
16. SKU-Level & Cumulative Error Reduction Diagnostics *(Embeds Figures 3 & 4)*
17. Residual Error Diagnostics & Homoscedasticity Analysis *(Embeds Figure 5)*
18. Theoretical Motivation for Inferential Statistical Testing
19. Mathematical Formulation of Paired Prediction Error Differences
20. Statistical Assumption Testing: Normality Diagnostics *(Embeds Figures 6 & 7)*
21. Serial Autocorrelation Analysis & Independence Diagnostics *(Embeds Figure 8)*
22. Non-Parametric Hypothesis Testing: Wilcoxon Signed-Rank Test
23. Time-Series Autocorrelation-Robust Testing: Diebold-Mariano HAC
24. Resampling Theory & Moving Block Bootstrap ($L = 7$ Days, $B = 2,000$)
25. 95% Bootstrap Confidence Interval Estimation & Sampling Distribution *(Embeds Figure 9)*
26. Moving Block Bootstrap Block Length Sensitivity Analysis *(Embeds Figure 10)*
27. Statistical vs. Practical Significance & Critical Audit of Financial Claims *(Purges ₹103,160.17)*
28. Threats to Validity, Methodological Limitations & Final Verdict

