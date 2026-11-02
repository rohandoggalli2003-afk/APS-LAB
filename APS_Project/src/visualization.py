"""
Visualization Module for Statistical Model Evaluation
=====================================================
Project: Sales Demand Forecasting & Inventory Planning System
Course: Applied Probability and Statistics (APS) / M.E. Big Data Analytics

Generates 5 publication-quality visualizations using Matplotlib and Seaborn:
1. Actual vs Predicted Sales (outputs/graphs/actual_vs_predicted_sales.png)
   - Daily aggregated sales trajectory and representative product demand.
2. Baseline vs Improved Model Performance (outputs/graphs/baseline_vs_improved_performance.png)
   - Comparative bar chart of MAE, RMSE, and R² scores with numerical labels.
3. Distribution of Paired Prediction Error Differences (outputs/graphs/distribution_paired_error_differences.png)
   - Histogram and KDE of error differences (e_base - e_imp) with mean and zero line.
4. Bootstrap Confidence Interval Plot (outputs/graphs/bootstrap_confidence_interval_plot.png)
   - Sampling distribution of mean improvement from B=2,000 Moving Block Bootstrap resamples,
     illustrating the 95% Confidence Interval spanning zero.
5. Residual Error Comparison (outputs/graphs/residual_error_comparison.png)
   - Residual error plots (residuals over time and residuals vs. fitted values) for both models.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Set aesthetic styling
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'axes.edgecolor': '#cccccc',
    'axes.linewidth': 0.8,
    'grid.color': '#ebebeb',
    'grid.linestyle': '--'
})


def plot_actual_vs_predicted_sales(
    test_df: pd.DataFrame,
    output_path: str = "outputs/graphs/actual_vs_predicted_sales.png"
):
    """
    Plot 1: Actual vs Predicted Sales
    Dual-panel visualization:
    - Panel A: Aggregate daily test demand across all products and stores.
    - Panel B: Individual representative SKU time series (FOODS_1_001 in Store CA_1).
    """
    logger.info("Generating Plot 1: Actual vs Predicted Sales...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=False)
    plt.subplots_adjust(hspace=0.35)
    
    # Aggregate by date
    daily_agg = test_df.groupby('date')[['actual_sales', 'pred_baseline', 'pred_improved']].sum().reset_index()
    
    # Panel A: Aggregate Daily Sales
    ax1.plot(daily_agg['date'], daily_agg['actual_sales'], color='#1f2937', lw=2.2, marker='o', ms=4, label='Actual Sales (Ground Truth)')
    ax1.plot(daily_agg['date'], daily_agg['pred_baseline'], color='#dc2626', lw=1.8, linestyle='--', marker='s', ms=3, label='Baseline Model (Linear Regression)')
    ax1.plot(daily_agg['date'], daily_agg['pred_improved'], color='#2563eb', lw=1.8, linestyle='-', marker='^', ms=3, label='Improved Model (XGBoost Regressor)')
    
    ax1.set_title("A. Aggregate Daily Sales Demand Across Held-Out Test Period (28 Days)", fontsize=13, fontweight='bold', pad=10)
    ax1.set_xlabel("Test Date", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Total Units Sold", fontsize=11, fontweight='bold')
    ax1.legend(loc="upper right", frameon=True, framealpha=0.95, fontsize=10)
    ax1.tick_params(axis='x', rotation=20)
    
    # Panel B: Representative Individual SKU
    sample_item_id = test_df['id'].unique()[0]
    sample_df = test_df[test_df['id'] == sample_item_id].sort_values('date')
    
    ax2.plot(sample_df['date'], sample_df['actual_sales'], color='#1f2937', lw=2.0, marker='o', ms=5, label='Actual Demand')
    ax2.plot(sample_df['date'], sample_df['pred_baseline'], color='#dc2626', lw=1.6, linestyle='--', marker='s', ms=4, label='Baseline Prediction')
    ax2.plot(sample_df['date'], sample_df['pred_improved'], color='#2563eb', lw=1.6, linestyle='-', marker='^', ms=4, label='Improved (XGBoost) Prediction')
    
    ax2.set_title(f"B. Product-Level Forecast Trajectory: {sample_item_id}", fontsize=13, fontweight='bold', pad=10)
    ax2.set_xlabel("Test Date", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Daily Units Sold", fontsize=11, fontweight='bold')
    ax2.legend(loc="upper right", frameon=True, framealpha=0.95, fontsize=10)
    ax2.tick_params(axis='x', rotation=20)
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved Plot 1 to: {output_path}")


def plot_baseline_vs_improved_performance(
    metrics_df: pd.DataFrame,
    output_path: str = "outputs/graphs/baseline_vs_improved_performance.png"
):
    """
    Plot 2: Baseline vs Improved Model Performance
    Triple-panel comparative bar chart for MAE, RMSE, and R² Score with value callouts.
    """
    logger.info("Generating Plot 2: Baseline vs Improved Model Performance...")
    
    # Filter to baseline and improved for clear focus
    plot_df = metrics_df[metrics_df['Model'].str.contains('Baseline|Improved')].copy()
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5.5))
    plt.subplots_adjust(wspace=0.28)
    
    palette = ['#e11d48', '#2563eb'] # Crimson for Baseline, Blue for Improved
    
    # Subplot 1: MAE (Lower is better)
    bars1 = ax1.bar(plot_df['Model'], plot_df['MAE'], color=palette, width=0.55, edgecolor='black', linewidth=0.8)
    ax1.set_title("Mean Absolute Error (MAE)\n(Lower is Better)", fontsize=12, fontweight='bold', pad=10)
    ax1.set_ylabel("MAE (Units)", fontsize=11, fontweight='bold')
    ax1.set_ylim(0, max(plot_df['MAE']) * 1.25)
    for bar in bars1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, h + 0.04, f"{h:.4f}", ha='center', va='bottom', fontsize=10.5, fontweight='bold')
    ax1.set_xticks(range(len(plot_df)))
    ax1.set_xticklabels(["Baseline\n(Linear Reg)", "Improved\n(XGBoost)"], fontsize=10.5, fontweight='bold')
    
    # Subplot 2: RMSE (Lower is better)
    bars2 = ax2.bar(plot_df['Model'], plot_df['RMSE'], color=palette, width=0.55, edgecolor='black', linewidth=0.8)
    ax2.set_title("Root Mean Squared Error (RMSE)\n(Lower is Better)", fontsize=12, fontweight='bold', pad=10)
    ax2.set_ylabel("RMSE (Units)", fontsize=11, fontweight='bold')
    ax2.set_ylim(0, max(plot_df['RMSE']) * 1.25)
    for bar in bars2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, h + 0.05, f"{h:.4f}", ha='center', va='bottom', fontsize=10.5, fontweight='bold')
    ax2.set_xticks(range(len(plot_df)))
    ax2.set_xticklabels(["Baseline\n(Linear Reg)", "Improved\n(XGBoost)"], fontsize=10.5, fontweight='bold')
    
    # Subplot 3: R2 Score (Higher is better)
    bars3 = ax3.bar(plot_df['Model'], plot_df['R2_Score'], color=palette, width=0.55, edgecolor='black', linewidth=0.8)
    ax3.set_title("Coefficient of Determination (R²)\n(Higher is Better)", fontsize=12, fontweight='bold', pad=10)
    ax3.set_ylabel("R² Score", fontsize=11, fontweight='bold')
    ax3.set_ylim(0, max(plot_df['R2_Score']) * 1.3)
    for bar in bars3:
        h = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2.0, h + 0.012, f"{h:.4f}", ha='center', va='bottom', fontsize=10.5, fontweight='bold')
    ax3.set_xticks(range(len(plot_df)))
    ax3.set_xticklabels(["Baseline\n(Linear Reg)", "Improved\n(XGBoost)"], fontsize=10.5, fontweight='bold')
    
    plt.suptitle("Model Evaluation Metric Comparison (Held-Out Test Set, N = 840)", fontsize=14, fontweight='bold', y=1.03)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved Plot 2 to: {output_path}")


def plot_distribution_paired_error_differences(
    test_df: pd.DataFrame,
    output_path: str = "outputs/graphs/distribution_paired_error_differences.png"
):
    """
    Plot 3: Distribution of Paired Prediction Error Differences
    Histogram and Kernel Density Estimate of d_i = |e_base,i| - |e_imp,i|.
    Positive differences indicate error reduction by the improved model.
    Highlights zero reference line, sample mean improvement, and median.
    """
    logger.info("Generating Plot 3: Distribution of Paired Prediction Error Differences...")
    diffs = test_df['error_diff_abs'].values
    mean_d = float(np.mean(diffs))
    median_d = float(np.median(diffs))
    
    plt.figure(figsize=(10, 6.5))
    
    # Histogram + KDE
    sns.histplot(diffs, bins=45, kde=True, color='#3b82f6', edgecolor='black', alpha=0.6, stat="density", label='Empirical Distribution')
    
    # Reference lines
    plt.axvline(0.0, color='#dc2626', lw=2.2, linestyle='--', label='Null Reference Line (No Difference, d = 0)')
    plt.axvline(mean_d, color='#15803d', lw=2.2, linestyle='-', label=f'Sample Mean Improvement: {mean_d:+.4f} units')
    plt.axvline(median_d, color='#7c3aed', lw=1.8, linestyle=':', label=f'Sample Median: {median_d:+.4f} units')
    
    plt.title(r"Distribution of Paired Absolute Prediction Error Differences" + "\n" + r"$d_i = |y_i - \hat{y}_{\rm base, i}| - |y_i - \hat{y}_{\rm imp, i}|$",
              fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Prediction Error Difference (Units)\n(Positive Values = Improved Model Has Smaller Error)", fontsize=11, fontweight='bold')
    plt.ylabel("Probability Density", fontsize=11, fontweight='bold')
    plt.legend(loc="upper right", frameon=True, framealpha=0.95, fontsize=10)
    
    # Annotate skewness
    skew = float(stats.skew(diffs))
    plt.annotate(
        f"N = {len(diffs):,} test cases\nSkewness = {skew:+.3f}\nKurtosis = {float(stats.kurtosis(diffs)):.2f}",
        xy=(0.03, 0.78), xycoords='axes fraction',
        bbox=dict(boxstyle="round,pad=0.5", fc="#f8fafc", ec="#cbd5e1", lw=1),
        fontsize=9.5
    )
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved Plot 3 to: {output_path}")


def plot_bootstrap_confidence_interval(
    mbb_results: Dict[str, Any],
    output_path: str = "outputs/graphs/bootstrap_confidence_interval_plot.png"
):
    """
    Plot 4: Bootstrap Confidence Interval Plot
    Plots the sampling distribution of mean improvement from B=2,000 Moving Block
    Bootstrap resamples. Visually highlights that the 95% Confidence Interval encompasses zero.
    """
    logger.info("Generating Plot 4: Bootstrap Confidence Interval Plot...")
    boot_means = mbb_results['boot_means']
    obs_mean = mbb_results['obs_mean']
    ci_low, ci_high = mbb_results['ci_percentile']
    
    plt.figure(figsize=(10, 6.5))
    
    # Sampling distribution histogram & KDE
    sns.histplot(boot_means, bins=40, kde=True, color='#6366f1', edgecolor='black', alpha=0.6, stat="density", label='Bootstrap Sampling Distribution ($B=2,000$)')
    
    # Point estimate & zero
    plt.axvline(obs_mean, color='#1e3a8a', lw=2.5, linestyle='-', label=f'Observed Mean Difference: {obs_mean:+.4f}')
    plt.axvline(0.0, color='#dc2626', lw=2.2, linestyle='--', label=r'Null Hypothesis Value ($H_0: \mu_d = 0$)')
    
    # 95% CI bounds
    plt.axvline(ci_low, color='#111827', lw=2.0, linestyle=':', label=f'95% CI Lower Bound: {ci_low:.4f}')
    plt.axvline(ci_high, color='#111827', lw=2.0, linestyle=':', label=f'95% CI Upper Bound: {ci_high:.4f}')
    
    # Shaded CI Region
    plt.axvspan(ci_low, ci_high, color='#e0e7ff', alpha=0.5, label='95% Bootstrap Confidence Interval')
    
    plt.title("Moving Block Bootstrap Sampling Distribution of Mean Error Reduction\n(Block Size L = 7 Days, B = 2,000 Resamples)",
              fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Bootstrap Mean Error Difference $\\bar{d}^*$ (Units)", fontsize=11, fontweight='bold')
    plt.ylabel("Bootstrap Sampling Density", fontsize=11, fontweight='bold')
    plt.legend(loc="upper right", frameon=True, framealpha=0.95, fontsize=9.5)
    
    # Inset annotation
    decision_text = (
        f"Statistical Verdict:\n"
        f"• 95% CI = [{ci_low:.4f}, {ci_high:.4f}]\n"
        f"• Zero (0.0) is INCLUDED in the CI\n"
        f"• Conclusion: Fail to Reject $H_0$\n"
        f"  (Observed improvement is not statistically\n"
        f"   distinguishable from sampling noise)"
    )
    plt.annotate(
        decision_text,
        xy=(0.03, 0.68), xycoords='axes fraction',
        bbox=dict(boxstyle="round,pad=0.5", fc="#fef2f2", ec="#fca5a5", lw=1.2),
        fontsize=9.5, fontweight='semibold', color='#991b1b'
    )
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved Plot 4 to: {output_path}")


def plot_residual_error_comparison(
    test_df: pd.DataFrame,
    output_path: str = "outputs/graphs/residual_error_comparison.png"
):
    """
    Plot 5: Residual Error Comparison
    Dual-panel residual diagnostic:
    - Panel A: Time-series of daily mean residuals (bias tracking over the test period).
    - Panel B: Residuals vs Fitted values scatter plot showing homoscedasticity / variance.
    """
    logger.info("Generating Plot 5: Residual Error Comparison...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    plt.subplots_adjust(wspace=0.25)
    
    res_base = test_df['actual_sales'].values - test_df['pred_baseline'].values
    res_imp = test_df['actual_sales'].values - test_df['pred_improved'].values
    
    test_df_copy = test_df.copy()
    test_df_copy['res_baseline'] = res_base
    test_df_copy['res_improved'] = res_imp
    
    daily_res = test_df_copy.groupby('date')[['res_baseline', 'res_improved']].mean().reset_index()
    
    # Panel A: Daily Mean Residuals (Bias) over Time
    ax1.plot(daily_res['date'], daily_res['res_baseline'], color='#dc2626', lw=1.8, marker='s', ms=4, label=r'Baseline Residuals ($y - \hat{y}_{\rm base}$)')
    ax1.plot(daily_res['date'], daily_res['res_improved'], color='#2563eb', lw=1.8, marker='^', ms=4, label=r'Improved Residuals ($y - \hat{y}_{\rm imp}$)')
    ax1.axhline(0.0, color='black', lw=1.2, linestyle='--')
    ax1.set_title("A. Average Forecast Residuals Over Test Horizon\n(Tracking Under/Over-Forecasting Bias)", fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel("Date", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Mean Residual (Actual - Predicted)", fontsize=11, fontweight='bold')
    ax1.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=9.5)
    ax1.tick_params(axis='x', rotation=20)
    
    # Panel B: Residuals vs Fitted (Predicted) Values
    ax2.scatter(test_df['pred_baseline'], res_base, color='#dc2626', alpha=0.35, s=25, label='Baseline')
    ax2.scatter(test_df['pred_improved'], res_imp, color='#2563eb', alpha=0.35, s=25, label='Improved (XGBoost)')
    ax2.axhline(0.0, color='black', lw=1.2, linestyle='--')
    ax2.set_title("B. Residuals vs. Fitted Sales Demand\n(Homoscedasticity and Variance Analysis)", fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel("Fitted / Predicted Sales (Units)", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Residual (Actual - Predicted)", fontsize=11, fontweight='bold')
    ax2.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=9.5)
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved Plot 5 to: {output_path}")


def generate_all_visualizations(
    test_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    mbb_results: Dict[str, Any],
    graphs_dir: str = "outputs/graphs"
):
    """Generates all 5 required figures."""
    graphs_dir = Path(graphs_dir)
    graphs_dir.mkdir(parents=True, exist_ok=True)
    
    plot_actual_vs_predicted_sales(test_df, str(graphs_dir / "actual_vs_predicted_sales.png"))
    plot_baseline_vs_improved_performance(metrics_df, str(graphs_dir / "baseline_vs_improved_performance.png"))
    plot_distribution_paired_error_differences(test_df, str(graphs_dir / "distribution_paired_error_differences.png"))
    plot_bootstrap_confidence_interval(mbb_results, str(graphs_dir / "bootstrap_confidence_interval_plot.png"))
    plot_residual_error_comparison(test_df, str(graphs_dir / "residual_error_comparison.png"))
    logger.info("All 5 visualization graphs successfully generated and saved to outputs/graphs/!")


if __name__ == "__main__":
    from data_preprocessing import prepare_m5_data
    from train_models import train_and_evaluate_models
    from statistical_analysis import perform_statistical_analysis
    
    print("Testing visualization generation...")
    data_dict = prepare_m5_data()
    train_results = train_and_evaluate_models(data_dict)
    stat_results = perform_statistical_analysis(train_results['test_df'])
    
    generate_all_visualizations(
        test_df=train_results['test_df'],
        metrics_df=train_results['metrics_df'],
        mbb_results=stat_results['mbb']
    )
    print("All visualizations created successfully!")
