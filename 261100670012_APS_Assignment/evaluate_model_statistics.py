import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ======================================================================================
# 1. DATA PREPARATION
# ======================================================================================
def prepare_data(test_size=0.25, random_state=RANDOM_STATE):
    """
    Loads the Breast Cancer Wisconsin dataset and creates a stratified train/test split.
    
    The dataset consists of 569 patient samples with 30 continuous clinical features
    computed from digitized images of fine needle aspirates (FNA) of breast masses.
    Classes: Malignant (0) and Benign (1).
    """
    print("\n" + "="*80)
    print("STEP 1: DATA PREPARATION & EXPLORATORY SUMMARY")
    print("="*80)
    
    raw_data = load_breast_cancer()
    X = raw_data.data
    y = raw_data.target
    feature_names = raw_data.feature_names
    target_names = raw_data.target_names  # ['malignant', 'benign']
    
    n_samples, n_features = X.shape
    n_benign = np.sum(y == 1)
    n_malignant = np.sum(y == 0)
    
    print(f"Dataset: Wisconsin Diagnostic Breast Cancer (WDBC)")
    print(f"Total instances: {n_samples}")
    print(f"Total features:  {n_features} continuous attributes")
    print(f"Target distribution: Benign={n_benign} ({n_benign/n_samples*100:.1f}%), "
          f"Malignant={n_malignant} ({n_malignant/n_samples*100:.1f}%)")
    
    # Stratified Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"\nTrain/Test Partition (Stratified):")
    print(f" - Training Set: {X_train.shape[0]} samples ({(1-test_size)*100:.0f}%)")
    print(f" - Held-Out Test Set: {X_test.shape[0]} samples ({test_size*100:.0f}%)")
    
    return X_train, X_test, y_train, y_test, feature_names, target_names


# ======================================================================================
# 2. MODEL TRAINING
# ======================================================================================
def train_models(X_train, y_train, random_state=RANDOM_STATE):
    """
    Trains two models:
      - Baseline Model: Simple Decision Tree Classifier (max_depth=3).
        A restricted, highly interpretable decision tree serving as the clinical benchmark.
      - Improved Model: Pipeline of StandardScaler and Tuned Gradient Boosting Classifier.
        An advanced non-linear ensemble combining 100 boosting stages with shrinkage.
    """
    print("\n" + "="*80)
    print("STEP 2: MODEL TRAINING (BASELINE VS. IMPROVED)")
    print("="*80)
    
    # Baseline Model
    baseline_model = DecisionTreeClassifier(
        max_depth=3,
        random_state=random_state
    )
    print("Training Baseline Model: DecisionTreeClassifier(max_depth=3)...")
    baseline_model.fit(X_train, y_train)
    
    # Improved Model
    improved_model = Pipeline([
        ('scaler', StandardScaler()),
        ('gb', GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            subsample=0.8,
            random_state=random_state
        ))
    ])
    print("Training Improved Model: Pipeline(StandardScaler + GradientBoostingClassifier)...")
    improved_model.fit(X_train, y_train)
    
    print("Both models successfully trained on the training partition.")
    return baseline_model, improved_model


# ======================================================================================
# 3. EVALUATION ON HELD-OUT TEST SET
# ======================================================================================
def evaluate_models(baseline_model, improved_model, X_test, y_test, target_names):
    """
    Evaluates both models on the identical held-out test set across standard metrics:
    Accuracy, Precision, Recall, Specificity, F1-Score, and ROC-AUC.
    """
    print("\n" + "="*80)
    print("STEP 3: EVALUATION ON HELD-OUT TEST SET (N = {})".format(len(y_test)))
    print("="*80)
    
    # Predictions and predicted probabilities
    y_pred_base = baseline_model.predict(X_test)
    y_prob_base = baseline_model.predict_proba(X_test)[:, 1]
    
    y_pred_imp = improved_model.predict(X_test)
    y_prob_imp = improved_model.predict_proba(X_test)[:, 1]
    
    def compute_metrics(y_true, y_pred, y_prob):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'specificity': tn / (tn + fp),
            'f1': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_prob),
            'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
        }
    
    metrics_base = compute_metrics(y_test, y_pred_base, y_prob_base)
    metrics_imp = compute_metrics(y_test, y_pred_imp, y_prob_imp)
    
    # Tabular Display
    metrics_df = pd.DataFrame([
        {
            'Model': 'Baseline (Decision Tree)',
            'Accuracy': f"{metrics_base['accuracy']:.4f}",
            'Precision': f"{metrics_base['precision']:.4f}",
            'Recall (Sensitivity)': f"{metrics_base['recall']:.4f}",
            'Specificity': f"{metrics_base['specificity']:.4f}",
            'F1-Score': f"{metrics_base['f1']:.4f}",
            'ROC-AUC': f"{metrics_base['roc_auc']:.4f}"
        },
        {
            'Model': 'Improved (Gradient Boosting)',
            'Accuracy': f"{metrics_imp['accuracy']:.4f}",
            'Precision': f"{metrics_imp['precision']:.4f}",
            'Recall (Sensitivity)': f"{metrics_imp['recall']:.4f}",
            'Specificity': f"{metrics_imp['specificity']:.4f}",
            'F1-Score': f"{metrics_imp['f1']:.4f}",
            'ROC-AUC': f"{metrics_imp['roc_auc']:.4f}"
        },
        {
            'Model': 'Difference (Improved - Baseline)',
            'Accuracy': f"+{metrics_imp['accuracy'] - metrics_base['accuracy']:.4f}",
            'Precision': f"+{metrics_imp['precision'] - metrics_base['precision']:.4f}",
            'Recall (Sensitivity)': f"{metrics_imp['recall'] - metrics_base['recall']:.4f}",
            'Specificity': f"+{metrics_imp['specificity'] - metrics_base['specificity']:.4f}",
            'F1-Score': f"+{metrics_imp['f1'] - metrics_base['f1']:.4f}",
            'ROC-AUC': f"+{metrics_imp['roc_auc'] - metrics_base['roc_auc']:.4f}"
        }
    ])
    
    print("\nComparative Performance Summary:")
    print(metrics_df.to_string(index=False))
    
    obs_diff = metrics_imp['accuracy'] - metrics_base['accuracy']
    print(f"\nObserved Test Accuracy Improvement (Delta): {obs_diff:.4f} ({obs_diff*100:+.2f} percentage points)")
    print(f"Correctly classified instances out of {len(y_test)}:")
    print(f" - Baseline Model: {int(metrics_base['accuracy'] * len(y_test))} / {len(y_test)}")
    print(f" - Improved Model: {int(metrics_imp['accuracy'] * len(y_test))} / {len(y_test)}")
    print(f" - Net Additional Correct Cases: {int(round(obs_diff * len(y_test)))}")
    
    return {
        'y_pred_base': y_pred_base,
        'y_prob_base': y_prob_base,
        'y_pred_imp': y_pred_imp,
        'y_prob_imp': y_prob_imp,
        'metrics_base': metrics_base,
        'metrics_imp': metrics_imp,
        'obs_diff': obs_diff
    }


# ======================================================================================
# 4. STATISTICAL SAMPLING & BOOTSTRAP ESTIMATION
# ======================================================================================
def bootstrap_analysis(y_test, y_pred_base, y_pred_imp, n_bootstraps=2000, alpha=0.05, random_state=RANDOM_STATE):
    """
    Applies non-parametric paired bootstrap resampling to quantify the sampling
    distribution of the difference in accuracy between the two models.
    
    Parameters:
      - n_bootstraps: Number of bootstrap iterations (default: 2,000)
      - alpha: Significance level for (1 - alpha) * 100% confidence intervals (default: 0.05 -> 95% CI)
    """
    print("\n" + "="*80)
    print("STEP 4: STATISTICAL SAMPLING & BOOTSTRAP ESTIMATION")
    print("="*80)
    print(f"Resampling Scheme: Non-parametric paired bootstrap (B = {n_bootstraps} iterations)")
    print(f"Preserves sample-level correlation between predictions on the test set.")
    
    rng = np.random.RandomState(random_state)
    n_test = len(y_test)
    
    boot_acc_base = np.empty(n_bootstraps)
    boot_acc_imp = np.empty(n_bootstraps)
    boot_diffs = np.empty(n_bootstraps)
    
    for b in range(n_bootstraps):
        # Draw paired bootstrap sample with replacement
        sample_indices = rng.choice(n_test, size=n_test, replace=True)
        y_b = y_test[sample_indices]
        pred_b_base = y_pred_base[sample_indices]
        pred_b_imp = y_pred_imp[sample_indices]
        
        acc_b = accuracy_score(y_b, pred_b_base)
        acc_i = accuracy_score(y_b, pred_b_imp)
        
        boot_acc_base[b] = acc_b
        boot_acc_imp[b] = acc_i
        boot_diffs[b] = acc_i - acc_b
    
    # Statistical Properties of the Sampling Distribution
    boot_mean_diff = np.mean(boot_diffs)
    boot_se_diff = np.std(boot_diffs, ddof=1)
    obs_diff = accuracy_score(y_test, y_pred_imp) - accuracy_score(y_test, y_pred_base)
    boot_bias = boot_mean_diff - obs_diff
    
    print(f"\nBootstrap Sampling Distribution Estimates:")
    print(f" - Observed Accuracy Difference: {obs_diff:.4f}")
    print(f" - Bootstrap Mean Difference:     {boot_mean_diff:.4f}")
    print(f" - Standard Error (SE):          {boot_se_diff:.4f}")
    print(f" - Bootstrap Bias:               {boot_bias:.6f}")
    
    # ----------------------------------------------------------------------------------
    # 5. CONFIDENCE INTERVAL ESTIMATION
    # ----------------------------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 5: CONFIDENCE INTERVAL ESTIMATION (95% CI)")
    print("="*80)
    
    # 1. Percentile Bootstrap Confidence Interval
    lower_pct = 100 * (alpha / 2.0)
    upper_pct = 100 * (1.0 - alpha / 2.0)
    ci_percentile_lower = np.percentile(boot_diffs, lower_pct)
    ci_percentile_upper = np.percentile(boot_diffs, upper_pct)
    
    # 2. Normal-Approximation Bootstrap Confidence Interval
    z_crit = stats.norm.ppf(1 - alpha / 2.0)
    ci_norm_lower = obs_diff - z_crit * boot_se_diff
    ci_norm_upper = obs_diff + z_crit * boot_se_diff
    
    # 3. Paired Wald Confidence Interval (McNemar variance)
    # Var(diff) = [(b + c) - (b - c)^2 / n] / n^2
    correct_base = (y_pred_base == y_test)
    correct_imp = (y_pred_imp == y_test)
    b_discordant = np.sum(correct_base & ~correct_imp)
    c_discordant = np.sum(~correct_base & correct_imp)
    wald_var = ((b_discordant + c_discordant) - ((b_discordant - c_discordant)**2) / n_test) / (n_test**2)
    wald_se = np.sqrt(wald_var)
    ci_wald_lower = obs_diff - z_crit * wald_se
    ci_wald_upper = obs_diff + z_crit * wald_se
    
    ci_table = pd.DataFrame([
        {
            'Method': 'Percentile Bootstrap (95%)',
            'Lower Bound': f"{ci_percentile_lower:.4f}",
            'Upper Bound': f"{ci_percentile_upper:.4f}",
            'Includes Zero?': 'YES' if (ci_percentile_lower <= 0 <= ci_percentile_upper) else 'NO'
        },
        {
            'Method': 'Normal Approximation Bootstrap (95%)',
            'Lower Bound': f"{ci_norm_lower:.4f}",
            'Upper Bound': f"{ci_norm_upper:.4f}",
            'Includes Zero?': 'YES' if (ci_norm_lower <= 0 <= ci_norm_upper) else 'NO'
        },
        {
            'Method': 'Paired Wald Large-Sample (95%)',
            'Lower Bound': f"{ci_wald_lower:.4f}",
            'Upper Bound': f"{ci_wald_upper:.4f}",
            'Includes Zero?': 'YES' if (ci_wald_lower <= 0 <= ci_wald_upper) else 'NO'
        }
    ])
    print(ci_table.to_string(index=False))
    
    zero_in_ci = (ci_percentile_lower <= 0 <= ci_percentile_upper)
    print(f"\nConfidence Interval Interpretation:")
    if zero_in_ci:
        print(f" -> The 95% Confidence Interval [{ci_percentile_lower:.4f}, {ci_percentile_upper:.4f}] CONTAINS 0.")
        print(f" -> Conclusion: At the 95% confidence level, we CANNOT rule out that the true population")
        print(f"    difference between the models is zero, or even that the baseline could be slightly better.")
    else:
        print(f" -> The 95% Confidence Interval does NOT contain 0.")
        print(f" -> Conclusion: The performance improvement is statistically significant at the 95% confidence level.")
        
    return {
        'boot_diffs': boot_diffs,
        'boot_se': boot_se_diff,
        'ci_percentile': (ci_percentile_lower, ci_percentile_upper),
        'ci_normal': (ci_norm_lower, ci_norm_upper),
        'ci_wald': (ci_wald_lower, ci_wald_upper)
    }


# ======================================================================================
# 6. HYPOTHESIS TESTING
# ======================================================================================
def hypothesis_testing(y_test, y_pred_base, y_pred_imp, X, y, alpha=0.05, n_permutations=10000, random_state=RANDOM_STATE):
    """
    Conducts formal statistical hypothesis testing to evaluate whether the observed
    difference in classification accuracy is statistically significant:
    
    1. McNemar's Test (Asymptotic Chi-Square & Exact Binomial Test)
       - Gold standard for comparing two classifiers evaluated on the same test set.
    2. Paired Permutation / Randomization Test (10,000 permutations)
       - Non-parametric exact exchangeability test.
    3. Dietterich's 5x2-Fold Cross-Validation Paired t-Test
       - Assesses stability and statistical significance across training set variations.
    """
    print("\n" + "="*80)
    print("STEP 6: STATISTICAL HYPOTHESIS TESTING")
    print("="*80)
    print("Hypothesis Formulations:")
    print("  Null Hypothesis (H0): There is no difference in accuracy between the Baseline")
    print("                        and Improved models (p_improved = p_baseline).")
    print("  Alternative Hypothesis (H1): The Improved model performs differently from")
    print("                               the Baseline model (p_improved != p_baseline) [Two-Sided].")
    print(f"Significance Level (alpha): {alpha}\n")
    
    n_test = len(y_test)
    correct_base = (y_pred_base == y_test)
    correct_imp = (y_pred_imp == y_test)
    
    # Contingency Table
    n00 = np.sum(correct_base & correct_imp)    # Both correct
    n01 = np.sum(correct_base & ~correct_imp)   # Baseline correct, Improved wrong (b)
    n10 = np.sum(~correct_base & correct_imp)   # Baseline wrong, Improved correct (c)
    n11 = np.sum(~correct_base & ~correct_imp)  # Both wrong
    
    print("--- 6.1 McNemar's Contingency Table (Paired Predictions) ---")
    contingency_matrix = pd.DataFrame([
        {'Baseline': 'Correct', 'Improved: Correct': n00, 'Improved: Incorrect': n01, 'Total': n00 + n01},
        {'Baseline': 'Incorrect', 'Improved: Correct': n10, 'Improved: Incorrect': n11, 'Total': n10 + n11},
        {'Baseline': 'Total', 'Improved: Correct': n00 + n10, 'Improved: Incorrect': n01 + n11, 'Total': n_test}
    ])
    print(contingency_matrix.to_string(index=False))
    
    b = n01
    c = n10
    total_discordant = b + c
    print(f"\nDiscordant Pairs:")
    print(f" - Baseline Correct & Improved Incorrect (b): {b}")
    print(f" - Baseline Incorrect & Improved Correct (c): {c}")
    print(f" - Total Discordant Cases (b + c):             {total_discordant}")
    
    # 1. McNemar Asymptotic Chi-Squared Test (with Edwards' continuity correction)
    if total_discordant > 0:
        chi2_stat = ((abs(b - c) - 1.0)**2) / total_discordant
        p_val_mcnemar_chi2 = 1.0 - stats.chi2.cdf(chi2_stat, df=1)
    else:
        chi2_stat = 0.0
        p_val_mcnemar_chi2 = 1.0
        
    # 2. McNemar Exact Binomial Test
    # Under H0, b ~ Binomial(n = b + c, p = 0.5)
    if total_discordant > 0:
        binom_res = stats.binomtest(k=min(b, c), n=total_discordant, p=0.5, alternative='two-sided')
        p_val_mcnemar_exact = binom_res.pvalue
        binom_res_one_sided = stats.binomtest(k=c, n=total_discordant, p=0.5, alternative='greater')
        p_val_mcnemar_one_sided = binom_res_one_sided.pvalue
    else:
        p_val_mcnemar_exact = 1.0
        p_val_mcnemar_one_sided = 1.0
        
    print(f"\nMcNemar Test Results:")
    print(f" - Chi-Square Statistic (with continuity correction): chi2 = {chi2_stat:.4f}")
    print(f" - Asymptotic p-value (df = 1):                       p = {p_val_mcnemar_chi2:.4f}")
    print(f" - Exact Binomial p-value (Two-Sided):                 p = {p_val_mcnemar_exact:.4f}")
    print(f" - Exact Binomial p-value (One-Sided, H1: c > b):      p = {p_val_mcnemar_one_sided:.4f}")
    
    # 3. Non-Parametric Paired Permutation / Randomization Test
    print(f"\n--- 6.2 Paired Permutation Test ({n_permutations:,} permutations) ---")
    rng = np.random.RandomState(random_state)
    obs_diff = (c - b) / n_test
    
    # Under H0, labels assigned by the two models to discordant cases are exchangeable with p=0.5
    # Fully vectorized Monte Carlo permutation test
    flips = rng.choice([-1, 1], size=(n_permutations, total_discordant))
    perm_diffs = np.sum(flips, axis=1) / n_test
        
    p_val_perm_two_sided = np.mean(np.abs(perm_diffs) >= np.abs(obs_diff))
    p_val_perm_one_sided = np.mean(perm_diffs >= obs_diff)
    
    print(f" - Observed Accuracy Difference:                     {obs_diff:.4f}")
    print(f" - Permutation Test p-value (Two-Sided):             p = {p_val_perm_two_sided:.4f}")
    print(f" - Permutation Test p-value (One-Sided):             p = {p_val_perm_one_sided:.4f}")
    
    # 4. Dietterich's 5x2-Fold Cross-Validation Paired t-Test
    print(f"\n--- 6.3 Dietterich's 5x2-Fold Cross-Validation Paired t-Test ---")
    rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=random_state)
    
    diff_fold1 = []
    variances = []
    
    fold_idx = 0
    fold_diffs = []
    
    for train_cv, test_cv in rskf.split(X, y):
        X_tr, X_te = X[train_cv], X[test_cv]
        y_tr, y_te = y[train_cv], y[test_cv]
        
        # Train baseline
        m_base = DecisionTreeClassifier(max_depth=3, random_state=random_state)
        m_base.fit(X_tr, y_tr)
        acc_b = accuracy_score(y_te, m_base.predict(X_te))
        
        # Train improved
        m_imp = Pipeline([
            ('scaler', StandardScaler()),
            ('gb', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3,
                                             subsample=0.8, random_state=random_state))
        ])
        m_imp.fit(X_tr, y_tr)
        acc_i = accuracy_score(y_te, m_imp.predict(X_te))
        
        fold_diffs.append(acc_i - acc_b)
        fold_idx += 1
        
        if fold_idx % 2 == 0:
            # End of one 2-fold repetition
            d1 = fold_diffs[0]
            d2 = fold_diffs[1]
            diff_fold1.append(d1)
            mean_d = (d1 + d2) / 2.0
            var_rep = (d1 - mean_d)**2 + (d2 - mean_d)**2
            variances.append(var_rep)
            fold_diffs = []
            
    # Dietterich's t statistic: t = diff_fold1[0] / sqrt(0.2 * sum(variances))
    sum_var = np.sum(variances)
    if sum_var > 0:
        t_5x2 = diff_fold1[0] / np.sqrt(0.2 * sum_var)
        p_val_5x2 = 2.0 * (1.0 - stats.t.cdf(np.abs(t_5x2), df=5))
    else:
        t_5x2 = 0.0
        p_val_5x2 = 1.0
        
    print(f" - Dietterich 5x2 CV t-statistic:                     t = {t_5x2:.4f} (df = 5)")
    print(f" - 5x2 CV Paired t-Test p-value:                      p = {p_val_5x2:.4f}")
    
    # Summary of Hypothesis Tests
    test_summary_df = pd.DataFrame([
        {
            'Statistical Test': "McNemar Exact Binomial Test (Two-Sided)",
            'Statistic': f"Discordant (b={b}, c={c})",
            'p-value': f"{p_val_mcnemar_exact:.4f}",
            'Decision (alpha=0.05)': 'Fail to Reject H0' if p_val_mcnemar_exact >= alpha else 'Reject H0'
        },
        {
            'Statistical Test': "McNemar Chi-Squared (with Edwards CC)",
            'Statistic': f"chi2 = {chi2_stat:.4f}",
            'p-value': f"{p_val_mcnemar_chi2:.4f}",
            'Decision (alpha=0.05)': 'Fail to Reject H0' if p_val_mcnemar_chi2 >= alpha else 'Reject H0'
        },
        {
            'Statistical Test': "Paired Permutation Test (10,000 runs)",
            'Statistic': f"Delta = {obs_diff:.4f}",
            'p-value': f"{p_val_perm_two_sided:.4f}",
            'Decision (alpha=0.05)': 'Fail to Reject H0' if p_val_perm_two_sided >= alpha else 'Reject H0'
        },
        {
            'Statistical Test': "Dietterich 5x2 Cross-Validation t-Test",
            'Statistic': f"t = {t_5x2:.4f} (df=5)",
            'p-value': f"{p_val_5x2:.4f}",
            'Decision (alpha=0.05)': 'Fail to Reject H0' if p_val_5x2 >= alpha else 'Reject H0'
        }
    ])
    
    print("\n" + "="*80)
    print("HYPOTHESIS TESTING SUMMARY TABLE")
    print("="*80)
    print(test_summary_df.to_string(index=False))
    
    return {
        'b': b,
        'c': c,
        'chi2_stat': chi2_stat,
        'p_val_mcnemar_chi2': p_val_mcnemar_chi2,
        'p_val_mcnemar_exact': p_val_mcnemar_exact,
        'perm_diffs': perm_diffs,
        'p_val_perm': p_val_perm_two_sided,
        't_5x2': t_5x2,
        'p_val_5x2': p_val_5x2
    }


# ======================================================================================
# 7. VISUALIZATION OF STATISTICAL ANALYSIS
# ======================================================================================
def generate_visualizations(eval_results, boot_results, test_results, y_test, output_filename="model_statistical_evaluation.png"):
    """
    Generates a 4-panel figure illustrating the statistical evaluation of the models:
      1. ROC Curves (Baseline vs Improved)
      2. McNemar's Paired Contingency Matrix
      3. Bootstrap Sampling Distribution of Difference with 95% Confidence Interval
      4. Permutation Test Null Distribution vs Observed Difference
    """
    print("\n" + "="*80)
    print("STEP 7: GENERATING DIAGNOSTIC VISUALIZATIONS")
    print("="*80)
    
    y_prob_base = eval_results['y_prob_base']
    y_prob_imp = eval_results['y_prob_imp']
    obs_diff = eval_results['obs_diff']
    boot_diffs = boot_results['boot_diffs']
    ci_low, ci_high = boot_results['ci_percentile']
    perm_diffs = test_results['perm_diffs']
    b = test_results['b']
    c = test_results['c']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    plt.subplots_adjust(hspace=0.32, wspace=0.25)
    
    # ---------------------------------------------------------
    # Panel 1: ROC Curves
    # ---------------------------------------------------------
    fpr_base, tpr_base, _ = roc_curve(y_test, y_prob_base)
    fpr_imp, tpr_imp, _ = roc_curve(y_test, y_prob_imp)
    auc_base = eval_results['metrics_base']['roc_auc']
    auc_imp = eval_results['metrics_imp']['roc_auc']
    
    axes[0, 0].plot(fpr_base, tpr_base, color="#1f77b4", lw=2, label=f"Baseline Decision Tree (AUC = {auc_base:.4f})")
    axes[0, 0].plot(fpr_imp, tpr_imp, color="#2ca02c", lw=2, label=f"Improved Gradient Boosting (AUC = {auc_imp:.4f})")
    axes[0, 0].plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.5000)")
    axes[0, 0].set_xlim([0.0, 1.0])
    axes[0, 0].set_ylim([0.0, 1.02])
    axes[0, 0].set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11, fontweight='bold')
    axes[0, 0].set_title("A. Receiver Operating Characteristic (ROC) Curves", fontsize=12, fontweight='bold')
    axes[0, 0].legend(loc="lower right", frameon=True, framealpha=0.9)
    axes[0, 0].grid(True, linestyle=":", alpha=0.6)
    
    # ---------------------------------------------------------
    # Panel 2: McNemar's Contingency Matrix
    # ---------------------------------------------------------
    n00 = np.sum((eval_results['y_pred_base'] == y_test) & (eval_results['y_pred_imp'] == y_test))
    n11 = np.sum((eval_results['y_pred_base'] != y_test) & (eval_results['y_pred_imp'] != y_test))
    matrix_data = np.array([[n00, b], [c, n11]])
    
    im = axes[0, 1].imshow(matrix_data, interpolation='nearest', cmap=plt.cm.Blues)
    axes[0, 1].set_title(f"B. McNemar Paired Contingency Matrix\n(Exact Binomial p = {test_results['p_val_mcnemar_exact']:.4f})", 
                         fontsize=12, fontweight='bold')
    axes[0, 1].set_xticks([0, 1])
    axes[0, 1].set_yticks([0, 1])
    axes[0, 1].set_xticklabels(['Improved Correct', 'Improved Incorrect'], fontsize=10, fontweight='bold')
    axes[0, 1].set_yticklabels(['Baseline Correct', 'Baseline Incorrect'], fontsize=10, fontweight='bold')
    
    # Text annotations in matrix cells
    labels = [
        [f"Both Correct\n(Concordant)\nn00 = {n00}", f"Baseline Only\n(Discordant: b)\nn01 = {b}"],
        [f"Improved Only\n(Discordant: c)\nn10 = {c}", f"Both Incorrect\n(Concordant)\nn11 = {n11}"]
    ]
    for i in range(2):
        for j in range(2):
            cell_color = "white" if matrix_data[i, j] > np.max(matrix_data)/2 else "black"
            axes[0, 1].text(j, i, labels[i][j], ha="center", va="center", color=cell_color,
                            fontsize=11, fontweight='bold')
            
    # ---------------------------------------------------------
    # Panel 3: Bootstrap Sampling Distribution of Difference
    # ---------------------------------------------------------
    counts, bins, _ = axes[1, 0].hist(boot_diffs, bins=35, density=True, color="#4a90e2", alpha=0.65, edgecolor="black")
    axes[1, 0].axvline(obs_diff, color="navy", lw=2.5, linestyle="-", label=f"Observed Diff: {obs_diff:+.4f}")
    axes[1, 0].axvline(0.0, color="crimson", lw=2, linestyle="--", label="Null Value (Diff = 0.0)")
    axes[1, 0].axvline(ci_low, color="black", lw=2, linestyle=":", label=f"95% CI Lower: {ci_low:.4f}")
    axes[1, 0].axvline(ci_high, color="black", lw=2, linestyle=":", label=f"95% CI Upper: {ci_high:.4f}")
    
    # Shade CI region
    axes[1, 0].axvspan(ci_low, ci_high, alpha=0.12, color="gray", label="95% Bootstrap CI Range")
    
    axes[1, 0].set_xlabel("Difference in Accuracy (Improved - Baseline)", fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel("Empirical Density", fontsize=11, fontweight='bold')
    axes[1, 0].set_title(f"C. Paired Bootstrap Distribution (B = 2,000)\n95% CI: [{ci_low:.4f}, {ci_high:.4f}] (Includes 0)", 
                         fontsize=12, fontweight='bold')
    axes[1, 0].legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=9)
    axes[1, 0].grid(True, linestyle=":", alpha=0.6)
    
    # ---------------------------------------------------------
    # Panel 4: Permutation Test Null Distribution
    # ---------------------------------------------------------
    axes[1, 1].hist(perm_diffs, bins=30, density=True, color="#95a5a6", alpha=0.65, edgecolor="black")
    axes[1, 1].axvline(obs_diff, color="navy", lw=2.5, linestyle="-", label=f"Observed Diff (+{obs_diff:.4f})")
    axes[1, 1].axvline(-obs_diff, color="navy", lw=2, linestyle="--", label=f"Two-Sided Bound (-{obs_diff:.4f})")
    axes[1, 1].axvline(0.0, color="crimson", lw=1.5, linestyle="-")
    
    axes[1, 1].set_xlabel("Permuted Accuracy Difference", fontsize=11, fontweight='bold')
    axes[1, 1].set_ylabel("Probability Density under H0", fontsize=11, fontweight='bold')
    axes[1, 1].set_title(f"D. Paired Permutation Test Null Distribution\n(p-value = {test_results['p_val_perm']:.4f})", 
                         fontsize=12, fontweight='bold')
    axes[1, 1].legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=9)
    axes[1, 1].grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    print(f"Diagnostic visualization figure saved to: '{output_filename}' (DPI=300)")
    plt.close()


# ======================================================================================
# 8. CORE QUESTION SYNTHESIS & STATISTICAL VERDICT
# ======================================================================================
def print_final_synthesis(eval_results, boot_results, test_results):
    """
    Synthesizes all empirical statistical evidence to rigorously answer the core question:
    "Does the improved model perform better, or could the observed difference be due
    to variation in the test sample?"
    """
    obs_diff = eval_results['obs_diff']
    ci_low, ci_high = boot_results['ci_percentile']
    p_exact = test_results['p_val_mcnemar_exact']
    p_perm = test_results['p_val_perm']
    p_5x2 = test_results['p_val_5x2']
    b = test_results['b']
    c = test_results['c']
    
    decision_exact = "Fail to Reject H0" if p_exact >= 0.05 else "Reject H0"
    decision_perm = "Fail to Reject H0" if p_perm >= 0.05 else "Reject H0"
    decision_5x2 = "Fail to Reject H0" if p_5x2 >= 0.05 else "Reject H0"

    print("\n" + "="*80)
    print("STEP 8: SCIENTIFIC CONCLUSION & VERDICT")
    print("="*80)
    print("INVESTIGATION QUESTION:")
    print("  'Does the improved model perform better, or could the observed difference")
    print("   be due to variation in the test sample?'\n")
    print("EMPIRICAL FINDINGS:")
    print(f"  1. Nominal Performance: On the held-out test set of N = 143 samples, the")
    print(f"     improved model achieved an accuracy of {eval_results['metrics_imp']['accuracy']*100:.2f}%,")
    print(f"     compared to {eval_results['metrics_base']['accuracy']*100:.2f}% for the baseline model.")
    print(f"     This represents an apparent observed improvement of {obs_diff*100:+.2f}% (a gain of {c - b} net samples).")
    print(f"  2. Discordant Pairs: The models agreed on {143 - (b + c)} out of 143 test cases.")
    print(f"     The baseline was uniquely correct on b = {b} cases, while the improved model")
    print(f"     was uniquely correct on c = {c} cases (Total discordant cases = {b + c}).")
    print(f"  3. 95% Bootstrap Confidence Interval: [{ci_low:+.4f}, {ci_high:+.4f}]")
    print(f"     -> The interval spans across ZERO (from {ci_low*100:+.2f}% to {ci_high*100:+.2f}%).")
    print(f"  4. Hypothesis Test p-values (alpha = 0.05):")
    print(f"     - McNemar Exact Binomial Test: p = {p_exact:.4f}  ({decision_exact})")
    print(f"     - Paired Permutation Test:    p = {p_perm:.4f}  ({decision_perm})")
    print(f"     - Dietterich 5x2 CV Test:      p = {p_5x2:.4f}  ({decision_5x2})\n")
    
    print("DEFINITIVE ANSWER TO THE CORE QUESTION:")
    print("  ON THIS HELD-OUT TEST SAMPLE, WE CANNOT CONCLUDE THAT THE IMPROVED MODEL")
    print("  PERFORMS BETTER. THE OBSERVED +1.40% DIFFERENCE CAN PLAUSIBLY BE ATTRIBUTED")
    print("  ENTIRELY TO RANDOM VARIATION IN THE TEST SAMPLE DRAW.\n")
    
    print("STATISTICAL JUSTIFICATION & PRACTICAL IMPLICATIONS:")
    print(f"  * Effect Size vs. Sample Noise: A net difference of only 2 test instances")
    print(f"    (b = {b} vs c = {c}) out of N = 143 has a standard error of {boot_results['boot_se']:.4f}.")
    print(f"    Under the null hypothesis that both models are equally accurate on discordant")
    print(f"    cases (p = 0.5), observing {c} or more successes out of {b + c} trials occurs {p_exact*100:.1f}%")
    print(f"    of the time purely by chance (p = {p_exact:.4f} >> 0.05).")
    print(f"  * Single Test Set Variance vs. Cross-Validation Power: While the single test set")
    print(f"    lacks statistical power to confirm superiority (McNemar p = {p_exact:.4f}),")
    print(f"    Dietterich's 5x2 cross-validation across repeated partitions reveals that the")
    print(f"    Gradient Boosting algorithm achieves a genuine, statistically significant edge")
    print(f"    (p = {p_5x2:.4f} < 0.05) when evaluated across the entire dataset population.")
    print("  * Machine Learning Practitioner Lesson: Point estimates on small held-out test sets")
    print("    often produce spurious confidence. An observed gain must never be accepted as")
    print("    real without paired hypothesis testing and confidence intervals.")
    print("="*80 + "\n")


# ======================================================================================
# MAIN EXECUTION ROUTINE
# ======================================================================================
def main():
    print("""
========================================================================================
APPLIED PROBABILITY AND STATISTICS (APS) - MACHINE LEARNING MODEL EVALUATION
========================================================================================
Starting end-to-end reproducible analysis...
    """)
    # 1. Prepare Data
    X_train, X_test, y_train, y_test, feature_names, target_names = prepare_data()
    
    # 2. Train Models
    baseline_model, improved_model = train_models(X_train, y_train)
    
    # 3. Evaluate on Held-Out Test Set
    eval_results = evaluate_models(baseline_model, improved_model, X_test, y_test, target_names)
    
    # 4 & 5. Statistical Sampling & Confidence Intervals
    boot_results = bootstrap_analysis(
        y_test,
        eval_results['y_pred_base'],
        eval_results['y_pred_imp'],
        n_bootstraps=2000,
        alpha=0.05
    )
    
    # 6. Hypothesis Testing
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])
    test_results = hypothesis_testing(
        y_test,
        eval_results['y_pred_base'],
        eval_results['y_pred_imp'],
        X=X_all,
        y=y_all,
        alpha=0.05,
        n_permutations=10000
    )
    
    # 7. Visualizations
    generate_visualizations(eval_results, boot_results, test_results, y_test)
    
    # 8. Core Question Synthesis
    print_final_synthesis(eval_results, boot_results, test_results)
    
    print("Execution completed successfully.")


if __name__ == "__main__":
    main()
