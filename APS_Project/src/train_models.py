"""
Model Training and Comparative Evaluation Module
================================================
Project: Sales Demand Forecasting & Inventory Planning System
Course: Applied Probability and Statistics (APS) / M.E. Big Data Analytics

This module implements:
1. Baseline Model:
   - Linear Regression with Feature Standardization (StandardScaler).
   - Serves as the classic linear econometric/time-series benchmark.
   
2. Improved Model:
   - Tuned XGBoost Regressor (with option for Random Forest Regressor).
   - Captures complex non-linear interactions, seasonal spikes, and store-item dynamics.
   
3. Rigorous Evaluation:
   - Both models are trained on the exact same chronological training set.
   - Both models predict on the exact same held-out test set (final 28 days, N=840).
   - Domain-specific post-processing: Non-negative clipping (sales >= 0).
   - Standard Regression Metrics: MAE, RMSE, R² Score, and sMAPE.
   
4. Serialization:
   - Exports trained models, test predictions, and comparative metrics to outputs/results/.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Tuple, Any
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import xgboost as xgb

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def calculate_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes Symmetric Mean Absolute Percentage Error (sMAPE) in percentage:
    sMAPE = (100% / n) * sum( 2 * |y_true - y_pred| / (|y_true| + |y_pred| + 1e-5) )
    """
    denominator = (np.abs(y_true) + np.abs(y_pred)) + 1e-5
    diff = np.abs(y_true - y_pred)
    return float(100.0 * np.mean(2.0 * diff / denominator))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> Dict[str, float]:
    """Computes standard evaluation metrics."""
    # Enforce non-negativity constraint for sales predictions
    y_pred_clipped = np.clip(y_pred, a_min=0.0, a_max=None)
    
    mae = float(mean_absolute_error(y_true, y_pred_clipped))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred_clipped)))
    r2 = float(r2_score(y_true, y_pred_clipped))
    smape = calculate_smape(y_true, y_pred_clipped)
    
    return {
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2_Score": r2,
        "sMAPE (%)": smape
    }


def train_baseline_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    random_state: int = 42
) -> Pipeline:
    """
    Trains the Baseline Linear Regression model using StandardScaler pipeline.
    """
    logger.info("Training Baseline Model: LinearRegression (StandardScaler)...")
    baseline_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])
    baseline_pipeline.fit(X_train, y_train)
    logger.info("Baseline Linear Regression model successfully trained.")
    return baseline_pipeline


def train_improved_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    model_type: str = "xgboost",
    random_state: int = 42
) -> Any:
    """
    Trains the Improved Machine Learning model:
    - Default: XGBoost Regressor (extreme gradient boosting with regularized trees).
    - Alternative: Random Forest Regressor (ensemble of bagging trees).
    """
    if model_type.lower() == "xgboost":
        logger.info("Training Improved Model: XGBoost Regressor (n_estimators=150, max_depth=6)...")
        model = xgb.XGBRegressor(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=random_state,
            n_jobs=-1
        )
    elif model_type.lower() == "random_forest":
        logger.info("Training Improved Model: Random Forest Regressor (n_estimators=100, max_depth=12)...")
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1
        )
    else:
        raise ValueError(f"Unsupported model_type: '{model_type}'. Choose 'xgboost' or 'random_forest'.")
        
    model.fit(X_train, y_train)
    logger.info(f"Improved model ({model_type}) successfully trained.")
    return model


def train_and_evaluate_models(
    data_dict: Dict,
    output_dir: str = "outputs/results",
    improved_model_type: str = "xgboost",
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Executes training and comparative evaluation of both baseline and improved models
    on the exact same held-out test dataset.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    X_train = data_dict['X_train']
    y_train = data_dict['y_train']
    X_test = data_dict['X_test']
    y_test = data_dict['y_test']
    test_df = data_dict['test_df'].copy()
    
    # 1. Train Baseline
    baseline_model = train_baseline_model(X_train, y_train, random_state=random_state)
    
    # 2. Train Improved Model
    improved_model = train_improved_model(
        X_train, y_train, model_type=improved_model_type, random_state=random_state
    )
    
    # Optional: Also train Random Forest for complete tripartite benchmarking
    rf_model = None
    if improved_model_type.lower() == "xgboost":
        logger.info("Also training complementary Random Forest benchmark for comprehensive comparison...")
        rf_model = train_improved_model(X_train, y_train, model_type="random_forest", random_state=random_state)
        
    # 3. Generate Predictions on Held-Out Test Set
    logger.info(f"Generating test predictions on held-out test partition (N = {len(y_test)})...")
    pred_base = np.clip(baseline_model.predict(X_test), 0.0, None)
    pred_imp = np.clip(improved_model.predict(X_test), 0.0, None)
    
    # 4. Compute Evaluation Metrics
    metrics_base = compute_metrics(y_test, pred_base, "Baseline (Linear Regression)")
    metrics_imp = compute_metrics(y_test, pred_imp, f"Improved ({improved_model_type.upper()})")
    
    comparison_rows = [metrics_base, metrics_imp]
    
    pred_rf = None
    if rf_model is not None:
        pred_rf = np.clip(rf_model.predict(X_test), 0.0, None)
        metrics_rf = compute_metrics(y_test, pred_rf, "Random Forest Benchmark")
        comparison_rows.append(metrics_rf)
        
    # Calculate Percentage Improvement of Improved vs Baseline
    delta_mae = metrics_base['MAE'] - metrics_imp['MAE']
    pct_imp_mae = (delta_mae / metrics_base['MAE']) * 100.0
    
    delta_rmse = metrics_base['RMSE'] - metrics_imp['RMSE']
    pct_imp_rmse = (delta_rmse / metrics_base['RMSE']) * 100.0
    
    delta_r2 = metrics_imp['R2_Score'] - metrics_base['R2_Score']
    
    diff_row = {
        "Model": "Absolute Improvement (Baseline - Improved)",
        "MAE": f"+{delta_mae:.4f} ({pct_imp_mae:.2f}%)",
        "RMSE": f"+{delta_rmse:.4f} ({pct_imp_rmse:.2f}%)",
        "R2_Score": f"+{delta_r2:.4f}",
        "sMAPE (%)": f"-{metrics_base['sMAPE (%)'] - metrics_imp['sMAPE (%)']:.2f}%"
    }
    
    metrics_df = pd.DataFrame(comparison_rows)
    metrics_csv_path = output_dir / "model_evaluation_metrics.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)
    logger.info(f"Saved evaluation metrics to: {metrics_csv_path}")
    
    # Build complete test predictions dataframe
    test_df['actual_sales'] = y_test
    test_df['pred_baseline'] = pred_base
    test_df['pred_improved'] = pred_imp
    test_df['error_abs_baseline'] = np.abs(y_test - pred_base)
    test_df['error_abs_improved'] = np.abs(y_test - pred_imp)
    test_df['error_squared_baseline'] = (y_test - pred_base) ** 2
    test_df['error_squared_improved'] = (y_test - pred_imp) ** 2
    test_df['error_diff_abs'] = test_df['error_abs_baseline'] - test_df['error_abs_improved']
    test_df['error_diff_squared'] = test_df['error_squared_baseline'] - test_df['error_squared_improved']
    
    if pred_rf is not None:
        test_df['pred_random_forest'] = pred_rf
        
    predictions_csv_path = output_dir / "test_predictions_comparison.csv"
    test_df.to_csv(predictions_csv_path, index=False)
    logger.info(f"Saved paired test predictions to: {predictions_csv_path}")
    
    # Save model artifacts
    artifacts = {
        "baseline_model": baseline_model,
        "improved_model": improved_model,
        "rf_model": rf_model,
        "feature_cols": data_dict['feature_cols'],
        "metrics_df": metrics_df,
        "metrics_base": metrics_base,
        "metrics_imp": metrics_imp,
        "delta_mae": delta_mae,
        "delta_rmse": delta_rmse,
        "delta_r2": delta_r2,
        "pct_imp_mae": pct_imp_mae,
        "pct_imp_rmse": pct_imp_rmse
    }
    model_save_path = output_dir / "trained_forecasting_models.joblib"
    joblib.dump(artifacts, model_save_path)
    logger.info(f"Saved model artifacts to: {model_save_path}")
    
    return {
        "artifacts": artifacts,
        "metrics_df": metrics_df,
        "diff_row": diff_row,
        "test_df": test_df,
        "pred_base": pred_base,
        "pred_imp": pred_imp,
        "y_test": y_test
    }


if __name__ == "__main__":
    from data_preprocessing import prepare_m5_data
    print("Executing model training and comparative evaluation...")
    data_dict = prepare_m5_data()
    results = train_and_evaluate_models(data_dict)
    print("\n--- Model Evaluation Comparison Table ---")
    print(results['metrics_df'].to_string(index=False))
    print("\nDifference / Observed Improvement:")
    print(results['diff_row'])
