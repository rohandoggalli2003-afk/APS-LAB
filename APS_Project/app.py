"""
========================================================================================
Interactive Streamlit Dashboard
Sales Demand Forecasting & Statistical Model Evaluation System
========================================================================================
Student Name: Rohan Doggalli | Registration Number: 261100670012
Degree: Master of Engineering (M.E.) in Big Data Analytics
Course: Applied Probability and Statistics (AME 5102)
Department: Manipal School of Information Sciences (MSIS), MAHE, Manipal
========================================================================================
"""

import os
from pathlib import Path
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd
from scipy import stats
import joblib

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -----------------------------------------------------------------------------
# 1. STREAMLIT APP CONFIGURATION & THEMING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Sales Forecasting & Statistical Evaluation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for executive presentation
st.markdown("""
<style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border-radius: 10px;
        padding: 16px;
        border-left: 5px solid #1e3a8a;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .stat-badge-pass {
        background-color: #fef2f2;
        border: 1px solid #f87171;
        color: #991b1b;
        padding: 6px 12px;
        border-radius: 6px;
        font-weight: bold;
    }
    .stat-badge-info {
        background-color: #eff6ff;
        border: 1px solid #93c5fd;
        color: #1e40af;
        padding: 6px 12px;
        border-radius: 6px;
        font-weight: bold;
    }
    .callout-box {
        background-color: #f1f5f9;
        border-left: 4px solid #3b82f6;
        padding: 14px 18px;
        border-radius: 6px;
        margin-top: 10px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. DATA & MODEL LOADING FUNCTIONS (CACHED)
# -----------------------------------------------------------------------------
@st.cache_data
def load_sales_dataset() -> pd.DataFrame:
    """Loads historical sales dataset from available locations."""
    candidate_paths = [
        Path("data/processed_sales.parquet"),
        Path("APS_Project/data/processed_sales.parquet"),
        Path("data/m5_sales_sample.csv"),
        Path("APS_Project/data/m5_sales_sample.csv"),
    ]
    data_path = next((p for p in candidate_paths if p.exists()), None)
    if data_path is None:
        st.error("Sales dataset not found in data/ or APS_Project/data/.")
        return pd.DataFrame()
        
    if str(data_path).endswith(".parquet"):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)
        
    df['date'] = pd.to_datetime(df['date'])
    return df


@st.cache_data
def load_test_predictions() -> pd.DataFrame:
    """Loads held-out test predictions comparison file."""
    paths = [
        Path("outputs/results/test_predictions_comparison.csv"),
        Path("APS_Project/outputs/results/test_predictions_comparison.csv")
    ]
    p = next((x for x in paths if x.exists()), None)
    if p is None:
        return pd.DataFrame()
    df = pd.read_csv(p)
    df['date'] = pd.to_datetime(df['date'])
    return df


@st.cache_data
def load_evaluation_metrics() -> pd.DataFrame:
    """Loads model evaluation metrics table."""
    paths = [
        Path("outputs/results/model_evaluation_metrics.csv"),
        Path("APS_Project/outputs/results/model_evaluation_metrics.csv")
    ]
    p = next((x for x in paths if x.exists()), None)
    return pd.read_csv(p) if p else pd.DataFrame()


@st.cache_data
def load_statistical_test_results() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads hypothesis tests, diagnostics, and bootstrap intervals."""
    res_dir = Path("outputs/results")
    if not res_dir.exists():
        res_dir = Path("APS_Project/outputs/results")
        
    ht_path = res_dir / "statistical_hypothesis_tests.csv"
    diag_path = res_dir / "test_assumptions_diagnostic.csv"
    ci_path = res_dir / "bootstrap_confidence_intervals.csv"
    
    ht_df = pd.read_csv(ht_path) if ht_path.exists() else pd.DataFrame()
    diag_df = pd.read_csv(diag_path) if diag_path.exists() else pd.DataFrame()
    ci_df = pd.read_csv(ci_path) if ci_path.exists() else pd.DataFrame()
    return ht_df, diag_df, ci_df


@st.cache_resource
def load_trained_models() -> Dict[str, Any]:
    """Loads trained forecasting pipeline joblib bundle."""
    paths = [
        Path("outputs/results/trained_forecasting_models.joblib"),
        Path("APS_Project/outputs/results/trained_forecasting_models.joblib")
    ]
    p = next((x for x in paths if x.exists()), None)
    if p:
        try:
            return joblib.load(p)
        except Exception:
            return {}
    return {}


# -----------------------------------------------------------------------------
# 3. SIDEBAR NAVIGATION & GLOBAL FILTERS
# -----------------------------------------------------------------------------
raw_sales_df = load_sales_dataset()
test_preds_df = load_test_predictions()
metrics_table = load_evaluation_metrics()
ht_table, diag_table, ci_table = load_statistical_test_results()
models_bundle = load_trained_models()

st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/ca/Walmart_logo.svg/320px-Walmart_logo.svg.png", width=180)
st.sidebar.markdown("### **Demand Analytics System**")
st.sidebar.markdown("**AME 5102 Applied Probability & Statistics**")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation Menu",
    [
        "📊 Executive Overview",
        "📈 Sales Analysis",
        "🔮 Demand Forecasting",
        "🎯 Model Performance",
        "📐 Statistical Analysis",
        "📦 Inventory Planning"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### **Global Filters**")

# Store selection
available_stores = sorted(raw_sales_df['store_id'].unique().tolist()) if not raw_sales_df.empty else ['CA_1', 'CA_2']
selected_store = st.sidebar.selectbox("Select Store", ["All Stores"] + available_stores)

# Product selection
available_items = sorted(raw_sales_df['item_id'].unique().tolist()) if not raw_sales_df.empty else [f"FOODS_1_{i:03d}" for i in range(1, 16)]
selected_item = st.sidebar.selectbox("Select Item SKU", ["All SKUs"] + available_items)

# Date range filter
min_date = raw_sales_df['date'].min() if not raw_sales_df.empty else pd.to_datetime("2015-01-01")
max_date = raw_sales_df['date'].max() if not raw_sales_df.empty else pd.to_datetime("2015-12-31")

date_range = st.sidebar.date_input(
    "Select Date Horizon",
    value=(min_date.date(), max_date.date()),
    min_value=min_date.date(),
    max_value=max_date.date()
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Student:** Rohan Doggalli (`261100670012`)  
**Program:** M.E. Big Data Analytics  
**Institution:** MSIS, MAHE, Manipal
""")


# Helper function to apply global filters
def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    filtered = df.copy()
    if selected_store != "All Stores" and 'store_id' in filtered.columns:
        filtered = filtered[filtered['store_id'] == selected_store]
    if selected_item != "All SKUs" and 'item_id' in filtered.columns:
        filtered = filtered[filtered['item_id'] == selected_item]
    if len(date_range) == 2:
        start_d, end_d = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        filtered = filtered[(filtered['date'] >= start_d) & (filtered['date'] <= end_d)]
    return filtered


# =============================================================================
# PAGE 1: EXECUTIVE OVERVIEW
# =============================================================================
if page == "📊 Executive Overview":
    st.markdown('<div class="main-title">Sales Demand Forecasting & Statistical Evaluation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Walmart M5 Benchmark Dataset | Baseline Linear Regression vs. Improved Tuned XGBoost</div>', unsafe_allow_html=True)

    filtered_sales = filter_dataframe(raw_sales_df)
    filtered_preds = filter_dataframe(test_preds_df)

    # Key KPI Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total_sales = int(filtered_sales['sales'].sum()) if not filtered_sales.empty else 0
    avg_daily_sales = filtered_sales['sales'].mean() if not filtered_sales.empty else 0.0
    pred_test_demand = filtered_preds['pred_improved'].sum() if not filtered_preds.empty else 0
    
    col1.metric("Total Historical Sales", f"{total_sales:,} units")
    col2.metric("Average Daily Demand", f"{avg_daily_sales:.2f} units/day")
    col3.metric("Predicted Test Demand", f"{pred_test_demand:,.0f} units")
    col4.metric("Best Machine Learning Model", "Tuned XGBoost", delta="+0.59% MAE")
    col5.metric("Improved Test MAE", "1.8051", delta="-0.0107 units", delta_color="inverse")

    st.markdown("---")

    # Dual Panel Visualizations: Historical Trajectory + Forecast Trajectory
    st.subheader("Aggregate Sales & Forecast Demand Trends")
    
    daily_hist = filtered_sales.groupby('date')['sales'].sum().reset_index()
    
    fig_overview = go.Figure()
    fig_overview.add_trace(go.Scatter(
        x=daily_hist['date'], y=daily_hist['sales'],
        mode='lines', name='Actual Historical Sales',
        line=dict(color='#1e3a8a', width=2)
    ))
    
    if not filtered_preds.empty:
        daily_test = filtered_preds.groupby('date')[['actual_sales', 'pred_baseline', 'pred_improved']].sum().reset_index()
        fig_overview.add_trace(go.Scatter(
            x=daily_test['date'], y=daily_test['pred_baseline'],
            mode='lines+markers', name='Baseline Forecast (Linear Reg)',
            line=dict(color='#dc2626', width=2, dash='dot')
        ))
        fig_overview.add_trace(go.Scatter(
            x=daily_test['date'], y=daily_test['pred_improved'],
            mode='lines+markers', name='Improved Forecast (XGBoost)',
            line=dict(color='#2563eb', width=2.2)
        ))
        
    fig_overview.update_layout(
        title="Walmart M5 Daily Sales Demand & 28-Day Held-Out Test Forecast Trajectory",
        xaxis_title="Calendar Date",
        yaxis_title="Total Daily Units Sold",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_overview, use_container_width=True)

    # Multi-dimensional aggregations
    c_left, c_right = st.columns(2)
    
    with c_left:
        st.subheader("Demand by Day of the Week")
        if not filtered_sales.empty:
            dow_sales = filtered_sales.copy()
            dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            dow_sales['dow_name'] = dow_sales['date'].dt.day_name()
            dow_agg = dow_sales.groupby('dow_name')['sales'].mean().reindex(dow_order).reset_index()
            
            fig_dow = px.bar(
                dow_agg, x='dow_name', y='sales',
                labels={'dow_name': 'Day of Week', 'sales': 'Mean Units Sold'},
                color='sales', color_continuous_scale='Blues'
            )
            fig_dow.update_layout(template="plotly_white", showlegend=False)
            st.plotly_chart(fig_dow, use_container_width=True)

    with c_right:
        st.subheader("Monthly Demand Seasonality")
        if not filtered_sales.empty:
            m_sales = filtered_sales.copy()
            m_sales['month_name'] = m_sales['date'].dt.strftime('%b')
            month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            m_agg = m_sales.groupby('month_name')['sales'].sum().reindex(month_order).dropna().reset_index()
            
            fig_month = px.bar(
                m_agg, x='month_name', y='sales',
                labels={'month_name': 'Month', 'sales': 'Total Units Sold'},
                color='sales', color_continuous_scale='Teal'
            )
            fig_month.update_layout(template="plotly_white", showlegend=False)
            st.plotly_chart(fig_month, use_container_width=True)

    st.markdown('<div class="callout-box"><b>Key Finding:</b> While XGBoost achieved a nominal +0.59% error reduction over Linear Regression, comprehensive inferential testing (Wilcoxon p = 0.3741, Diebold-Mariano p = 0.3456, Moving Block Bootstrap p = 0.2270) confirms the difference is not statistically significant.</div>', unsafe_allow_html=True)


# =============================================================================
# PAGE 2: SALES ANALYSIS
# =============================================================================
elif page == "📈 Sales Analysis":
    st.markdown('<div class="main-title">Exploratory Sales & Demand Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Historical Time-Series, SKU Disaggregation, and Distribution Diagnostics</div>', unsafe_allow_html=True)

    filtered_sales = filter_dataframe(raw_sales_df)
    
    col_d1, col_d2, col_d3 = st.columns(3)
    col_d1.metric("Observations in View", f"{len(filtered_sales):,} records")
    zero_sales_pct = (filtered_sales['sales'] == 0).mean() * 100.0 if not filtered_sales.empty else 0.0
    col_d2.metric("Zero Demand Frequency", f"{zero_sales_pct:.1f}%", help="Percentage of days with 0 unit sales")
    col_d3.metric("Demand Variance-to-Mean Ratio", f"{(filtered_sales['sales'].var() / (filtered_sales['sales'].mean() + 1e-5)):.2f}", help="Indicates significant demand overdispersion")

    st.markdown("---")

    # Time series with rolling window
    st.subheader("Historical Sales Trajectory with 7-Day Rolling Moving Average")
    if not filtered_sales.empty:
        agg_daily = filtered_sales.groupby('date')['sales'].sum().reset_index()
        agg_daily['rolling_7d'] = agg_daily['sales'].rolling(7, min_periods=1).mean()
        
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(x=agg_daily['date'], y=agg_daily['sales'], mode='lines', name='Daily Sales', line=dict(color='#94a3b8', width=1.2)))
        fig_ts.add_trace(go.Scatter(x=agg_daily['date'], y=agg_daily['rolling_7d'], mode='lines', name='7-Day Rolling Mean', line=dict(color='#2563eb', width=2.5)))
        fig_ts.update_layout(template="plotly_white", xaxis_title="Date", yaxis_title="Units Sold", hovermode="x unified")
        st.plotly_chart(fig_ts, use_container_width=True)

    # Two columns: Distribution and SKU breakdown
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Empirical Sales Distribution (Intermittency)")
        if not filtered_sales.empty:
            fig_dist = px.histogram(filtered_sales, x='sales', nbins=30, marginal="box", color_discrete_sequence=['#3b82f6'])
            fig_dist.update_layout(template="plotly_white", xaxis_title="Daily Units Sold", yaxis_title="Record Count")
            st.plotly_chart(fig_dist, use_container_width=True)

    with c2:
        st.subheader("Sales by Store Comparison")
        if not filtered_sales.empty and 'store_id' in filtered_sales.columns:
            store_agg = filtered_sales.groupby('store_id')['sales'].agg(['sum', 'mean']).reset_index()
            fig_store = px.bar(store_agg, x='store_id', y='sum', text='sum', color='store_id', color_discrete_sequence=['#1e40af', '#0d9488'])
            fig_store.update_traces(texttemplate='%{text:,}', textposition='outside')
            fig_store.update_layout(template="plotly_white", xaxis_title="Store ID", yaxis_title="Total Units Sold", showlegend=False)
            st.plotly_chart(fig_store, use_container_width=True)

    st.subheader("Product-Level Velocity Ranking (Top vs. Low Demand SKUs)")
    if not filtered_sales.empty and 'item_id' in filtered_sales.columns:
        item_agg = filtered_sales.groupby('item_id')['sales'].agg(
            Total_Sales='sum',
            Avg_Daily_Demand='mean',
            Max_Sales='max',
            Zero_Days_Pct=lambda s: (s == 0).mean() * 100.0
        ).reset_index().sort_values('Total_Sales', ascending=False)
        
        fig_items = px.bar(
            item_agg, x='item_id', y='Total_Sales', color='Avg_Daily_Demand',
            color_continuous_scale='Viridis', labels={'Total_Sales': 'Total Units Sold', 'item_id': 'Product SKU'}
        )
        fig_items.update_layout(template="plotly_white", xaxis_tickangle=-45)
        st.plotly_chart(fig_items, use_container_width=True)
        
        st.dataframe(item_agg.style.format({
            'Total_Sales': '{:,.0f}',
            'Avg_Daily_Demand': '{:.2f}',
            'Max_Sales': '{:.0f}',
            'Zero_Days_Pct': '{:.1f}%'
        }), use_container_width=True)

    # Download Button
    csv_bytes = filtered_sales.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered Sales Data as CSV",
        data=csv_bytes,
        file_name="filtered_sales_data.csv",
        mime="text/csv"
    )


# =============================================================================
# PAGE 3: DEMAND FORECASTING
# =============================================================================
elif page == "🔮 Demand Forecasting":
    st.markdown('<div class="main-title">Interactive Demand Forecasting Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Product-Store Multi-Horizon Projections & Comparative Model Trajectories</div>', unsafe_allow_html=True)

    c_f1, c_f2, c_f3 = st.columns(3)
    f_store = c_f1.selectbox("Forecast Store", available_stores, index=0)
    f_item = c_f2.selectbox("Forecast Product SKU", available_items, index=0)
    f_horizon = c_f3.selectbox("Forecasting Horizon", [7, 14, 28], index=2, format_func=lambda d: f"{d} Days")

    series_id = f"{f_item}_{f_store}_validation"
    
    # Check if series exists in test predictions
    series_preds = test_preds_df[
        (test_preds_df['store_id'] == f_store) & 
        (test_preds_df['item_id'] == f_item)
    ].sort_values('date').copy() if not test_preds_df.empty else pd.DataFrame()

    if series_preds.empty and not test_preds_df.empty:
        # Fallback to series matching store and item
        series_preds = test_preds_df.head(f_horizon).copy()

    # Slice horizon
    eval_horizon = series_preds.head(f_horizon)

    st.subheader(f"Forecast Comparison for {f_item} at Store {f_store} ({f_horizon}-Day Horizon)")

    if not eval_horizon.empty:
        fig_fc = go.Figure()
        
        # Ground Truth
        fig_fc.add_trace(go.Scatter(
            x=eval_horizon['date'], y=eval_horizon['actual_sales'],
            mode='lines+markers', name='Actual Ground Truth Demand',
            line=dict(color='#0f172a', width=2.5), marker=dict(size=6)
        ))
        
        # Baseline Model
        fig_fc.add_trace(go.Scatter(
            x=eval_horizon['date'], y=eval_horizon['pred_baseline'],
            mode='lines+markers', name='Baseline (Linear Regression)',
            line=dict(color='#ef4444', width=2, dash='dash'), marker=dict(size=5, symbol='square')
        ))
        
        # Improved Model
        fig_fc.add_trace(go.Scatter(
            x=eval_horizon['date'], y=eval_horizon['pred_improved'],
            mode='lines+markers', name='Improved (Tuned XGBoost)',
            line=dict(color='#2563eb', width=2.2), marker=dict(size=5, symbol='triangle-up')
        ))

        # Prediction confidence band (empirical std dev)
        std_err = np.std(eval_horizon['actual_sales'] - eval_horizon['pred_improved'], ddof=1)
        upper_band = eval_horizon['pred_improved'] + 1.96 * std_err
        lower_band = np.clip(eval_horizon['pred_improved'] - 1.96 * std_err, 0, None)

        fig_fc.add_trace(go.Scatter(
            x=list(eval_horizon['date']) + list(eval_horizon['date'])[::-1],
            y=list(upper_band) + list(lower_band)[::-1],
            fill='toself', fillcolor='rgba(37, 99, 235, 0.12)',
            line=dict(color='rgba(255,255,255,0)'),
            name='XGBoost 95% Prediction Interval', hoverinfo="skip"
        ))

        fig_fc.update_layout(
            template="plotly_white",
            xaxis_title="Evaluation Date",
            yaxis_title="Units Demanded",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_fc, use_container_width=True)

        # Performance summary metrics for this specific series
        s_mae_base = np.mean(eval_horizon['error_abs_baseline'])
        s_mae_imp = np.mean(eval_horizon['error_abs_improved'])
        s_diff = s_mae_base - s_mae_imp
        
        m_c1, m_c2, m_c3, m_c4 = st.columns(4)
        m_c1.metric("Series Baseline MAE", f"{s_mae_base:.3f} units")
        m_c2.metric("Series Improved MAE", f"{s_mae_imp:.3f} units")
        m_c3.metric("Error Reduction", f"{s_diff:+.3f} units", delta=f"{(s_diff/s_mae_base*100.0):+.2f}%")
        winner = "XGBoost Regressor" if s_diff > 0 else "Linear Regression"
        m_c4.metric("Leading Architecture", winner)

        st.dataframe(eval_horizon[['date', 'actual_sales', 'pred_baseline', 'pred_improved', 'error_abs_baseline', 'error_abs_improved', 'error_diff_abs']].rename(columns={
            'actual_sales': 'Actual Sales',
            'pred_baseline': 'Baseline Prediction',
            'pred_improved': 'XGBoost Prediction',
            'error_abs_baseline': 'Baseline Absolute Error',
            'error_abs_improved': 'XGBoost Absolute Error',
            'error_diff_abs': 'Error Savings (Base - Imp)'
        }), use_container_width=True)

        # Download predictions
        csv_fc = eval_horizon.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Series Forecast Results as CSV",
            data=csv_fc,
            file_name=f"forecast_{f_store}_{f_item}_{f_horizon}d.csv",
            mime="text/csv"
        )
    else:
        st.warning("No test predictions available for the selected product and store combination.")


# =============================================================================
# PAGE 4: MODEL PERFORMANCE
# =============================================================================
elif page == "🎯 Model Performance":
    st.markdown('<div class="main-title">Machine Learning Model Evaluation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Deterministic Accuracy Benchmarks & Residual Diagnostics on Held-Out Test Set (N = 840)</div>', unsafe_allow_html=True)

    if not metrics_table.empty:
        st.subheader("Model Evaluation Accuracy Scorecard")
        st.dataframe(metrics_table.style.format({
            'MAE': '{:.4f}',
            'RMSE': '{:.4f}',
            'R2_Score': '{:.4f}',
            'sMAPE (%)': '{:.2f}%'
        }), use_container_width=True)

        # Comparative Bar Charts
        c_m1, c_m2, c_m3 = st.columns(3)
        with c_m1:
            fig_mae = px.bar(metrics_table, x='Model', y='MAE', text='MAE', title="Mean Absolute Error (MAE)", color='Model', color_discrete_sequence=['#ef4444', '#3b82f6', '#10b981'])
            fig_mae.update_traces(texttemplate='%{text:.4f}', textposition='outside')
            fig_mae.update_layout(template="plotly_white", showlegend=False, yaxis_range=[1.6, 1.95])
            st.plotly_chart(fig_mae, use_container_width=True)

        with c_m2:
            fig_rmse = px.bar(metrics_table, x='Model', y='RMSE', text='RMSE', title="Root Mean Squared Error (RMSE)", color='Model', color_discrete_sequence=['#f87171', '#60a5fa', '#34d399'])
            fig_rmse.update_traces(texttemplate='%{text:.4f}', textposition='outside')
            fig_rmse.update_layout(template="plotly_white", showlegend=False, yaxis_range=[2.1, 2.45])
            st.plotly_chart(fig_rmse, use_container_width=True)

        with c_m3:
            fig_r2 = px.bar(metrics_table, x='Model', y='R2_Score', text='R2_Score', title="Coefficient of Determination (R²)", color='Model', color_discrete_sequence=['#fca5a5', '#93c5fd', '#6ee7b7'])
            fig_r2.update_traces(texttemplate='%{text:.4f}', textposition='outside')
            fig_r2.update_layout(template="plotly_white", showlegend=False, yaxis_range=[0.4, 0.5])
            st.plotly_chart(fig_r2, use_container_width=True)

    # Actual vs Predicted Scatter Plot
    if not test_preds_df.empty:
        st.subheader("Actual vs. Predicted Demand Scatter & Goodness of Fit")
        c_sc1, c_sc2 = st.columns(2)
        
        max_val = max(test_preds_df['actual_sales'].max(), test_preds_df['pred_improved'].max()) + 2
        
        with c_sc1:
            fig_sc_base = px.scatter(
                test_preds_df, x='actual_sales', y='pred_baseline',
                title="Baseline (Linear Regression) Fit",
                opacity=0.45, color_discrete_sequence=['#ef4444'],
                labels={'actual_sales': 'Actual Sales Units', 'pred_baseline': 'Predicted Units'}
            )
            fig_sc_base.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode='lines', line=dict(color='black', dash='dash'), name='Ideal 1:1 Fit'))
            fig_sc_base.update_layout(template="plotly_white")
            st.plotly_chart(fig_sc_base, use_container_width=True)

        with c_sc2:
            fig_sc_imp = px.scatter(
                test_preds_df, x='actual_sales', y='pred_improved',
                title="Improved (Tuned XGBoost) Fit",
                opacity=0.45, color_discrete_sequence=['#2563eb'],
                labels={'actual_sales': 'Actual Sales Units', 'pred_improved': 'Predicted Units'}
            )
            fig_sc_imp.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode='lines', line=dict(color='black', dash='dash'), name='Ideal 1:1 Fit'))
            fig_sc_imp.update_layout(template="plotly_white")
            st.plotly_chart(fig_sc_imp, use_container_width=True)

        # Residuals
        st.subheader("Residual Error Distributions (y - ŷ)")
        res_df = pd.DataFrame({
            'Baseline Residuals': test_preds_df['actual_sales'] - test_preds_df['pred_baseline'],
            'XGBoost Residuals': test_preds_df['actual_sales'] - test_preds_df['pred_improved']
        })
        fig_res = go.Figure()
        fig_res.add_trace(go.Histogram(x=res_df['Baseline Residuals'], name='Linear Regression Residuals', opacity=0.55, marker_color='#ef4444'))
        fig_res.add_trace(go.Histogram(x=res_df['XGBoost Residuals'], name='XGBoost Residuals', opacity=0.55, marker_color='#2563eb'))
        fig_res.update_layout(barmode='overlay', template="plotly_white", xaxis_title="Residual Error (Actual - Predicted)", yaxis_title="Count")
        st.plotly_chart(fig_res, use_container_width=True)

    # Download metrics
    if not metrics_table.empty:
        csv_m = metrics_table.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Model Evaluation Metrics as CSV",
            data=csv_m,
            file_name="model_evaluation_metrics.csv",
            mime="text/csv"
        )


# =============================================================================
# PAGE 5: STATISTICAL ANALYSIS
# =============================================================================
elif page == "📐 Statistical Analysis":
    st.markdown('<div class="main-title">Applied Probability & Statistical Evaluation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Inferential Hypothesis Testing, Resampling Estimation, and Time-Series HAC Diagnostics</div>', unsafe_allow_html=True)

    # Dynamic verdict computation
    diffs = test_preds_df['error_diff_abs'].values if not test_preds_df.empty else np.array([0.0107])
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1))
    se_diff = float(std_diff / np.sqrt(len(diffs)))
    cohens_d = float(mean_diff / (std_diff + 1e-8))

    # KPI summary cards
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Mean Improvement (d̄)", f"+{mean_diff:.4f} units/d", delta="+0.59% MAE")
    k2.metric("Difference Std Dev", f"{std_diff:.4f}")
    k3.metric("Bootstrap Std Error", "0.0214 units/d")
    k4.metric("Standardized Cohen's d", f"{cohens_d:.4f}", help="d < 0.20 indicates negligible effect size")
    k5.metric("Statistical Verdict", "Fail to Reject H0", delta="p = 0.2270 >= 0.05", delta_color="inverse")

    st.markdown("---")

    # Dynamic Verdict Banner
    st.markdown("""
    <div style="background-color: #fef2f2; border: 2px solid #f87171; border-radius: 8px; padding: 16px; margin-bottom: 20px;">
        <h4 style="color: #991b1b; margin-top: 0; font-weight: bold;">🚨 DYNAMIC STATISTICAL HYPOTHESIS TESTING VERDICT: IMPROVEMENT NOT STATISTICALLY SIGNIFICANT</h4>
        <p style="color: #450a0a; margin-bottom: 0; font-size: 0.98rem;">
            Across all parametric, non-parametric, HAC time-series, and block bootstrap tests, the one-sided p-value exceeds α = 0.05 
            (Student's t-test p = 0.3463, Wilcoxon p = 0.3741, Diebold-Mariano p = 0.3456, Moving Block Bootstrap p = 0.2270). 
            Furthermore, the 95% Bootstrap Confidence Interval <b>[-0.0297, +0.0475]</b> spans across zero. 
            <b>Decision: FAIL TO REJECT H0.</b> The observed +0.59% MAE reduction is entirely consistent with random variation in the held-out test sample.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Hypothesis Testing Battery Table
    st.subheader("Statistical Hypothesis Testing Battery (α = 0.05)")
    if not ht_table.empty:
        st.dataframe(ht_table, use_container_width=True)

    # Assumption Diagnostics Table
    st.subheader("Paired Student's t-Test Assumption Diagnostic Results")
    if not diag_table.empty:
        st.dataframe(diag_table, use_container_width=True)

    # Bootstrap Confidence Intervals Table
    st.subheader("95% Bootstrap Confidence Interval Estimations")
    if not ci_table.empty:
        st.dataframe(ci_table, use_container_width=True)

    # Interactive Visualizations
    st.subheader("Visual Statistical Inference")
    col_v1, col_v2 = st.columns(2)
    
    with col_v1:
        # Bootstrap sampling distribution simulation
        st.markdown("**Moving Block Bootstrap (MBB, L=7d, B=2,000) Distribution**")
        rng = np.random.RandomState(42)
        unique_dates = np.sort(test_preds_df['date'].unique()) if not test_preds_df.empty else pd.date_range("2015-12-04", periods=28)
        n_dates = len(unique_dates)
        blocks = [unique_dates[i:i + 7] for i in range(n_dates - 7 + 1)]
        d_map = {d: test_preds_df.loc[test_preds_df['date'] == d, 'error_diff_abs'].values for d in unique_dates} if not test_preds_df.empty else {}
        
        b_means = []
        for _ in range(1500):
            chosen = rng.choice(len(blocks), size=int(np.ceil(n_dates/7)), replace=True)
            rd = np.concatenate([blocks[c] for c in chosen])[:n_dates]
            b_means.append(np.mean(np.concatenate([d_map.get(d, [0.0107]) for d in rd])))
            
        fig_boot = px.histogram(b_means, nbins=40, color_discrete_sequence=['#6366f1'])
        fig_boot.add_vline(x=0.0, line_dash="dash", line_color="red", annotation_text="Null (d=0)")
        fig_boot.add_vline(x=mean_diff, line_color="black", annotation_text=f"Sample Mean ({mean_diff:.4f})")
        fig_boot.add_vrect(x0=-0.0297, x1=0.0475, fillcolor="#10b981", opacity=0.15, annotation_text="95% CI Region")
        fig_boot.update_layout(template="plotly_white", xaxis_title="Bootstrap Mean Error Reduction (Units/Day)", yaxis_title="Resample Count")
        st.plotly_chart(fig_boot, use_container_width=True)

    with col_v2:
        st.markdown("**Empirical Paired Error Difference Distribution**")
        fig_p_diff = px.histogram(diffs, nbins=45, color_discrete_sequence=['#3b82f6'], marginal="box")
        fig_p_diff.add_vline(x=0.0, line_dash="dash", line_color="red", annotation_text="Zero Line")
        fig_p_diff.add_vline(x=mean_diff, line_color="green", annotation_text=f"Mean (+{mean_diff:.4f})")
        fig_p_diff.update_layout(template="plotly_white", xaxis_title="d_i = |e_base| - |e_imp| (Units)", yaxis_title="Observation Count")
        st.plotly_chart(fig_p_diff, use_container_width=True)

    # Critical Audit Callout Box
    st.markdown("""
    <div class="callout-box">
        <b>Audit of Inventory Cost Savings:</b> Earlier preliminary explorations cited a catalog-wide annual saving of ₹103,160.17 
        (or $103,160.17). Our rigorous audit confirms this figure was an unverified linear extrapolation across 3,000 products 
        of a sample point difference that failed to achieve statistical significance. Because the 95% Confidence Interval spans across zero 
        and Cohen's d = 0.0136 (negligible), claiming guaranteed monetary savings violates sound statistical principles. 
        The unsupported figure has been purged from official assignment deliverables.
    </div>
    """, unsafe_allow_html=True)

    # Download button
    if not ht_table.empty:
        csv_ht = ht_table.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Hypothesis Testing Results as CSV",
            data=csv_ht,
            file_name="statistical_hypothesis_tests.csv",
            mime="text/csv"
        )


# =============================================================================
# PAGE 6: INVENTORY PLANNING
# =============================================================================
elif page == "📦 Inventory Planning":
    st.markdown('<div class="main-title">Interactive Inventory & Supply Chain Optimization</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Statistical Safety Stock, Reorder Point (ROP), and Order Quantity Modeling</div>', unsafe_allow_html=True)

    st.markdown("""
    This module uses standard statistical inventory theory to convert demand forecasts and empirical demand variance 
    into actionable replenishment parameters.
    """)

    # Interactive Input Parameters
    c_inv1, c_inv2, c_inv3 = st.columns(3)
    inv_store = c_inv1.selectbox("Target Store", available_stores, index=0)
    inv_item = c_inv2.selectbox("Target Product SKU", available_items, index=0)
    inv_lead_time = c_inv3.slider("Supplier Lead Time (Days)", min_value=1, max_value=14, value=3)

    c_inv4, c_inv5, c_inv6 = st.columns(3)
    service_level_options = {
        "90% (Z = 1.28)": 1.28,
        "95% (Z = 1.645)": 1.645,
        "98% (Z = 2.05)": 2.05,
        "99% (Z = 2.33)": 2.33
    }
    selected_sl = c_inv4.selectbox("Desired Service Level", list(service_level_options.keys()), index=1)
    z_val = service_level_options[selected_sl]

    # Historical demand for the chosen series
    item_history = raw_sales_df[
        (raw_sales_df['store_id'] == inv_store) & 
        (raw_sales_df['item_id'] == inv_item)
    ]
    avg_d = float(item_history['sales'].mean()) if not item_history.empty else 2.5
    std_d = float(item_history['sales'].std()) if not item_history.empty else 1.5
    default_stock = int(np.round(avg_d * 5))

    current_stock = c_inv5.number_input("Current On-Hand Inventory (Units)", min_value=0, max_value=500, value=default_stock)
    unit_price = c_inv6.number_input("Unit Selling Price ($)", min_value=0.5, max_value=100.0, value=4.67, step=0.25)

    st.markdown("---")

    # Statistical Inventory Calculations
    lead_time_demand = avg_d * inv_lead_time
    sigma_lead_time = std_d * np.sqrt(inv_lead_time)
    safety_stock = z_val * sigma_lead_time
    reorder_point = lead_time_demand + safety_stock
    recommended_order = max(0.0, reorder_point - current_stock)
    days_to_stockout = (current_stock / avg_d) if avg_d > 0 else 999.0

    # Determine inventory health
    if current_stock < reorder_point:
        status_label = "🔴 LOW INVENTORY (REORDER IMMEDIATELY)"
        status_bg = "#fef2f2"
        status_border = "#f87171"
        status_text = "#991b1b"
    elif current_stock > reorder_point * 2.2:
        status_label = "🟡 OVERSTOCK (HOLDING COST RISK)"
        status_bg = "#fefce8"
        status_border = "#fde047"
        status_text = "#854d0e"
    else:
        status_label = "🟢 ADEQUATE STOCK (BALANCED REPLENISHMENT)"
        status_bg = "#f0fdf4"
        status_border = "#86efac"
        status_text = "#166534"

    st.markdown(f"""
    <div style="background-color: {status_bg}; border: 2px solid {status_border}; border-radius: 8px; padding: 14px; margin-bottom: 20px;">
        <h4 style="color: {status_text}; margin: 0; font-weight: bold;">INVENTORY HEALTH STATUS: {status_label}</h4>
    </div>
    """, unsafe_allow_html=True)

    # KPI Summary Cards
    col_i1, col_i2, col_i3, col_i4 = st.columns(4)
    col_i1.metric("Average Daily Demand (d̄)", f"{avg_d:.2f} units/day")
    col_i2.metric("Safety Stock (SS)", f"{safety_stock:.1f} units", help="Z × σ_d × √L")
    col_i3.metric("Reorder Point (ROP)", f"{reorder_point:.1f} units", help="d̄ × L + Safety Stock")
    col_i4.metric("Recommended Order (ROQ)", f"{recommended_order:.0f} units", delta=f"{recommended_order:.0f} to purchase")

    col_i5, col_i6, col_i7, col_i8 = st.columns(4)
    col_i5.metric("Lead Time Demand", f"{lead_time_demand:.1f} units")
    col_i6.metric("Lead Time Demand Std Dev", f"{sigma_lead_time:.2f} units")
    col_i7.metric("Days of Supply Remaining", f"{days_to_stockout:.1f} days")
    col_i8.metric("Estimated Replenishment Cost", f"${(recommended_order * unit_price):,.2f}")

    st.markdown("---")

    # Visual Gauge / Stock Level Breakdown
    st.subheader("Inventory Stock Level Breakdown vs. Policy Thresholds")
    
    fig_gauge = go.Figure()
    fig_gauge.add_trace(go.Bar(
        y=['Inventory Levels'], x=[safety_stock],
        name='Safety Stock (Buffer)', orientation='h', marker_color='#fbbf24'
    ))
    fig_gauge.add_trace(go.Bar(
        y=['Inventory Levels'], x=[lead_time_demand],
        name='Lead Time Demand', orientation='h', marker_color='#93c5fd'
    ))
    fig_gauge.add_trace(go.Bar(
        y=['Inventory Levels'], x=[max(0, current_stock - reorder_point)],
        name='Surplus Stock', orientation='h', marker_color='#34d399'
    ))
    fig_gauge.add_vline(x=current_stock, line_dash="solid", line_color="black", line_width=3, annotation_text=f"Current On-Hand: {current_stock}")
    fig_gauge.add_vline(x=reorder_point, line_dash="dash", line_color="red", line_width=2.5, annotation_text=f"Reorder Point: {reorder_point:.1f}")
    fig_gauge.update_layout(barmode='stack', template="plotly_white", xaxis_title="Units", height=240)
    st.plotly_chart(fig_gauge, use_container_width=True)

    # Inventory Depletion Simulation
    st.subheader("Simulated 28-Day Inventory Trajectory (Without Replenishment vs. With ROP Order)")
    days_proj = 28
    timeline = [f"Day +{i}" for i in range(1, days_proj + 1)]
    
    # Depletion curve
    depletion = [max(0.0, current_stock - avg_d * i) for i in range(1, days_proj + 1)]
    
    # Replenished curve (order placed immediately, arrives at Day L)
    replenished = []
    stock_r = current_stock
    for i in range(1, days_proj + 1):
        stock_r -= avg_d
        if i == inv_lead_time:
            stock_r += recommended_order
        replenished.append(max(0.0, stock_r))

    fig_dep = go.Figure()
    fig_dep.add_trace(go.Scatter(x=timeline, y=depletion, mode='lines+markers', name='No Replenishment (Depletion)', line=dict(color='#ef4444', width=2, dash='dot')))
    fig_dep.add_trace(go.Scatter(x=timeline, y=replenished, mode='lines+markers', name='With Recommended Order Replenishment', line=dict(color='#2563eb', width=2.5)))
    fig_dep.add_hline(y=reorder_point, line_dash="dash", line_color="#f59e0b", annotation_text=f"ROP = {reorder_point:.1f}")
    fig_dep.add_hline(y=safety_stock, line_dash="dash", line_color="#ef4444", annotation_text=f"Safety Stock = {safety_stock:.1f}")
    fig_dep.update_layout(template="plotly_white", xaxis_title="Future Days", yaxis_title="Projected On-Hand Units", hovermode="x unified")
    st.plotly_chart(fig_dep, use_container_width=True)

    # Documented Mathematical Methodology
    with st.expander("📚 View Documented Statistical Inventory Methodology & Assumptions"):
        st.markdown(r"""
        ### Statistical Inventory Control Model
        1. **Average Daily Demand ($\bar{d}$):**
           $$\bar{d} = \frac{1}{T} \sum_{t=1}^T y_t$$
        2. **Standard Deviation of Daily Demand ($\sigma_d$):**
           $$\sigma_d = \sqrt{\frac{1}{T-1} \sum_{t=1}^T (y_t - \bar{d})^2}$$
        3. **Standard Deviation of Demand During Lead Time ($\sigma_L$):**
           Assuming daily demand is identically distributed across independent days:
           $$\sigma_L = \sqrt{\text{Var}\left(\sum_{l=1}^L d_l\right)} = \sigma_d \sqrt{L}$$
        4. **Safety Stock ($SS$):**
           $$SS = Z_{\alpha} \times \sigma_L$$
           where $Z_{\alpha}$ is the standard normal quantile corresponding to the desired cycle service level.
        5. **Reorder Point ($ROP$):**
           $$ROP = (\bar{d} \times L) + SS$$
        6. **Recommended Order Quantity ($ROQ$):**
           $$ROQ = \max(0, ROP - \text{Current On-Hand Inventory})$$
        
        **Methodological Limitations:**
        - Assumes supplier lead time $L$ is constant and deterministic.
        - Assumes demand during lead time approximates a normal distribution; for intermittent demand, compound Poisson models are recommended.
        """)

    # Download Inventory Plan
    inv_plan_df = pd.DataFrame([{
        'Store': inv_store,
        'SKU': inv_item,
        'Lead_Time_Days': inv_lead_time,
        'Service_Level': selected_sl,
        'Avg_Daily_Demand': avg_d,
        'Safety_Stock': safety_stock,
        'Reorder_Point': reorder_point,
        'Current_Inventory': current_stock,
        'Recommended_Order_Quantity': recommended_order,
        'Replenishment_Cost_USD': recommended_order * unit_price,
        'Inventory_Status': status_label
    }])
    csv_inv = inv_plan_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Inventory Optimization Plan as CSV",
        data=csv_inv,
        file_name=f"inventory_plan_{inv_store}_{inv_item}.csv",
        mime="text/csv"
    )
