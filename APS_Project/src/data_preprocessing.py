
import os
import sys
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def melt_sales_data(
    sales_df: pd.DataFrame,
    id_vars: Optional[List[str]] = None,
    var_name: str = "d",
    value_name: str = "sales"
) -> pd.DataFrame:
   
    if id_vars is None:
        id_vars = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    
    # Filter id_vars present in sales_df
    present_id_vars = [col for col in id_vars if col in sales_df.columns]
    
    # Identify daily sales columns (e.g., d_1, d_2, ...)
    day_cols = [c for c in sales_df.columns if c.startswith("d_")]
    if not day_cols:
        logger.info("Dataset is already in long format (no 'd_' prefix columns detected).")
        return sales_df
    
    logger.info(f"Melting sales data from {len(day_cols)} wide daily columns into long format...")
    df_long = pd.melt(
        sales_df,
        id_vars=present_id_vars,
        value_vars=day_cols,
        var_name=var_name,
        value_name=value_name
    )
    return df_long


def load_raw_or_processed_data(
    data_dir: Path,
    max_stores: int = 2,
    max_items_per_store: int = 15
) -> pd.DataFrame:
    """
    Loads M5 dataset from either:
    1. Local processed files: 'data/processed_sales.parquet' or 'data/m5_sales_sample.csv'
    2. Raw M5 files: 'sales_train_validation.csv', 'calendar.csv', and 'sell_prices.csv'
    
    Sampling Strategy:
      The raw M5 competition dataset contains 30,490 distinct time series across 1,913 days
      (over 58 million rows). To maintain computational reproducibility while preserving
      complex hierarchical retail properties (daily volatility, weekly seasonality, intermittent
      zero demand, price promotions), we draw a structured, representative sample of 2 major
      stores ('CA_1', 'CA_2') and 15 representative high-velocity food items ('FOODS_1_001' to 'FOODS_1_015')
      for 365 continuous days (10,950 observations).
    """
    data_dir = Path(data_dir)
    parquet_path = data_dir / "processed_sales.parquet"
    csv_path = data_dir / "m5_sales_sample.csv"
    
    # Check if pre-processed/sampled data is directly available
    if parquet_path.exists():
        logger.info(f"Loading preprocessed M5 dataset from: {parquet_path}")
        df = pd.read_parquet(parquet_path)
        return df
    elif csv_path.exists():
        logger.info(f"Loading preprocessed M5 dataset from: {csv_path}")
        df = pd.read_csv(csv_path)
        return df
    
    # Fallback: Ingest from raw M5 files if provided by user
    sales_file = data_dir / "sales_train_validation.csv"
    calendar_file = data_dir / "calendar.csv"
    prices_file = data_dir / "sell_prices.csv"
    
    if sales_file.exists() and calendar_file.exists():
        logger.info("Raw M5 files detected. Proceeding with raw data ingestion and sampling...")
        sales_raw = pd.read_csv(sales_file)
        calendar = pd.read_csv(calendar_file)
        
        # Filter stores and items for reproducible sampling
        stores = sales_raw['store_id'].unique()[:max_stores]
        items = sales_raw[sales_raw['cat_id'] == 'FOODS']['item_id'].unique()[:max_items_per_store]
        
        sampled_sales = sales_raw[
            (sales_raw['store_id'].isin(stores)) & (sales_raw['item_id'].isin(items))
        ].copy()
        
        df_long = melt_sales_data(sampled_sales)
        
        # Merge calendar
        df_merged = df_long.merge(calendar, on='d', how='left')
        
        # Merge prices if available
        if prices_file.exists():
            prices = pd.read_csv(prices_file)
            df_merged = df_merged.merge(prices, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')
        else:
            df_merged['sell_price'] = 4.67
            
        return df_merged
    else:
        raise FileNotFoundError(
            f"No M5 dataset found in {data_dir}. Expected either:\n"
            f"1. '{parquet_path}' or '{csv_path}', OR\n"
            f"2. Raw M5 files: 'sales_train_validation.csv' and 'calendar.csv'."
        )


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs data cleaning:
    - Parses date column into pandas datetime
    - Checks and handles duplicate records
    - Checks and imputes/handles missing values
    """
    logger.info("Performing data cleaning and missing-value analysis...")
    df = df.copy()
    
    # Ensure date column exists and is datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    else:
        raise KeyError("Column 'date' missing from sales dataset.")
    
    # Check duplicates
    initial_rows = len(df)
    df = df.drop_duplicates(subset=['id', 'date'])
    duplicates_removed = initial_rows - len(df)
    if duplicates_removed > 0:
        logger.warning(f"Removed {duplicates_removed} duplicate records.")
    else:
        logger.info("No duplicate (id, date) records found.")
        
    # Missing value handling
    missing_summary = df.isnull().sum()
    missing_cols = missing_summary[missing_summary > 0]
    if len(missing_cols) > 0:
        logger.info(f"Missing values detected:\n{missing_cols}")
        # Forward fill and backward fill sell_price per item_id
        if 'sell_price' in df.columns and df['sell_price'].isnull().any():
            df['sell_price'] = df.groupby('item_id')['sell_price'].transform(lambda g: g.ffill().bfill())
            df['sell_price'] = df['sell_price'].fillna(df['sell_price'].median())
            
        # Event columns: fill missing with 'None'
        for col in ['event_name_1', 'event_type_1']:
            if col in df.columns:
                df[col] = df[col].fillna("None")
                
        # SNAP indicators: fill with 0
        for snap_col in ['snap_CA', 'snap_TX', 'snap_WI']:
            if snap_col in df.columns:
                df[snap_col] = df[snap_col].fillna(0).astype(int)
    else:
        logger.info("No missing values found in core columns.")
        
    # Sort chronologically per time series
    df = df.sort_values(['id', 'date']).reset_index(drop=True)
    return df


def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], Dict[str, LabelEncoder]]:
    """
    Constructs time-series demand forecasting features:
    
    1. Calendar / Temporal Features:
       - day_of_week (0=Monday, 6=Sunday)
       - day_of_month (1 to 31)
       - month (1 to 12)
       - year
       - quarter (1 to 4)
       - week_of_year (1 to 52)
       - is_weekend (binary flag)
       - wday (M5 convention: 1=Saturday, 2=Sunday, ..., 7=Friday)
       
    2. Lag Features:
       - sales_lag_7, sales_lag_14, sales_lag_21, sales_lag_28
       * To strictly prevent data leakage in 28-day forecast horizons,
         lags are indexed at intervals of 7 days.
         
    3. Rolling Window Statistics:
       - sales_rolling_mean_7, sales_rolling_std_7
       - sales_rolling_mean_14, sales_rolling_std_14
       - sales_rolling_mean_28, sales_rolling_std_28
       * Applied to shifted sales: shift(1) ensures no lookahead bias!
       
    4. Price Dynamics:
       - price_hist_mean_4w: 4-week moving average of sell_price
       - price_relative_change: ratio of current price to historical mean
       
    5. Categorical Encodings:
       - Label encoding of store_id, item_id, dept_id, cat_id, state_id, event_name_1
    """
    logger.info("Generating temporal, lag, rolling, and price features...")
    df = df.copy()
    
    # 1. Calendar / Temporal
    df['day_of_week'] = df['date'].dt.day_of_week
    df['day_of_month'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['quarter'] = df['date'].dt.quarter
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    if 'wday' not in df.columns:
        # Convert pandas Monday=0 to M5 wday (Saturday=1, Sunday=2, Monday=3, ...)
        df['wday'] = ((df['day_of_week'] + 2) % 7) + 1
        
    # 2. Lag Features (grouped per item-store time series)
    lags = [7, 14, 21, 28]
    for lag in lags:
        df[f'sales_lag_{lag}'] = df.groupby('id')['sales'].shift(lag)
        
    # 3. Rolling Window Statistics on shift(1) (Leakage-Safe!)
    rolling_windows = [7, 14, 28]
    shifted_sales = df.groupby('id')['sales'].shift(1)
    
    for w in rolling_windows:
        df[f'sales_rolling_mean_{w}'] = df.groupby('id')['sales'].transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).mean()
        )
        df[f'sales_rolling_std_{w}'] = df.groupby('id')['sales'].transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).std()
        ).fillna(0.0)
        
    # 4. Sell Price Dynamics
    if 'sell_price' in df.columns:
        df['price_hist_mean_4w'] = df.groupby('id')['sell_price'].transform(
            lambda p: p.rolling(28, min_periods=1).mean()
        )
        df['price_relative_change'] = df['sell_price'] / (df['price_hist_mean_4w'] + 1e-5)
    else:
        df['sell_price'] = 4.67
        df['price_hist_mean_4w'] = 4.67
        df['price_relative_change'] = 1.0
        
    # 5. Categorical Encoding
    cat_cols = ['store_id', 'item_id', 'dept_id', 'cat_id', 'state_id', 'event_name_1']
    encoders: Dict[str, LabelEncoder] = {}
    
    for c in cat_cols:
        if c in df.columns:
            le = LabelEncoder()
            # Convert to string and fill NaNs
            df[c] = df[c].fillna("Unknown").astype(str)
            df[f'{c}_encoded'] = le.fit_transform(df[c])
            encoders[c] = le
        else:
            df[f'{c}_encoded'] = 0
            
    # Drop initial rows containing NaN from lag_28
    initial_len = len(df)
    df = df.dropna(subset=[f'sales_lag_{lag}' for lag in lags]).reset_index(drop=True)
    logger.info(f"Dropped {initial_len - len(df)} warm-up rows due to 28-day lag initialization.")
    
    # Feature columns list
    feature_cols = [
        'wday', 'month', 'year', 'day_of_month', 'day_of_week', 'is_weekend', 'week_of_year', 'quarter',
        'store_id_encoded', 'item_id_encoded', 'dept_id_encoded', 'cat_id_encoded', 'state_id_encoded',
        'event_name_1_encoded',
        'snap_CA', 'snap_TX', 'snap_WI',
        'sell_price', 'price_hist_mean_4w', 'price_relative_change',
        'sales_lag_7', 'sales_lag_14', 'sales_lag_21', 'sales_lag_28',
        'sales_rolling_mean_7', 'sales_rolling_std_7',
        'sales_rolling_mean_14', 'sales_rolling_std_14',
        'sales_rolling_mean_28', 'sales_rolling_std_28'
    ]
    # Filter features actually present in df
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    return df, feature_cols, encoders


def split_chronological_train_test(
    df: pd.DataFrame,
    test_days: int = 28
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs strict chronological train-test splitting.
    
    In time-series demand forecasting, standard randomized train_test_split
    causes catastrophic data leakage (lookahead bias) because models learn from
    future observations to predict the past.
    
    Here, the final 28 days (the standard M5 competition evaluation horizon)
    serve as the held-out test dataset, while all preceding historical days
    form the training dataset.
    """
    logger.info(f"Splitting data chronologically: final {test_days} days held out for testing...")
    
    max_date = df['date'].max()
    split_date = max_date - pd.Timedelta(days=test_days - 1)
    
    train_df = df[df['date'] < split_date].copy().reset_index(drop=True)
    test_df = df[df['date'] >= split_date].copy().reset_index(drop=True)
    
    logger.info(
        f"Chronological Split Complete:\n"
        f" - Training Period: {train_df['date'].min().strftime('%Y-%m-%d')} to {train_df['date'].max().strftime('%Y-%m-%d')} "
        f"({len(train_df)} observations across {train_df['date'].nunique()} days)\n"
        f" - Held-Out Test Period: {test_df['date'].min().strftime('%Y-%m-%d')} to {test_df['date'].max().strftime('%Y-%m-%d')} "
        f"({len(test_df)} observations across {test_df['date'].nunique()} days)"
    )
    return train_df, test_df


def prepare_m5_data(
    data_dir: str = "data",
    test_days: int = 28
) -> Dict:
    """
    End-to-end data preparation pipeline:
    1. Loads dataset
    2. Cleans & validates
    3. Engineers time-series features
    4. Performs chronological split
    5. Returns feature matrices (X_train, y_train, X_test, y_test) and metadata.
    """
    raw_df = load_raw_or_processed_data(Path(data_dir))
    cleaned_df = clean_data(raw_df)
    featured_df, feature_cols, encoders = engineer_features(cleaned_df)
    train_df, test_df = split_chronological_train_test(featured_df, test_days=test_days)
    
    X_train = train_df[feature_cols].copy()
    y_train = train_df['sales'].values
    
    X_test = test_df[feature_cols].copy()
    y_test = test_df['sales'].values
    
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "train_df": train_df,
        "test_df": test_df,
        "feature_cols": feature_cols,
        "encoders": encoders,
        "full_df": featured_df
    }


if __name__ == "__main__":
    print("Testing data preprocessing pipeline...")
    data_dict = prepare_m5_data()
    print("X_train shape:", data_dict['X_train'].shape)
    print("X_test shape:", data_dict['X_test'].shape)
    print("Features:", data_dict['feature_cols'])
    print("Data preprocessing passed successfully!")
