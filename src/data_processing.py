"""
Data Processing Utilities

Functions for loading, cleaning, and preprocessing the UNSW-NB15 dataset.
"""

import pandas as pd
import numpy as np
import glob
from typing import Tuple, List, Optional


def load_unsw_data(data_dir: str = '../data') -> pd.DataFrame:
    """
    Load all UNSW-NB15 CSV files and combine them into a single DataFrame.
    
    Args:
        data_dir: Directory containing UNSW-NB15 CSV files
        
    Returns:
        Combined DataFrame with all records
    """
    # TODO: Implement data loading logic
    data_files = glob.glob(f'{data_dir}/UNSW-NB15_*.csv')
    if not data_files:
        raise FileNotFoundError(f"No UNSW-NB15 CSV files found in {data_dir}")
    
    dfs = [pd.read_csv(file) for file in sorted(data_files)]
    df = pd.concat(dfs, ignore_index=True)
    
    print(f"Loaded {len(df):,} records from {len(data_files)} files")
    return df


def handle_missing_values(df: pd.DataFrame, 
                          threshold: float = 0.5,
                          strategy: str = 'median') -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    
    Args:
        df: Input DataFrame
        threshold: Drop columns with missing percentage > threshold
        strategy: Imputation strategy ('median', 'mean', 'mode')
        
    Returns:
        DataFrame with missing values handled
    """
    # TODO: Implement missing value handling
    df_clean = df.copy()
    
    # Drop columns with too many missing values
    missing_pct = df_clean.isnull().sum() / len(df_clean)
    cols_to_drop = missing_pct[missing_pct > threshold].index
    df_clean = df_clean.drop(columns=cols_to_drop)
    
    # Impute remaining missing values
    # Implementation depends on strategy
    
    return df_clean


def detect_outliers_iqr(df: pd.DataFrame, 
                        column: str,
                        multiplier: float = 1.5) -> Tuple[pd.Series, float, float]:
    """
    Detect outliers using Interquartile Range (IQR) method.
    
    Args:
        df: Input DataFrame
        column: Column name to analyze
        multiplier: IQR multiplier (default 1.5)
        
    Returns:
        Tuple of (outlier_mask, lower_bound, upper_bound)
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    outlier_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
    
    return outlier_mask, lower_bound, upper_bound


def cap_outliers(df: pd.DataFrame,
                 columns: List[str],
                 lower_percentile: float = 0.01,
                 upper_percentile: float = 0.99) -> pd.DataFrame:
    """
    Cap outliers using Winsorization (percentile clipping).
    
    Args:
        df: Input DataFrame
        columns: List of columns to cap
        lower_percentile: Lower percentile for capping
        upper_percentile: Upper percentile for capping
        
    Returns:
        DataFrame with capped outliers
    """
    df_capped = df.copy()
    
    for col in columns:
        if col in df_capped.columns:
            lower_cap = df_capped[col].quantile(lower_percentile)
            upper_cap = df_capped[col].quantile(upper_percentile)
            df_capped[col] = df_capped[col].clip(lower=lower_cap, upper=upper_cap)
    
    return df_capped


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Generate comprehensive data summary statistics.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        'n_records': len(df),
        'n_features': df.shape[1],
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
        'dtypes': df.dtypes.value_counts().to_dict()
    }
    
    return summary
