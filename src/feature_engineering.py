"""
Feature Engineering Utilities

Functions for creating new features and transforming existing ones.
"""

import pandas as pd
import numpy as np
from typing import List


def create_rate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create rate-based features (packets/bytes per time unit).
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with new rate features
    """
    df_eng = df.copy()
    
    # Packet rates (avoid division by zero)
    if 'spkts' in df.columns and 'dur' in df.columns:
        df_eng['spkts_rate'] = df_eng['spkts'] / (df_eng['dur'] + 1e-6)
        df_eng['dpkts_rate'] = df_eng['dpkts'] / (df_eng['dur'] + 1e-6)
    
    # Byte rates
    if 'sbytes' in df.columns and 'dur' in df.columns:
        df_eng['sbytes_rate'] = df_eng['sbytes'] / (df_eng['dur'] + 1e-6)
        df_eng['dbytes_rate'] = df_eng['dbytes'] / (df_eng['dur'] + 1e-6)
    
    return df_eng


def create_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create ratio-based features comparing source and destination metrics.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with new ratio features
    """
    df_eng = df.copy()
    
    # Byte ratios
    if 'sbytes' in df.columns and 'dbytes' in df.columns:
        df_eng['byte_ratio'] = df_eng['sbytes'] / (df_eng['dbytes'] + 1)
    
    # Packet ratios
    if 'spkts' in df.columns and 'dpkts' in df.columns:
        df_eng['pkt_ratio'] = df_eng['spkts'] / (df_eng['dpkts'] + 1)
    
    return df_eng


def create_aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create aggregate features combining multiple columns.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with new aggregate features
    """
    df_eng = df.copy()
    
    # Total traffic volume
    if 'sbytes' in df.columns and 'dbytes' in df.columns:
        df_eng['total_bytes'] = df_eng['sbytes'] + df_eng['dbytes']
    
    if 'spkts' in df.columns and 'dpkts' in df.columns:
        df_eng['total_pkts'] = df_eng['spkts'] + df_eng['dpkts']
    
    # Average packet size
    if 'total_bytes' in df_eng.columns and 'total_pkts' in df_eng.columns:
        df_eng['avg_pkt_size'] = df_eng['total_bytes'] / (df_eng['total_pkts'] + 1)
    
    return df_eng


def create_time_features(df: pd.DataFrame, time_column: str = 'stime') -> pd.DataFrame:
    """
    Create time-based features from timestamp columns.
    
    Args:
        df: Input DataFrame
        time_column: Name of timestamp column
        
    Returns:
        DataFrame with new time features
    """
    df_eng = df.copy()
    
    if time_column in df.columns:
        # Convert to datetime
        df_eng[time_column] = pd.to_datetime(df_eng[time_column], unit='s')
        
        # Extract time components
        df_eng['hour'] = df_eng[time_column].dt.hour
        df_eng['day_of_week'] = df_eng[time_column].dt.dayofweek
        
        # Binary indicators
        df_eng['is_night'] = df_eng['hour'].apply(lambda x: 1 if x >= 22 or x <= 6 else 0)
        df_eng['is_weekend'] = df_eng['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    return df_eng


def apply_log_transform(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Apply log transformation to skewed features.
    
    Args:
        df: Input DataFrame
        columns: List of columns to transform
        
    Returns:
        DataFrame with log-transformed features
    """
    df_trans = df.copy()
    
    for col in columns:
        if col in df_trans.columns:
            # Use log1p to handle zeros
            df_trans[f'{col}_log'] = np.log1p(df_trans[col])
    
    return df_trans


def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering transformations.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with all engineered features
    """
    df_eng = df.copy()
    
    # Apply all transformations
    df_eng = create_rate_features(df_eng)
    df_eng = create_ratio_features(df_eng)
    df_eng = create_aggregate_features(df_eng)
    
    # Optional: time features (if timestamp exists)
    # df_eng = create_time_features(df_eng)
    
    print(f"Original features: {df.shape[1]}")
    print(f"After engineering: {df_eng.shape[1]}")
    print(f"New features added: {df_eng.shape[1] - df.shape[1]}")
    
    return df_eng
