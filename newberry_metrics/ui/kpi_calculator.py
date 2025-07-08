"""
KPI calculation functions for Newberry Metrics Dashboard.
"""

import pandas as pd
from typing import Dict, Any


def calculate_kpis(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate key performance indicators from the metrics DataFrame."""
    if df.empty:
        return {
            'avg_cost': 0.0,
            'total_cost': 0.0,
            'avg_latency': 0.0,
            'total_latency': 0.0
        }
    
    return {
        'avg_cost': df['cost'].mean(),
        'total_cost': df['cost'].sum(),
        'avg_latency': df['latency'].mean(),
        'total_latency': df['latency'].sum()
    }


def prepare_hourly_data(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Prepare data aggregated by hour for visualization."""
    if df.empty:
        return {}
    
    # Ensure timestamp is datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df_hour = df.copy()
    df_hour['hour'] = df_hour['timestamp'].dt.strftime('%Y-%m-%d %H:00')
    
    # Aggregate data
    hourly_cost = df_hour.groupby('hour')['cost'].sum().reset_index()
    hourly_latency = df_hour.groupby('hour')['latency'].mean().reset_index()
    
    # Calculate hourly input-output ratio
    hourly_io = df_hour.groupby('hour').agg({
        'input_tokens': 'sum',
        'output_tokens': 'sum'
    }).reset_index()
    # Add a check for division by zero if input_tokens can be zero
    hourly_io['io_ratio'] = hourly_io.apply(
        lambda row: row['output_tokens'] / row['input_tokens'] if row['input_tokens'] else 0, 
        axis=1
    )
    
    return {
        'cost': hourly_cost,
        'latency': hourly_latency,
        'io_tokens': hourly_io
    }


def prepare_daily_data(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Prepare data aggregated by day for visualization."""
    if df.empty:
        return {}
    
    # Ensure timestamp is datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df_day = df.copy()
    df_day['day'] = df_day['timestamp'].dt.strftime('%Y-%m-%d')
    
    # Aggregate data
    daily_cost = df_day.groupby('day')['cost'].sum().reset_index()
    daily_latency = df_day.groupby('day')['latency'].mean().reset_index()
    
    # Calculate daily input-output ratio
    daily_io = df_day.groupby('day').agg({
        'input_tokens': 'sum',
        'output_tokens': 'sum'
    }).reset_index()
    daily_io['io_ratio'] = daily_io.apply(
        lambda row: row['output_tokens'] / row['input_tokens'] if row['input_tokens'] else 0, 
        axis=1
    )
    
    return {
        'cost': daily_cost,
        'latency': daily_latency,
        'io_tokens': daily_io
    }
