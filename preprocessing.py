import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(df: pd.DataFrame, feature_col: str = 'Close', n_lags: int = 1, add_ma: bool = True, ma_windows: list = [5, 10]) -> tuple:
    """
    Preprocess stock data: handle missing values, normalize, create lag features, and moving averages.
    Args:
        df (pd.DataFrame): Raw stock data.
        feature_col (str): Column to predict (default 'Close').
        n_lags (int): Number of lag days to use as features.
        add_ma (bool): Whether to add moving averages.
        ma_windows (list): List of window sizes for moving averages.
    Returns:
        tuple: (processed DataFrame, scaler)
    """
    df = df.copy()
    df = df[[feature_col]].fillna(method='ffill').dropna()
    if add_ma:
        for window in ma_windows:
            df[f'MA_{window}'] = df[feature_col].rolling(window=window).mean()
    for lag in range(1, n_lags+1):
        df[f'lag_{lag}'] = df[feature_col].shift(lag)
    df = df.dropna()
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(scaled, columns=df.columns, index=df.index)
    return df_scaled, scaler 