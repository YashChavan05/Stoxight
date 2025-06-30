import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def predict_future(model, scaler, last_data, n_days=7, feature_col='Close'):
    """
    Predict future stock prices for the next n_days.
    Args:
        model: Trained LSTM model.
        scaler: Fitted MinMaxScaler.
        last_data: Last available data point.
        n_days: Number of days to predict.
        feature_col: Column name for the feature.
    Returns:
        tuple: (dates, predicted_prices)
    """
    predictions = []
    current_data = last_data.copy()
    
    for i in range(n_days):
        # Prepare input for prediction
        X_pred = current_data.reshape((1, 1, -1))
        
        # Make prediction
        pred = model.predict(X_pred, verbose=0)
        pred_price = inverse_transform(scaler, pred)
        predictions.append(pred_price[0])
        
        # Update data for next prediction (shift and add new prediction)
        # This is a simplified approach - in practice, you might want to update all features
        current_data = np.roll(current_data, -1)
        current_data[-1] = pred[0, 0]
    
    # Generate future dates
    last_date = pd.Timestamp.now().date()
    future_dates = [last_date + timedelta(days=i+1) for i in range(n_days)]
    
    return future_dates, predictions

def inverse_transform(scaler, data, feature_index=0):
    """
    Inverse transform the scaled data.
    """
    dummy = np.zeros((len(data), scaler.n_features_in_))
    dummy[:, feature_index] = data.flatten()
    return scaler.inverse_transform(dummy)[:, feature_index] 