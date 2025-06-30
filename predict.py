import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from sklearn.metrics import mean_squared_error

def make_predictions(model, X):
    """
    Make predictions using the trained model.
    Args:
        model: Trained Keras model.
        X: Input features.
    Returns:
        np.ndarray: Predicted values.
    """
    return model.predict(X)

def inverse_transform(scaler, data, feature_index=0):
    """
    Inverse transform the scaled data.
    Args:
        scaler: Fitted MinMaxScaler.
        data: Scaled data to inverse.
        feature_index: Index of the feature to inverse.
    Returns:
        np.ndarray: Inverse transformed data.
    """
    dummy = np.zeros((len(data), scaler.n_features_in_))
    dummy[:, feature_index] = data.flatten()
    return scaler.inverse_transform(dummy)[:, feature_index]

def calculate_rmse(y_true, y_pred):
    """
    Calculate RMSE between true and predicted values.
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))

def plot_predictions(y_true, y_pred, title='Actual vs Predicted'):
    """
    Plot actual vs predicted values using matplotlib.
    """
    plt.figure(figsize=(10,5))
    plt.plot(y_true, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.title(title)
    plt.legend()
    plt.show()

def plotly_predictions(y_true, y_pred, title='Actual vs Predicted'):
    """
    Plot actual vs predicted values using Plotly.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y_true, mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(y=y_pred, mode='lines', name='Predicted'))
    fig.update_layout(title=title, xaxis_title='Time', yaxis_title='Price')
    return fig 