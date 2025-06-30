import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def build_lstm_model(input_shape):
    """
    Build and compile an LSTM model.
    Args:
        input_shape (tuple): Shape of input data (timesteps, features).
    Returns:
        model: Compiled Keras model.
    """
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer=Adam(), loss='mse')
    return model

def train_lstm_model(model, X_train, y_train, epochs=20, batch_size=32, validation_data=None):
    """
    Train the LSTM model.
    Args:
        model: Compiled Keras model.
        X_train: Training features.
        y_train: Training targets.
        epochs (int): Number of epochs.
        batch_size (int): Batch size.
        validation_data: Tuple (X_val, y_val) for validation.
    Returns:
        History object from model.fit().
    """
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=validation_data,
        verbose=1
    )
    return history 