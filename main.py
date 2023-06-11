import time
import os
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog
from sklearn.preprocessing import MinMaxScaler
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ProgbarLogger, Callback
from joblib import Parallel, delayed
import mplfinance as mpf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.set_visible_devices(gpus[0], 'GPU')  # use GPU 0
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=15*1024)])  # limit to 15 GB
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    print(e)

class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

def plot_stock_chart(data):
    # Plot stock chart using mplfinance library
    mpf.plot(data, type='candle', volume=True, title='Stock Market Prices', ylabel='Price', ylabel_lower='Volume',
             show_nontrading=True, style='yahoo')

def remove_characters_and_convert_to_integer(value):
    if pd.isnull(value):
        return None

    if isinstance(value, str):
        characters_to_remove = [",", "-", ".", "/"]
        for character in characters_to_remove:
            value = value.replace(character, "")

        try:
            integer_value = int(value)
        except ValueError:
            raise ValueError("Invalid string value. Cannot convert to integer.")

    elif isinstance(value, float):
        integer_value = int(value)

    else:
        raise ValueError("Invalid value type. Must be a string or float.")

    return integer_value

def select_csv_file():
    # Open file dialog to select a CSV file
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if not file_path:
        raise ValueError("No file selected.")
    return file_path


def load_and_preprocess_data(filename, lookback):
    # Load and preprocess data from a CSV file
    data = pd.read_csv(filename)
    data = data.iloc[:, [0, 1, 6]]  # Select columns 0, 1, and 6
    data.columns = ['Date', 'Close', 'Volume']
    data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%y')
    data.sort_values(by='Date', ascending=True, inplace=True)
    data.reset_index(drop=True, inplace=True)
    data['Day'] = data['Date'].dt.day
    data['Month'] = data['Date'].dt.month
    data['Year'] = data['Date'].dt.year
    data['Close'] = data['Close'].apply(remove_characters_and_convert_to_integer)
    data['Volume'] = data['Volume'].apply(remove_characters_and_convert_to_integer)  # Remove dots and convert to integer
    data['Historical Close'] = data['Close'].shift(1)
    data.dropna(inplace=True)
    data['Historical Close'] = data['Historical Close'].shift(lookback)
    data.drop(data.head(lookback).index, inplace=True)

    # Drop rows containing NaN values
    data.dropna(inplace=True)

    scaler = MinMaxScaler(feature_range=(0, 1))
    data[['Close', 'Volume']] = scaler.fit_transform(data[['Close', 'Volume']])
    X, Y = [], []
    for i in range(len(data) - lookback - 7):
        # Construct feature matrix X and target variable Y
        X.append(np.column_stack((data['Day'].values[i:(i + lookback)],
                                  data['Month'].values[i:(i + lookback)],
                                  data['Year'].values[i:(i + lookback)],
                                  data['Volume'].values[i:(i + lookback)],
                                  data['Historical Close'].values[i:(i + lookback)])))
        Y.append(data['Close'].values[(i + lookback):(i + lookback + 7)])
    X = np.array(X)
    Y = np.array(Y)
    return scaler, X, Y

def build_model(lookback):
    # Build LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(lookback, 5)))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(7))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def train_model(model, train_X, train_Y, epochs=10, batch_size=32):
    # Train the LSTM model
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    time_callback = TimeHistory()
    tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=1)
    model.fit(train_X, train_Y, epochs=epochs, batch_size=batch_size, validation_split=0.2,
              callbacks=[early_stopping, time_callback, tensorboard_callback])
    return model, time_callback.times


def cross_validation_process(i, X, Y, lookback):
    # Perform cross-validation process for a given index
    train_X = np.delete(X, i, axis=0)
    train_Y = np.delete(Y, i, axis=0)
    test_X = X[i:i + 1]
    test_Y = Y[i:i + 1]
    model = build_model(lookback)
    model, times = train_model(model, train_X, train_Y)
    train_loss = model.evaluate(train_X, train_Y, verbose=0)
    test_loss = model.evaluate(test_X, test_Y, verbose=0)
    return train_loss, test_loss


def main():
    try:
        file_path = select_csv_file()
        lookback = 14
        scaler, X, Y = load_and_preprocess_data(file_path, lookback)
        train_losses = []
        test_losses = []

        # Perform Leave-One-Out Cross-Validation
        num_samples = X.shape[0]
        results = Parallel(n_jobs=-1)(
            delayed(cross_validation_process)(i, X, Y, lookback) for i in range(num_samples)
        )
        train_losses, test_losses = zip(*results)

        # Calculate average losses
        avg_train_loss = np.mean(train_losses)
        avg_test_loss = np.mean(test_losses)

        print("Average Train Loss:", avg_train_loss)
        print("Average Test Loss:", avg_test_loss)

    except Exception as e:
        print("An error occurred:", str(e))


if __name__ == '__main__':
    main()