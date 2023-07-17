import os
import re
import subprocess
import numpy as np
import pandas as pd
import tensorflow as tf
from glob import glob
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from sklearn.preprocessing import MinMaxScaler
import mplfinance as mpf

CHECKPOINT_DIR = "PATH_TO_CHECKPOINT"

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


def select_gpu_with_most_memory():
    gpus = tf.config.list_physical_devices('GPU')
    print(f"Num GPUs Available: {len(gpus)}")

    if not gpus:
        return

    output = subprocess.check_output("nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits", shell=True)
    memories = [int(x) for x in output.decode('utf-8').strip().split('\n')]
    gpu_most_memory = np.argmax(memories)

    try:
        tf.config.set_visible_devices(gpus[gpu_most_memory], 'GPU')
        tf.config.experimental.set_virtual_device_configuration(gpus[gpu_most_memory],
                                                                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=15*1024)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
    except RuntimeError as e:
        print(e)


def plot_stock_chart(data):
    mpf.plot(data, type='candle', volume=True, title='Stock Market Prices', ylabel='Price', ylabel_lower='Volume',
             show_nontrading=True, style='yahoo')


def remove_characters_and_convert_to_integer(value):
    if pd.isnull(value):
        return None

    if isinstance(value, (str, float)):
        value = re.sub(r"[,.-/]", "", str(value))
        try:
            return int(value)
        except ValueError:
            raise ValueError("Invalid string value. Cannot convert to integer.")

    raise ValueError("Invalid value type. Must be a string or float.")


def select_csv_file():
    file_path = input("Please enter the path to the CSV file: ")
    if not os.path.isfile(file_path):
        raise ValueError("No such file found.")
    return file_path


def load_and_preprocess_data(filename, lookback):
    data = pd.read_csv(filename)
    data = data.iloc[:, [0, 1, 2, 3, 4, 6]]  # Select columns 0, 1, 2, 3, 4 and 6
    data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%y')
    data.sort_values(by='Date', ascending=True, inplace=True)
    data.reset_index(drop=True, inplace=True)
    data['Day'] = data['Date'].dt.day
    data['Month'] = data['Date'].dt.month
    data['Year'] = data['Date'].dt.year
    data[['Open', 'High', 'Low', 'Close', 'Volume']] = data[['Open', 'High', 'Low', 'Close', 'Volume']].applymap(remove_characters_and_convert_to_integer)
    data['Historical Close'] = data['Close'].shift(1)
    data.dropna(inplace=True)
    data['Historical Close'] = data['Historical Close'].shift(lookback)
    data.drop(data.head(lookback).index, inplace=True)
    data.dropna(inplace=True)

    scaler = MinMaxScaler(feature_range=(0, 1))
    data[['Open', 'High', 'Low', 'Close', 'Volume']] = scaler.fit_transform(data[['Open', 'High', 'Low', 'Close', 'Volume']])

    X, Y = [], []
    for i in range(len(data) - lookback - 7):
        X.append(np.column_stack((data['Day'].values[i:(i + lookback)],
                                  data['Month'].values[i:(i + lookback)],
                                  data['Year'].values[i:(i + lookback)],
                                  data['Open'].values[i:(i + lookback)],
                                  data['High'].values[i:(i + lookback)],
                                  data['Low'].values[i:(i + lookback)],
                                  data['Volume'].values[i:(i + lookback)],
                                  data['Historical Close'].values[i:(i + lookback)])))
        Y.append(data['Close'].values[(i + lookback):(i + lookback + 7)])

    return scaler, np.array(X), np.array(Y)


def build_model(lookback, l2_factor=0.0085):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, kernel_regularizer=l2(l2_factor), input_shape=(lookback, 8)))  # Updated input shape
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=True, kernel_regularizer=l2(l2_factor)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=True, kernel_regularizer=l2(l2_factor)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=True, kernel_regularizer=l2(l2_factor)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(LSTM(64, kernel_regularizer=l2(l2_factor)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(7))
    model.compile(loss='mean_squared_error', optimizer='adam')

    return model


def get_user_input(prompt, valid_values):
    while True:
        user_input = input(prompt).lower()
        if user_input in valid_values:
            return user_input
        print(f"Invalid input. Please enter one of {valid_values}.")


def get_existing_directory(prompt):
    while True:
        dir_path = input(prompt)
        if os.path.isdir(dir_path):
            return dir_path
        print("Invalid directory path. Please enter a valid path.")


def get_latest_checkpoint_file(checkpoint_dir):
    list_of_files = glob(os.path.join(checkpoint_dir, '*.hdf5'))
    return max(list_of_files, key=os.path.getctime)


def train_model(train_X, train_Y, val_X, val_Y, lookback, checkpoint_dir):
    if checkpoint_dir:
        checkpoint_file = get_latest_checkpoint_file(checkpoint_dir)
        print(f"Loading model from {checkpoint_file}...")
        model = tf.keras.models.load_model(checkpoint_file)
    else:
        print("User chose not to use checkpoints. Building a new model...")
        model = build_model(lookback)

    lr_schedule = ExponentialDecay(
        initial_learning_rate=1e-2,
        decay_steps=10000,
        decay_rate=0.9)

    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule))

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True)

    time_callback = TimeHistory()

    callbacks = [time_callback, early_stopping]

    if checkpoint_dir:
        checkpoint_callback = ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, 'model_{epoch:02d}.hdf5'),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,  # Save the entire model
            verbose=1)
        callbacks.append(checkpoint_callback)

    model.fit(train_X, train_Y, epochs=50, batch_size=128, validation_data=(val_X, val_Y), verbose=1, callbacks=callbacks)
    train_loss = model.evaluate(train_X, train_Y, verbose=1)
    return train_loss, model, time_callback.times


def time_series_cross_validation_process(X, Y, lookback, checkpoint_dir):
    train_losses = []
    validation_losses = []
    models = []
    training_times = []

    print("Starting time series cross-validation process...")
    for i in range(45, X.shape[0], 45):
        print(f"Training model {i + 1} of {X.shape[0]}...")

        train_X = X[:i]
        train_Y = Y[:i]
        val_X = X[i:i + 45]
        val_Y = Y[i:i + 45]

        train_loss, model, training_time = train_model(train_X, train_Y, val_X, val_Y, lookback, checkpoint_dir)
        val_loss = model.evaluate(val_X, val_Y, verbose=1)  # Calculate validation loss

        train_losses.append(train_loss)
        validation_losses.append(val_loss)  # Store validation loss
        models.append(model)
        training_times.append(training_time)

        print(f"Finished training model {i + 1}. Train loss: {train_loss}, Validation loss: {val_loss}, Training time: {sum(training_time)} seconds.")

    print("Time series cross-validation process completed.")
    return train_losses, validation_losses, models, training_times


def main():
    print("Starting the script...")
    select_gpu_with_most_memory()

    lookback = 14  # use past 14 days data to predict next 7 days
    filename = select_csv_file()
    print("Loading and preprocessing data...")
    scaler, X, Y = load_and_preprocess_data(filename, lookback)

    start_from_checkpoint = get_user_input("Do you want to start from a checkpoint? (yes/no): ", {"yes", "no"})
    checkpoint_dir = get_existing_directory("Please enter the path of the checkpoint directory: ") if start_from_checkpoint == "yes" else None

    if start_from_checkpoint == "yes":
        checkpoint_file = get_latest_checkpoint_file(checkpoint_dir)
        print(f"Loading model from {checkpoint_file}...")
        model = tf.keras.models.load_model(checkpoint_file)
    else:
        print("User chose not to use checkpoints. Training a new model...")
        train_losses, validation_losses, models, training_times = time_series_cross_validation_process(X, Y, lookback, checkpoint_dir)
        best_model_index = np.argmin(train_losses)
        model = models[best_model_index]
        print(f"Best model selected with a training loss of {train_losses[best_model_index]}.")

    if get_user_input("Do you want to save the best model? (yes/no): ", {"yes", "no"}) == "yes":
        model_filename = input("Please enter the filename to save the model: ")
        model.save(model_filename)
        print(f"Model saved as {model_filename}.")


if __name__ == "__main__":
    main()
