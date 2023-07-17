import time
import os
import numpy as np
import subprocess
import re
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ProgbarLogger, Callback
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.regularizers import l2
import mplfinance as mpf


def select_gpu_with_most_memory():

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    # get the memory usage information
    output = subprocess.check_output("nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits", shell=True)

    # parse the output to get the memory for each GPU
    memories = [int(x) for x in output.decode('utf-8').strip().split('\n')]

    # get the GPU with the most memory
    gpu_most_memory = np.argmax(memories)

    # now set TensorFlow to use this GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # use the GPU with the most memory
            tf.config.set_visible_devices(gpus[gpu_most_memory], 'GPU')
            tf.config.experimental.set_virtual_device_configuration(gpus[gpu_most_memory], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=15*1024)])
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
    # Ask user to enter a CSV file path
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



def train_model(train_X, train_Y, lookback, checkpoint_dir, validation_run_number, checkpoint_file=None):
    if checkpoint_file is not None:
        print(f"Loading model from {checkpoint_file}...")
        model = tf.keras.models.load_model(checkpoint_file)
    else:
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

    # Add a ModelCheckpoint callback
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, f"{validation_run_number}.h5"),
        save_best_only=False,
        verbose=1)

    model.fit(train_X, train_Y, epochs=50, batch_size=128, validation_split=0.2, verbose=1, callbacks=[time_callback, early_stopping, model_checkpoint])
    train_loss = model.evaluate(train_X, train_Y, verbose=1)
    return train_loss, model, time_callback.times


def rolling_window_validation_process(X, Y, lookback, window_size, checkpoint_dir, checkpoint_file=None):
    train_losses = []
    models = []
    training_times = []
    test_X, test_Y = None, None

    print("Starting rolling window validation process...")
    for i in range(lookback, X.shape[0] - window_size):
        print(f"Training model {i + 1} of {X.shape[0] - window_size}...")
        train_X = X[:i]
        train_Y = Y[:i]
        test_X = X[i:i + window_size]
        test_Y = Y[i:i + window_size]

        # Calculate the validation run number
        validation_run_number = (i - lookback) // window_size + 1
        train_loss, model, training_time = train_model(train_X, train_Y, lookback, checkpoint_dir,
                                                       validation_run_number, checkpoint_file)
        train_losses.append(train_loss)
        models.append(model)
        training_times.append(training_time)
        print(
            f"Finished training model {i + 1}. Train loss: {train_loss}, Training time: {sum(training_time)} seconds.")

    print("Rolling window validation process completed.")
    return train_losses, models, training_times, test_X, test_Y

def save_model(model):
    # Ask user where to save the LSTM model
    while True:
        # Ask user to enter a path and filename.
        file_path = input("Please enter the path to save the model (with .h5 extension): ")

        if not file_path:
            raise ValueError("No location selected.")

        model_name = os.path.basename(file_path)
        model_name_no_ext = os.path.splitext(model_name)[0]

        if not re.match("^[a-z0-9_]+$", model_name_no_ext):
            print("Invalid model name. Use only lower case letters, digits, and underscores.")
            continue  # if filename is invalid, ask again

        # save the model to the entered path
        model.save(file_path)
        print(f"Model saved at location : {file_path}")
        break  # if filename is valid, break the loop

def ask_user_to_save_model():
    # Ask the user whether they want to save the model
    while True:
        user_input = input("Do you want to save the model? (yes/no): ")
        if user_input.lower() == "yes":
            return True
        elif user_input.lower() == "no":
            return False
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")


def main():
    print("Starting the script...")
    select_gpu_with_most_memory()
    # Run the script
    lookback = 14  # use past 14 days data to predict next 7 days
    filename = select_csv_file()
    print("Loading and preprocessing data...")
    scaler, X, Y = load_and_preprocess_data(filename, lookback)

    # Get the checkpoint directory from the user
    checkpoint_dir = input("Please enter the directory to save the checkpoints: ")

    # Ask the user if they want to start from a checkpoint
    use_checkpoint = input("Do you want to start from a checkpoint? (yes/no): ").lower() == "yes"
    checkpoint_file = None
    if use_checkpoint:
        checkpoint_file = input("Please enter the path to the checkpoint file: ")

    train_losses, models, training_times, test_X, test_Y = rolling_window_validation_process(X, Y, lookback, 7,
                                                                                             checkpoint_dir,
                                                                                             checkpoint_file)
    best_model_index = np.argmin(train_losses)
    best_model = models[best_model_index]
    print(f"Best model selected with a training loss of {train_losses[best_model_index]}.")
    if ask_user_to_save_model():
        save_model(best_model)
    print("Finished the script.")

if __name__ == "__main__":
    main()
