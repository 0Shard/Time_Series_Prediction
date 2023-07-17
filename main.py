import time
import json
import os
import numpy as np
import subprocess
import re
import pandas as pd
import glob
from sklearn.preprocessing import MinMaxScaler
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ProgbarLogger, Callback
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint
import mplfinance as mpf


CHECKPOINT_DIR = "PATH_TO_CHECKPOINT"

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

def build_model(lookback, l2_factor=0.0085):
    # Build LSTM model
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, kernel_regularizer=l2(l2_factor), input_shape=(lookback, 5)))
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


def ask_user_to_start_from_checkpoint():
    # Ask the user whether they want to start from a checkpoint
    while True:
        user_input = input("Do you want to start from a checkpoint? (yes/no): ")
        if user_input.lower() == "yes":
            return True
        elif user_input.lower() == "no":
            return False
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")

def ask_user_for_checkpoint_dir():
    # Ask the user to enter the path of the checkpoint directory
    while True:
        checkpoint_dir = input("Please enter the path of the checkpoint directory: ")
        if os.path.isdir(checkpoint_dir):
            return checkpoint_dir
        else:
            print("Invalid directory path. Please enter a valid path.")


def get_latest_checkpoint_file(checkpoint_dir):
    list_of_files = glob.glob(os.path.join(checkpoint_dir, '*.hdf5')) # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file


def train_model(train_X, train_Y, lookback, checkpoint_dir):
    if checkpoint_dir:
        checkpoint_file = get_latest_checkpoint_file(checkpoint_dir)
        print(f"Loading model from {checkpoint_file}...")
        model = tf.keras.models.load_model(checkpoint_file)
    else:
        print("User chose not to use checkpoints. Building a new model...")
        model = build_model(lookback)

    callbacks = [time_callback, early_stopping]

    # Add ModelCheckpoint callback if checkpoint_dir is not None
    if checkpoint_dir:
        checkpoint_callback = ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, 'model_{epoch:02d}.hdf5'),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,  # Save the entire model
            verbose=1)
        callbacks.append(checkpoint_callback)

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
    model.fit(train_X, train_Y, epochs=50, batch_size=128, validation_split=0.2, verbose=1, callbacks=callbacks)
    train_loss = model.evaluate(train_X, train_Y, verbose=1)
    return train_loss, model, time_callback.times


def time_series_cross_validation_process(X, Y, lookback, checkpoint_dir):
    train_losses = []
    validation_losses = []  # Added this line
    models = []
    training_times = []

    print("Starting time series cross-validation process...")
    # Change this to the number of iterations you want for the cross-validation
    for i in range(10, X.shape[0], 10):
        print(f"Training model {i + 1} of {X.shape[0]}...")

        # Splitting data into training and validation sets
        train_X = X[:i]
        train_Y = Y[:i]
        val_X = X[i:i + 10]
        val_Y = Y[i:i + 10]

        train_loss, model, training_time = train_model(train_X, train_Y, lookback, checkpoint_dir)
        val_loss = model.evaluate(val_X, val_Y, verbose=1)  # Calculate validation loss

        train_losses.append(train_loss)
        validation_losses.append(val_loss)  # Store validation loss
        models.append(model)
        training_times.append(training_time)

        print(
            f"Finished training model {i + 1}. Train loss: {train_loss}, Validation loss: {val_loss}, Training time: {sum(training_time)} seconds.")

    print("Time series cross-validation process completed.")
    return train_losses, validation_losses, models, training_times

def ask_user_to_train_new_model():
    # Ask the user whether they want to train a new model or use a pre-trained one
    while True:
        user_input = input("Do you want to train a new model? (yes/no): ")
        if user_input.lower() == "yes":
            return True
        elif user_input.lower() == "no":
            return False
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")

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

    checkpoint_dir = None
    if ask_user_to_start_from_checkpoint():
        checkpoint_dir = ask_user_for_checkpoint_dir()

    if ask_user_to_train_new_model():
        # Training a new model
        train_losses, validation_losses, models, training_times = time_series_cross_validation_process(X, Y, lookback, checkpoint_dir)
        best_model_index = np.argmin(train_losses)
        best_model = models[best_model_index]
        print(f"Best model selected with a training loss of {train_losses[best_model_index]}.")
    else:
        # Using a pre-trained model
        # Load the model with the smallest validation loss
        losses = []
        for i in range(X.shape[0]):
            with open(os.path.join(checkpoint_dir, f'loss_{i:03d}.json'), 'r') as f:
                losses.append(json.load(f)['loss'])
        min_loss_index = np.argmin(losses)
        CHECKPOINT_FILE = os.path.join(checkpoint_dir, f'model_{min_loss_index:03d}.hdf5')
        best_model = tf.keras.models.load_model(CHECKPOINT_FILE)
        print(f"Pre-trained model loaded with a training loss of {losses[min_loss_index]}.")

    if ask_user_to_save_model():
        save_model(best_model)
    print("Finished the script.")

if __name__ == "__main__":
    main()