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
from sklearn.model_selection import train_test_split
import datetime
import matplotlib.pyplot as plt


# Create Log Directory
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Create TensorBoard Callback
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)


def setup_gpu_for_training(gpu_id=0, memory_limit_gb=15):
    """
    Set up GPU for training with specified memory limit.

    Args:
    - gpu_id (int): ID of the GPU to be used.
    - memory_limit_gb (int): Memory limit in gigabytes. Default is 15GB.

    Returns:
    - None
    """
    gpus = tf.config.list_physical_devices('GPU')

    if not gpus:
        print("No GPUs available. Using CPU.")
        return

    if gpu_id >= len(gpus):
        raise ValueError(f"GPU:{gpu_id} not available. Maximum GPU ID is {len(gpus) - 1}.")

    # Convert memory limit to megabytes
    memory_limit_mb = memory_limit_gb * 1024

    # Limit GPU memory usage
    tf.config.set_logical_device_configuration(
        gpus[gpu_id],
        [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit_mb)]
    )
    print(f"Configured GPU:{gpu_id} with memory limit: {memory_limit_mb}MB")



class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


def convert_string_to_float(s, i=None, j=None):
    # Check if string contains non-numeric characters (excluding comma and period)
    for char in s:
        if not (char.isdigit() or char in ',.'):
            raise ValueError(f"String contains non-numeric characters at row {i}, column {j}")

    # Count periods and commas in the string
    period_count = s.count('.')
    comma_count = s.count(',')

    # For 'Όγκος' column
    if period_count > 1 and comma_count == 0:
        # Remove periods (thousands separators) and convert to float
        s = s.replace('.', '')
        return float(s)
    # For 'Τζίρος' column
    elif period_count > 1 and comma_count == 1:
        # Remove periods (thousands separators), replace comma with period (decimal point) and convert to float
        s = s.replace('.', '').replace(',', '.')
        return float(s)
    # For the rest of the columns
    else:
        # Replace comma with period (decimal point) and convert to float
        s = s.replace(',', '.')
        return float(s)


def plot_values(true_values, train_predictions, val_predictions, filename):
    # Flatten the true values and predictions
    true_values_flat = [item for sublist in true_values for item in sublist]
    train_predictions_flat = [item for sublist in train_predictions for item in sublist]
    val_predictions_flat = [item for sublist in val_predictions for item in sublist]

    # Define plot ranges
    train_range = range(len(train_predictions_flat))
    val_range = range(len(train_predictions_flat), len(train_predictions_flat) + len(val_predictions_flat))

    # Plot true values
    plt.plot(true_values_flat, color='blue', label='True Values')

    # Plot training predictions
    plt.plot(train_range, train_predictions_flat, color='yellow', label='Training Predictions')

    # Plot validation predictions
    plt.plot(val_range, val_predictions_flat, color='red', label='Validation Predictions')

    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.title('True vs Predicted Values')

    # Save the plot to a file
    plt.savefig(filename)

def process_dataframe(data):
    # Make a copy of the original dataframe to avoid modifying it directly
    data_processed = data.copy()

    # Identify rows with NaN values
    nan_rows = data_processed.isnull().any(axis=1)

    # Collect indices of rows with NaN values
    nan_indices = [i for i, is_nan in enumerate(nan_rows) if is_nan]

    # Print row indices and drop rows
    for i in nan_indices:
        print(f"Row {i} contains NaN. Deleting the row.")
    data_processed.drop(nan_indices, inplace=True)

    # Reset dataframe index after dropping rows
    data_processed.reset_index(drop=True, inplace=True)

    # Loop over rows
    for i, row in data_processed.iterrows():
        # Loop over columns
        for j, cell in enumerate(row):
            # Skip the first (index 0) and third (index 2) columns
            if j not in [0, 2]:
                try:
                    # Try to convert the cell to a float
                    if pd.isna(cell):
                        print(f"NaN found at row {i}, column {j}")
                    else:
                        data_processed.iat[i, j] = convert_string_to_float(str(cell), i, j)
                except ValueError as e:
                    # If a ValueError is raised, add information about the row and column
                    raise ValueError(f"Error in row {i}, column {j}: {e}")

    return data_processed


def select_csv_file():
    # Ask user to enter a CSV file path
    file_path = input("Please enter the path to the CSV file: ")
    if not os.path.isfile(file_path):
        raise ValueError("No such file found.")
    return file_path


def load_and_preprocess_data(filename, lookback, batch_size=32, validation_split=0.3):
    # Read CSV skipping the first row (header and title)
    data = pd.read_csv(filename, skiprows=1)
    # Use process_dataframe function here
    data = process_dataframe(data)
    data = data.iloc[:, [0, 1, 3, 4, 5, 6, 7]]  # Reordered columns
    data.columns = ['Date', 'Close', 'Open', 'High', 'Low', 'Volume', 'Turnover']
    data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')
    data.sort_values(by='Date', ascending=True, inplace=True)
    data.reset_index(drop=True, inplace=True)
    data['Day'] = data['Date'].dt.day.astype(float)
    data['Month'] = data['Date'].dt.month.astype(float)
    data['Year'] = data['Date'].dt.year.astype(float)

    data['Historical Close'] = data['Close'].shift(1)
    data.dropna(inplace=True)
    data['Historical Close'] = data['Historical Close'].shift(lookback)
    data.drop(data.head(lookback).index, inplace=True)
    data.dropna(inplace=True)

    # Create a copy of the 'Close' column before scaling
    data_for_plot = data['Close'].copy()

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler_close = MinMaxScaler(feature_range=(-1, 1))  # separate scaler for 'Close'
    data[['Open', 'High', 'Low', 'Volume', 'Turnover', 'Historical Close']] = scaler.fit_transform(
        data[['Open', 'High', 'Low', 'Volume', 'Turnover', 'Historical Close']])
    data[['Close']] = scaler_close.fit_transform(data[['Close']])  # fit and transform 'Close' separately

    X, Y = [], []
    for i in range(len(data) - lookback - 7):
        X.append(np.column_stack((data['Day'].values[i:(i + lookback)],
                                  data['Month'].values[i:(i + lookback)],
                                  data['Year'].values[i:(i + lookback)],
                                  data['Open'].values[i:(i + lookback)],
                                  data['High'].values[i:(i + lookback)],
                                  data['Low'].values[i:(i + lookback)],
                                  data['Volume'].values[i:(i + lookback)],
                                  data['Turnover'].values[i:(i + lookback)],
                                  data['Historical Close'].values[i:(i + lookback)])))
        Y.append(data['Close'].values[(i + lookback):(i + lookback + 7)])


    X, Y = np.array(X), np.array(Y)

    # Split the data into training and validation sets
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=validation_split, shuffle=False)

    # Create tf.data.Dataset objects for training and validation
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val))

    # Batch the training and validation datasets
    train_dataset = train_dataset.batch(512)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    val_dataset = val_dataset.batch(512)
    val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return scaler, train_dataset, val_dataset, data_for_plot


def build_model(lookback, l2_factor=0.009):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, kernel_regularizer=l2(l2_factor), input_shape=(lookback, 9)))  # Updated input shape
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=True, kernel_regularizer=l2(l2_factor)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(LSTM(64, kernel_regularizer=l2(l2_factor)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(7, activation='tanh'))
    model.compile(loss='mean_squared_error', optimizer='adam')

    return model



def train_model(train_dataset, val_dataset, model):
    lr_schedule = ExponentialDecay(
        initial_learning_rate=1e-2,
        decay_steps=1000,
        decay_rate=0.9)

    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule))

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True)

    time_callback = TimeHistory()

    model.fit(train_dataset, epochs=100, validation_data=val_dataset, verbose=1, callbacks=[time_callback, early_stopping, tensorboard_callback])

    train_loss = model.evaluate(train_dataset, verbose=1)
    train_predictions = model.predict(train_dataset)  # Get predictions on training data
    return train_loss, train_predictions, model, time_callback.times


def rolling_window_validation_process(train_dataset, val_dataset, lookback, window_size, checkpoint_dir, starting_i=0, checkpoint_file=None):
    train_losses = []
    train_predictions_all = []  # Collect training predictions
    val_predictions_all = []  # Collect validation predictions
    models = []
    training_times = []

    print("Starting rolling window validation process...")

    # Load model from checkpoint or build a new model outside the loop
    if checkpoint_file and os.path.isfile(checkpoint_file):
        print(f"Loading model from {checkpoint_file}...")
        model = tf.keras.models.load_model(checkpoint_file)
    else:
        model = build_model(lookback)

    for i in range(starting_i if checkpoint_file else lookback, len(train_dataset) - window_size):
        print(f"Training model {i + 1} of {len(train_dataset) - window_size}...")
        train_data = train_dataset.take(i)
        test_data = train_dataset.skip(i).take(window_size)

        # Calculate the validation run number
        validation_run_number = (i - lookback) // window_size + 1

        # Add a ModelCheckpoint callback
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, f"{i}_{validation_run_number}.h5"),
            monitor='val_loss',
            save_best_only=True,
            verbose=1,
            save_weights_only=False)

        # Continue training the same model within the loop
        train_loss, train_predictions, model, training_time = train_model(train_data, val_dataset, model)
        val_predictions = model.predict(test_data)  # Get predictions on validation data

        model.fit(train_data, epochs=100, batch_size=512, validation_data=test_data, verbose=1,
                  callbacks=[model_checkpoint])

        train_losses.append(train_loss)
        train_predictions_all.append(train_predictions)
        val_predictions_all.append(val_predictions)
        models.append(model)
        training_times.append(training_time)
        print(
            f"Finished training model {i + 1}. Train loss: {train_loss}, Training time: {sum(training_time)} seconds.")

    print("Rolling window validation process completed.")
    return train_losses, train_predictions_all, val_predictions_all, models, training_times

def save_model(model,data_for_plot , train_predictions_all, val_predictions_all):
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
        plot_values(true_values=data_for_plot, train_predictions=train_predictions_all, val_predictions=val_predictions_all, filename=file_path)
        print(f"Plot saved at location : {file_path}")
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
    setup_gpu_for_training(0,15)
    # Run the script
    lookback = 30  # use past 14 days data to predict next 7 days
    filename = select_csv_file()
    print("Loading and preprocessing data...")
    scaler, train_dataset, val_dataset, data_for_plot = load_and_preprocess_data(filename, lookback)

    # Get the checkpoint directory from the user
    checkpoint_dir = input("Please enter the directory to save the checkpoints: ")

    # Ask the user if they want to start from a checkpoint
    use_checkpoint = input("Do you want to start from a checkpoint? (yes/no): ").lower() == "yes"
    checkpoint_file = None
    starting_i = 0
    if use_checkpoint:
        checkpoint_file = input("Please enter the path to the checkpoint file: ")
        starting_i = int(os.path.basename(checkpoint_file).split("_")[0])

    train_losses, train_predictions_all, val_predictions_all, models, training_times = rolling_window_validation_process(train_dataset, val_dataset, lookback, 30,
                                                                                             checkpoint_dir,
                                                                                             starting_i,
                                                                                             checkpoint_file)
    best_model_index = np.argmin(train_losses)
    best_model = models[best_model_index]
    print(f"Best model selected with a training loss of {train_losses[best_model_index]}.")
    if ask_user_to_save_model():
        save_model(best_model,data_for_plot , train_predictions_all, val_predictions_all)
    print("Finished the script.")

if __name__ == "__main__":
    main()
