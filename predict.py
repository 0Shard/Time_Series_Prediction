import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf
import mplfinance as mpf
import os


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

def load_and_preprocess_data(filename, lookback):
    data = pd.read_csv(filename, skiprows=1)
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

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler_close = MinMaxScaler(feature_range=(-1, 1))
    data_unnormalized = data.copy()  # Added line to copy unnormalized data
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

    return scaler, scaler_close, np.array(X), np.array(Y), data_unnormalized  # Return unnormalized data



def select_h5_file():
    # Ask user to enter a .h5 file path
    file_path = input("Please enter the path to the .h5 file: ")
    if not os.path.isfile(file_path):
        raise ValueError("No such file found.")
    return file_path

def select_csv_file():
    # Ask user to enter a CSV file path
    file_path = input("Please enter the path to the CSV file: ")
    if not os.path.isfile(file_path):
        raise ValueError("No such file found.")
    return file_path

def load_model(path_to_model):
    model = tf.keras.models.load_model(path_to_model)
    return model

def make_prediction(model, X, lookback):
    predictions = model.predict(X[-1].reshape(1, lookback, 9))  # Note: input shape is (lookback, 9)
    return predictions

def plot_predictions(data_unnormalized, predictions):
    data = data_unnormalized.copy()  # Use the unnormalized data
    frames = []
    for i in range(7):
        df = pd.DataFrame([{'Date': data['Date'].iloc[-1] + pd.DateOffset(days=1), 'Close': predictions[0][i]}])
        frames.append(df)
    temp_df = pd.concat(frames, ignore_index=True)
    data = pd.concat([data, temp_df])
    mpf.plot(data.set_index('Date')[-100:], type='line', title='Stock Prices', ylabel='Price', volume=True, style='yahoo')




def main():
    filename = select_csv_file()  # Adjust this if your CSV file is located elsewhere
    lookback = 30
    scaler, scaler_close, X, Y, data = load_and_preprocess_data(filename, lookback)  # get scaler_close

    path_to_model = select_h5_file()  # Adjust this if your model is located elsewhere
    model = load_model(path_to_model)

    predictions = make_prediction(model, X, lookback)

    # Scale predictions back to original scale
    predictions = scaler_close.inverse_transform(predictions)

    # Print the list of predicted values
    print(predictions)

    plot_predictions(data, predictions)

if __name__ == "__main__":
    main()
