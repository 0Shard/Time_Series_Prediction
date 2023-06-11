import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import mplfinance as mpf
import tkinter as tk
from tkinter import filedialog

def remove_characters_and_convert_to_integer(string):
    # Remove specific characters from the string
    characters_to_remove = [",", "-", ".", "/"]
    for character in characters_to_remove:
        string = string.replace(character, "")

    # Convert the string to an integer
    integer_value = int(string)

    return integer_value

def select_csv_file():
    root = tk.Tk()
    root.withdraw()

    # Open file manager dialog
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])

    return file_path

def load_data(filename):
    # Load the stock market data
    data = pd.read_csv(filename)
    return data

def preprocess_data(data, lookback):
    # Preprocess the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data['Close'] = data['Close'].apply(remove_characters_and_convert_to_integer)
    scaled_data = scaler.fit_transform(data[['Close']])

    # Prepare the data for LSTM
    X = []
    for i in range(len(scaled_data) - lookback):
        X.append(scaled_data[i:(i + lookback), :])
    X = np.array(X)

    # Reshape the input data for LSTM [samples, time steps, features]
    X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2]))

    return scaler, X

def predict(model, data, scaler, lookback):
    # Make predictions
    input_data = scaler.transform(data[-lookback:].values.reshape(-1, 1))
    input_data = input_data.reshape(1, lookback, 1)
    predictions = model.predict(input_data)
    predictions = scaler.inverse_transform(predictions)
    return predictions

def main():
    # Select model file
    model_path = filedialog.askopenfilename(filetypes=[("Model Files", "*.h5")])

    # Load the model
    model = load_model(model_path)

    # Select dataset file
    dataset_path = select_csv_file()

    # Load data
    data = load_data(dataset_path)

    # Preprocess data
    lookback = 14
    scaler, X = preprocess_data(data, lookback)

    # Predict the closing prices for the next week
    predictions = predict(model, data['Close'].values, scaler, lookback)

    # Plot the last two weeks as blue and the predicted prices as red
    last_two_weeks = data.tail(14 + lookback)
    predicted_prices = pd.Series(predictions.flatten(), index=last_two_weeks.index[-7:])

    # Convert the DataFrame to a format compatible with mplfinance
    last_two_weeks['Date'] = pd.to_datetime(last_two_weeks['Date'])
    last_two_weeks = last_two_weeks.set_index('Date')

    # Plot the stock chart
    mpf.plot(last_two_weeks, type='candle', volume=True, title='Last Two Weeks vs. Predicted Prices',
             ylabel='Price', ylabel_lower='Volume', style='yahoo', show_nontrading=True,
             addplot=[mpf.make_addplot(predicted_prices, color='red')])

if __name__ == '__main__':
    main()