import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf
import mplfinance as mpf

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

def load_and_preprocess_data(filename, lookback):
    data = pd.read_csv(filename)
    data = data.iloc[:, [0, 1, 6]]
    data.columns = ['Date', 'Close', 'Volume']
    data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%y')
    data.sort_values(by='Date', ascending=True, inplace=True)
    data.reset_index(drop=True, inplace=True)
    data['Day'] = data['Date'].dt.day
    data['Month'] = data['Date'].dt.month
    data['Year'] = data['Date'].dt.year
    data['Close'] = data['Close'].apply(remove_characters_and_convert_to_integer)
    data['Volume'] = data['Volume'].apply(remove_characters_and_convert_to_integer)
    data['Historical Close'] = data['Close'].shift(1)
    data.dropna(inplace=True)
    data['Historical Close'] = data['Historical Close'].shift(lookback)
    data.drop(data.head(lookback).index, inplace=True)

    data.dropna(inplace=True)

    scaler = MinMaxScaler(feature_range=(0, 1))
    data[['Close', 'Volume']] = scaler.fit_transform(data[['Close', 'Volume']])
    X, Y = [], []
    for i in range(len(data) - lookback - 7):
        X.append(np.column_stack((data['Day'].values[i:(i + lookback)],
                                  data['Month'].values[i:(i + lookback)],
                                  data['Year'].values[i:(i + lookback)],
                                  data['Volume'].values[i:(i + lookback)],
                                  data['Historical Close'].values[i:(i + lookback)])))
        Y.append(data['Close'].values[(i + lookback):(i + lookback + 7)])
    X = np.array(X)
    Y = np.array(Y)
    return scaler, X, Y

def load_model(path_to_model):
    model = tf.keras.models.load_model(path_to_model)
    return model

def make_prediction(model, X, lookback):
    predictions = model.predict(X[-1].reshape(1, lookback, 5))
    return predictions

def plot_predictions(data, predictions):
    for i in range(7):
        data = data.append({'Date': data['Date'].iloc[-1] + pd.DateOffset(days=1), 'Close': predictions[0][i]}, ignore_index=True)
    mpf.plot(data.set_index('Date')[-100:], type='line', title='Stock Prices', ylabel='Price', volume=True, style='yahoo')

def main():
    # Use the function to load and preprocess data
    filename = "History_Closes_For_AI.csv"  # Adjust this if your CSV file is located elsewhere
    lookback = 14
    scaler, X, Y = load_and_preprocess_data(filename, lookback)

    # Load the trained model
    # path_to_model = "model.h5"  # Adjust this if your model is located elsewhere
    # model = load_model(path_to_model)

    # Predict the next 7 days
    # predictions = make_prediction(model, X, lookback)

    # Scale predictions back to original scale
    # predictions = scaler.inverse_transform(predictions)

    # Plotting
    # plot_predictions(data, predictions)

if __name__ == "__main__":
    main()
