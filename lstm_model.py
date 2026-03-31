import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


def create_dataset(dataset, lookback=7):
    X, y = [], []
    for i in range(lookback, len(dataset)):
        X.append(dataset[i-lookback:i, 0])
        y.append(dataset[i, 0])
    return np.array(X), np.array(y)


def run_lstm(df):
    df['Ngày'] = pd.to_datetime(df['Ngày'], dayfirst=True, errors='coerce')
    df = df.dropna()

    data = df.groupby('Ngày')['Sản lượng xăng'].sum().sort_index()

    scaler = MinMaxScaler()
    dataset = scaler.fit_transform(data.values.reshape(-1,1))

    train_size = int(len(dataset) * 0.8)
    train, test = dataset[:train_size], dataset[train_size:]

    X_train, y_train = create_dataset(train)
    X_test, y_test = create_dataset(test)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    model = Sequential()
    model.add(LSTM(50, input_shape=(X_train.shape[1],1)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')

    model.fit(X_train, y_train, epochs=10, verbose=0)

    pred = model.predict(X_test)

    pred = scaler.inverse_transform(pred)
    y_test = scaler.inverse_transform(y_test.reshape(-1,1))

    rmse = np.sqrt(mean_squared_error(y_test, pred))

    return data, pred.flatten(), y_test.flatten(), rmse


def forecast_next_month_lstm(df, steps=30):
    df['Ngày'] = pd.to_datetime(df['Ngày'], dayfirst=True, errors='coerce')
    df = df.dropna()

    data = df.groupby('Ngày')['Sản lượng xăng'].sum().sort_index()

    scaler = MinMaxScaler()
    dataset = scaler.fit_transform(data.values.reshape(-1,1))

    lookback = 7
    X, y = create_dataset(dataset, lookback)

    X = X.reshape(X.shape[0], X.shape[1], 1)

    model = Sequential()
    model.add(LSTM(50, input_shape=(lookback,1)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')

    model.fit(X, y, epochs=10, verbose=0)

    input_seq = dataset[-lookback:]
    predictions = []

    for _ in range(steps):
        pred = model.predict(input_seq.reshape(1,lookback,1), verbose=0)
        predictions.append(pred[0][0])
        input_seq = np.append(input_seq[1:], pred, axis=0)

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1,1))

    return data, predictions.flatten()