import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np

def run_arima(df):
    df['Ngày'] = pd.to_datetime(df['Ngày'], dayfirst=True, errors='coerce')
    df = df.dropna()

    data = df.groupby('Ngày')['Sản lượng xăng'].sum().sort_index()

    train_size = int(len(data) * 0.8)
    train, test = data[:train_size], data[train_size:]

    model = ARIMA(train, order=(1,1,1))
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=len(test))

    rmse = np.sqrt(mean_squared_error(test, forecast))

    return train, test, forecast, rmse


def forecast_next_month_arima(df, steps=30):
    df['Ngày'] = pd.to_datetime(df['Ngày'], dayfirst=True, errors='coerce')
    df = df.dropna()

    data = df.groupby('Ngày')['Sản lượng xăng'].sum().sort_index()

    model = ARIMA(data, order=(1,1,1))
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=steps)

    return data, forecast