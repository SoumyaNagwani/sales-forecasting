import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def train_and_predict(df):
    df['ds'] = pd.to_datetime(df['ds'])
    df['t'] = np.arange(len(df))

    X = df[['t']]
    y = df['y']

    model = LinearRegression()
    model.fit(X, y)

    future_t = np.arange(len(df), len(df) + 30).reshape(-1, 1)
    future_pred = model.predict(future_t)

    # Make sure static folder exists
    os.makedirs('static', exist_ok=True)

    # Plotting
    plt.figure(figsize=(10, 6), dpi=120)
    plt.plot(df['ds'], y, label='Historical', linewidth=2, marker='o')
    plt.plot(pd.date_range(df['ds'].iloc[-1], periods=30, freq='D'),
             future_pred, label='Forecast', linewidth=2, linestyle='--', color='orange')

    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Sales", fontsize=14)
    plt.title("📈 Sales Forecast", fontsize=16)
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend(fontsize=12)

    fig_path = 'static/forecast.png'
    plt.tight_layout()
    plt.savefig(fig_path, dpi=120)

    return round(future_pred[-1], 2), fig_path
