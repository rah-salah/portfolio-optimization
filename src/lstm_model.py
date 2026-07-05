import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
import json
import warnings
warnings.filterwarnings("ignore")

WINDOW = 60
EPOCHS = 20
BATCH_SIZE = 32


def load_data(path="data/processed/prices.csv"):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df["TSLA"]


def make_sequences(data, window):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i - window:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


def build_lstm(window):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(window, 1)),
        tf.keras.layers.LSTM(50),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def evaluate(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "MAPE": round(mape, 2)}


def plot_forecast(dates, actual, predicted, path="data/processed/lstm_forecast.png"):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(dates, actual, label="Actual", color="#2ecc71")
    ax.plot(dates, predicted, label="LSTM Forecast", color="#e74c3c", linestyle="--")
    ax.set_title("TSLA Price: LSTM Forecast vs Actual", fontweight="bold")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {path}")


def main():
    tsla = load_data()
    split = int(len(tsla) * 0.8)
    train_raw = tsla.values[:split].reshape(-1, 1)
    test_raw = tsla.values[split:].reshape(-1, 1)
    print(f"Train: {len(train_raw)} rows | Test: {len(test_raw)} rows")

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_raw)
    test_scaled = scaler.transform(test_raw)

    X_train, y_train = make_sequences(train_scaled, WINDOW)
    X_test, y_test = make_sequences(
        np.vstack([train_scaled[-WINDOW:], test_scaled]), WINDOW
    )
    X_train = X_train.reshape(-1, WINDOW, 1)
    X_test = X_test.reshape(-1, WINDOW, 1)
    print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")

    model = build_lstm(WINDOW)
    print("Training LSTM...")
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

    preds_scaled = model.predict(X_test, verbose=0)
    preds = scaler.inverse_transform(preds_scaled).flatten()
    actual = tsla.values[split:].flatten()
    dates = tsla.index[split:]

    metrics = evaluate(actual, preds)
    print(f"LSTM Metrics: {metrics}")

    json.dump({"model": "LSTM", **metrics},
              open("data/processed/lstm_metrics.json", "w"))

    plot_forecast(dates, actual, preds)


if __name__ == "__main__":
    main()
