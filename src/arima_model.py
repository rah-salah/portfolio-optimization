import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import json
import sys
import warnings
warnings.filterwarnings("ignore")


def load_data(path="data/processed/prices.csv"):
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    except FileNotFoundError:
        print(f"ERROR: {path} not found. Run src/data_loader.py first.")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: failed to load {path}: {e}")
        sys.exit(1)

    if "TSLA" not in df.columns:
        print("ERROR: TSLA column not found in prices.csv")
        sys.exit(1)

    return df["TSLA"]


def train_test_split(series, train_ratio=0.8):
    split = int(len(series) * train_ratio)
    return series[:split], series[split:]


def fit_arima(train, order=(2, 1, 2)):
    try:
        model = ARIMA(train, order=order)
        return model.fit()
    except Exception as e:
        print(f"ERROR: ARIMA failed to fit with order {order}: {e}")
        sys.exit(1)


def evaluate(actual, predicted):
    try:
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "MAPE": round(mape, 2)}
    except Exception as e:
        print(f"ERROR: evaluation failed: {e}")
        return {"MAE": None, "RMSE": None, "MAPE": None}


def plot_forecast(train, test, forecast, path="data/processed/arima_forecast.png"):
    try:
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(train.index[-200:], train[-200:], label="Train", color="#3498db")
        ax.plot(test.index, test, label="Actual", color="#2ecc71")
        ax.plot(forecast.index, forecast, label="ARIMA Forecast", color="#e74c3c", linestyle="--")
        ax.set_title("TSLA Price: ARIMA(2,1,2) Forecast vs Actual", fontweight="bold")
        ax.set_ylabel("Price (USD)")
        ax.legend()
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Plot saved to {path}")
    except Exception as e:
        print(f"WARNING: could not save plot to {path}: {e}")


def save_metrics(metrics, path="data/processed/arima_metrics.json"):
    try:
        json.dump({"model": "ARIMA(2,1,2)", **metrics}, open(path, "w"))
    except Exception as e:
        print(f"WARNING: could not save metrics to {path}: {e}")


def main():
    tsla = load_data()
    train, test = train_test_split(tsla)
    print(f"Train: {len(train)} rows | Test: {len(test)} rows")
    print(f"Train period: {train.index[0].date()} to {train.index[-1].date()}")
    print(f"Test period:  {test.index[0].date()} to {test.index[-1].date()}")

    fitted = fit_arima(train, order=(2, 1, 2))
    print("ARIMA(2,1,2) fitted successfully")

    try:
        forecast = fitted.forecast(steps=len(test))
        forecast.index = test.index
    except Exception as e:
        print(f"ERROR: forecasting failed: {e}")
        sys.exit(1)

    metrics = evaluate(test, forecast)
    print(f"ARIMA Metrics: {metrics}")

    save_metrics(metrics)
    plot_forecast(train, test, forecast)


if __name__ == "__main__":
    main()
