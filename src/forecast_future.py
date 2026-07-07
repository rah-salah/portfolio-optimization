import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import json
import sys
import warnings
warnings.filterwarnings("ignore")

FORECAST_DAYS = 252  # ~12 months of trading days
WINDOW = 60
EPOCHS = 20
BATCH_SIZE = 32
Z_95 = 1.96


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


def load_one_step_rmse(path="data/processed/lstm_metrics.json"):
    """Use the Task 2 LSTM's one-step-ahead test RMSE as the basis for
    forecast uncertainty, since it's already a validated out-of-sample estimate
    of the model's per-step error."""
    try:
        metrics = json.load(open(path))
        return metrics["RMSE"]
    except Exception as e:
        print(f"WARNING: could not load LSTM RMSE from {path} ({e}); using a fallback estimate.")
        return None


def build_lstm(window):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(window, 1)),
        tf.keras.layers.LSTM(50),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def make_sequences(data, window):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i - window:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


def train_production_model(series, window=WINDOW):
    """Train on the FULL historical series (not a train/test split) since this
    model is used to forecast genuinely unseen future dates, not to be evaluated
    against a held-out period. Task 2 already validated LSTM's accuracy on a
    proper chronological holdout."""
    try:
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(series.values.reshape(-1, 1))
        X, y = make_sequences(scaled, window)
        X = X.reshape(-1, window, 1)

        model = build_lstm(window)
        print("Training production LSTM on full history...")
        model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
        return model, scaler
    except Exception as e:
        print(f"ERROR: LSTM training failed: {e}")
        sys.exit(1)


def iterative_forecast(model, scaler, series, periods, window=WINDOW):
    """Iteratively predict and feed predictions back into the window, as required
    for multi-step LSTM forecasting."""
    try:
        scaled_history = scaler.transform(series.values.reshape(-1, 1)).flatten()
        current_window = list(scaled_history[-window:])
        preds_scaled = []

        for _ in range(periods):
            x = np.array(current_window[-window:]).reshape(1, window, 1)
            pred = model.predict(x, verbose=0)[0, 0]
            preds_scaled.append(pred)
            current_window.append(pred)

        preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
        return preds
    except Exception as e:
        print(f"ERROR: iterative forecasting failed: {e}")
        sys.exit(1)


def make_future_index(last_date, periods):
    return pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=periods)


def compute_confidence_intervals(mean_forecast, one_step_rmse, z=Z_95):
    """Approximate growing uncertainty for an iterative multi-step forecast by
    scaling the validated one-step RMSE by sqrt(horizon), consistent with the
    assumption that forecast errors compound similarly to a random walk. This is
    a standard simplification when a model lacks a native uncertainty framework."""
    if one_step_rmse is None:
        one_step_rmse = mean_forecast.std() * 0.05  # rough fallback, rarely used
    horizons = np.arange(1, len(mean_forecast) + 1)
    width = z * one_step_rmse * np.sqrt(horizons)
    lower = mean_forecast - width
    upper = mean_forecast + width
    return lower, upper, width


def analyze_trend(history, mean_forecast, lower, upper, width, future_index):
    try:
        last_price = history.iloc[-1]
        end_price = mean_forecast[-1]
        pct_change = (end_price - last_price) / last_price * 100
        direction = "upward" if pct_change > 0 else "downward"

        milestones = {}
        for months, label in [(21, "1_month"), (126, "6_month"), (252, "12_month")]:
            if months <= len(mean_forecast):
                idx = months - 1
                milestones[label] = {
                    "date": str(future_index[idx].date()),
                    "forecast_price": round(float(mean_forecast[idx]), 2),
                    "ci_lower": round(float(lower[idx]), 2),
                    "ci_upper": round(float(upper[idx]), 2),
                    "ci_width": round(float(width[idx]), 2),
                }

        return {
            "model": "LSTM (iterative multi-step)",
            "last_actual_price": round(float(last_price), 2),
            "last_actual_date": str(history.index[-1].date()),
            "forecast_end_price": round(float(end_price), 2),
            "forecast_end_date": str(future_index[-1].date()),
            "pct_change": round(float(pct_change), 2),
            "trend_direction": direction,
            "milestones": milestones,
            "ci_widens_over_time": bool(width[-1] > width[0]),
        }
    except Exception as e:
        print(f"ERROR: trend analysis failed: {e}")
        return {}


def plot_forecast(history, mean_forecast, lower, upper, future_index, path="data/processed/future_forecast.png"):
    try:
        fig, ax = plt.subplots(figsize=(14, 6))
        recent = history[-252:]
        ax.plot(recent.index, recent, label="Historical (last 12mo)", color="#3498db")
        ax.plot(future_index, mean_forecast, label="LSTM Forecast", color="#e74c3c", linestyle="--")
        ax.fill_between(
            future_index, lower, upper,
            color="#e74c3c", alpha=0.2, label="~95% Confidence Interval"
        )
        ax.set_title("TSLA 12-Month Price Forecast with Confidence Intervals (LSTM)", fontweight="bold")
        ax.set_ylabel("Price (USD)")
        ax.legend()
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Plot saved to {path}")
    except Exception as e:
        print(f"WARNING: could not save plot to {path}: {e}")


def save_summary(summary, path="data/processed/future_forecast_summary.json"):
    try:
        json.dump(summary, open(path, "w"), indent=2)
        print(f"Summary saved to {path}")
    except Exception as e:
        print(f"WARNING: could not save summary to {path}: {e}")


def main():
    tsla = load_data()
    print(f"Loaded {len(tsla)} rows, last date: {tsla.index[-1].date()}")

    one_step_rmse = load_one_step_rmse()
    print(f"Using one-step RMSE for uncertainty scaling: {one_step_rmse}")

    model, scaler = train_production_model(tsla)

    future_index = make_future_index(tsla.index[-1], FORECAST_DAYS)
    mean_forecast = iterative_forecast(model, scaler, tsla, FORECAST_DAYS)
    lower, upper, width = compute_confidence_intervals(mean_forecast, one_step_rmse)

    summary = analyze_trend(tsla, mean_forecast, lower, upper, width, future_index)
    print(json.dumps(summary, indent=2))

    save_summary(summary)
    plot_forecast(tsla, mean_forecast, lower, upper, future_index)


if __name__ == "__main__":
    main()
