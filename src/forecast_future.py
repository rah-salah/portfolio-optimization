import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import json
import sys
import warnings
warnings.filterwarnings("ignore")

FORECAST_DAYS = 252  # ~12 months of trading days
ORDER = (2, 1, 2)


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


def fit_arima(series, order=ORDER):
    try:
        model = ARIMA(series, order=order)
        return model.fit()
    except Exception as e:
        print(f"ERROR: ARIMA failed to fit with order {order}: {e}")
        sys.exit(1)


def make_future_index(last_date, periods):
    return pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=periods)


def forecast_with_ci(fitted, periods, alpha=0.05):
    try:
        forecast_obj = fitted.get_forecast(steps=periods)
        mean = forecast_obj.predicted_mean
        ci = forecast_obj.conf_int(alpha=alpha)
        return mean, ci
    except Exception as e:
        print(f"ERROR: forecasting failed: {e}")
        sys.exit(1)


def analyze_trend(history, mean_forecast, ci, future_index):
    try:
        last_price = history.iloc[-1]
        end_price = mean_forecast.iloc[-1]
        pct_change = (end_price - last_price) / last_price * 100
        direction = "upward" if pct_change > 0 else "downward"

        ci_width = ci.iloc[:, 1] - ci.iloc[:, 0]
        milestones = {}
        for months, label in [(21, "1_month"), (126, "6_month"), (252, "12_month")]:
            if months <= len(ci_width):
                idx = months - 1
                milestones[label] = {
                    "date": str(future_index[idx].date()),
                    "forecast_price": round(float(mean_forecast.iloc[idx]), 2),
                    "ci_lower": round(float(ci.iloc[idx, 0]), 2),
                    "ci_upper": round(float(ci.iloc[idx, 1]), 2),
                    "ci_width": round(float(ci_width.iloc[idx]), 2),
                }

        return {
            "last_actual_price": round(float(last_price), 2),
            "last_actual_date": str(history.index[-1].date()),
            "forecast_end_price": round(float(end_price), 2),
            "forecast_end_date": str(future_index[-1].date()),
            "pct_change": round(float(pct_change), 2),
            "trend_direction": direction,
            "milestones": milestones,
            "ci_widens_over_time": bool(ci_width.iloc[-1] > ci_width.iloc[0]),
        }
    except Exception as e:
        print(f"ERROR: trend analysis failed: {e}")
        return {}


def plot_forecast(history, mean_forecast, ci, future_index, path="data/processed/future_forecast.png"):
    try:
        fig, ax = plt.subplots(figsize=(14, 6))
        recent = history[-252:]
        ax.plot(recent.index, recent, label="Historical (last 12mo)", color="#3498db")
        ax.plot(future_index, mean_forecast, label="Forecast", color="#e74c3c", linestyle="--")
        ax.fill_between(
            future_index, ci.iloc[:, 0], ci.iloc[:, 1],
            color="#e74c3c", alpha=0.2, label="95% Confidence Interval"
        )
        ax.set_title("TSLA 12-Month Price Forecast with Confidence Intervals (ARIMA(2,1,2))", fontweight="bold")
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

    fitted = fit_arima(tsla)
    print(f"ARIMA{ORDER} fitted on full TSLA series")

    future_index = make_future_index(tsla.index[-1], FORECAST_DAYS)
    mean_forecast, ci = forecast_with_ci(fitted, FORECAST_DAYS)
    mean_forecast.index = future_index
    ci.index = future_index

    summary = analyze_trend(tsla, mean_forecast, ci, future_index)
    print(json.dumps(summary, indent=2))

    save_summary(summary)
    plot_forecast(tsla, mean_forecast, ci, future_index)


if __name__ == "__main__":
    main()
