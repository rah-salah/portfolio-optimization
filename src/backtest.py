import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
import sys
import warnings
warnings.filterwarnings("ignore")

BACKTEST_DAYS = 252  # last ~12 months, held out from optimization period
RISK_FREE_RATE = 0.02


def load_data(path="data/processed/prices.csv"):
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    except FileNotFoundError:
        print(f"ERROR: {path} not found. Run src/data_loader.py first.")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: failed to load {path}: {e}")
        sys.exit(1)
    return df


def load_recommended_weights(path="data/processed/portfolio_optimization.json"):
    try:
        results = json.load(open(path))
        return {
            "max_sharpe": results["max_sharpe"]["weights"],
            "min_volatility": results["min_volatility"]["weights"],
        }
    except Exception as e:
        print(f"ERROR: could not load weights from {path}: {e}")
        print("Run src/portfolio_optimization.py first.")
        sys.exit(1)


def compute_backtest_window(df, days):
    if len(df) <= days:
        print(f"ERROR: not enough data ({len(df)} rows) for a {days}-day backtest window")
        sys.exit(1)
    return df.iloc[-days:]


def buy_and_hold_returns(prices, weights):
    """Simple buy-and-hold: invest according to weights on day 1, no rebalancing."""
    returns = prices.pct_change().dropna()
    weighted_returns = sum(returns[t] * w for t, w in weights.items() if t in returns.columns)
    cumulative = (1 + weighted_returns).cumprod()
    return weighted_returns, cumulative


def compute_metrics(daily_returns, cumulative):
    try:
        total_return = cumulative.iloc[-1] - 1
        n_days = len(daily_returns)
        annualized_return = (1 + total_return) ** (252 / n_days) - 1
        annualized_vol = daily_returns.std() * np.sqrt(252)
        sharpe = (annualized_return - RISK_FREE_RATE) / annualized_vol if annualized_vol > 0 else np.nan

        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        return {
            "total_return_pct": round(float(total_return) * 100, 2),
            "annualized_return_pct": round(float(annualized_return) * 100, 2),
            "annualized_volatility_pct": round(float(annualized_vol) * 100, 2),
            "sharpe_ratio": round(float(sharpe), 4),
            "max_drawdown_pct": round(float(max_drawdown) * 100, 2),
        }
    except Exception as e:
        print(f"ERROR: metric computation failed: {e}")
        return {}


def plot_comparison(cum_max_sharpe, cum_min_vol, cum_benchmark, path="data/processed/backtest_comparison.png"):
    try:
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(cum_max_sharpe.index, (cum_max_sharpe - 1) * 100,
                label="Max Sharpe Portfolio", color="#e74c3c", linewidth=1.5)
        ax.plot(cum_min_vol.index, (cum_min_vol - 1) * 100,
                label="Min Volatility Portfolio (Recommended)", color="#9b59b6", linewidth=1.5)
        ax.plot(cum_benchmark.index, (cum_benchmark - 1) * 100,
                label="Benchmark (60% SPY / 40% BND)", color="#3498db", linewidth=1.5)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title("Backtest: Portfolio Strategies vs. 60/40 Benchmark", fontweight="bold")
        ax.set_ylabel("Cumulative Return (%)")
        ax.legend()
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Plot saved to {path}")
    except Exception as e:
        print(f"WARNING: could not save comparison plot: {e}")


def save_results(results, path="data/processed/backtest_results.json"):
    try:
        json.dump(results, open(path, "w"), indent=2)
        print(f"Results saved to {path}")
    except Exception as e:
        print(f"WARNING: could not save results: {e}")


def main():
    df = load_data()
    weights = load_recommended_weights()
    print(f"Max Sharpe weights: {weights['max_sharpe']}")
    print(f"Min Volatility weights: {weights['min_volatility']}")

    window = compute_backtest_window(df, BACKTEST_DAYS)
    print(f"Backtest window: {window.index[0].date()} to {window.index[-1].date()} ({len(window)} rows)")

    ms_returns, ms_cum = buy_and_hold_returns(window, weights["max_sharpe"])
    mv_returns, mv_cum = buy_and_hold_returns(window, weights["min_volatility"])
    bench_weights = {"SPY": 0.6, "BND": 0.4}
    bench_returns, bench_cum = buy_and_hold_returns(window, bench_weights)

    ms_metrics = compute_metrics(ms_returns, ms_cum)
    mv_metrics = compute_metrics(mv_returns, mv_cum)
    bench_metrics = compute_metrics(bench_returns, bench_cum)

    print("\n=== Max Sharpe Portfolio ===")
    print(json.dumps(ms_metrics, indent=2))
    print("\n=== Min Volatility Portfolio ===")
    print(json.dumps(mv_metrics, indent=2))
    print("\n=== 60/40 Benchmark ===")
    print(json.dumps(bench_metrics, indent=2))

    results = {
        "max_sharpe_portfolio": ms_metrics,
        "min_volatility_portfolio": mv_metrics,
        "benchmark_60_40": bench_metrics,
        "weights_used": weights,
    }
    save_results(results)
    plot_comparison(ms_cum, mv_cum, bench_cum)


if __name__ == "__main__":
    main()
