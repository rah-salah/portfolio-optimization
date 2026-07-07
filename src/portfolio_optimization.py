import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
import sys
import warnings
warnings.filterwarnings("ignore")

try:
    from pypfopt import EfficientFrontier, risk_models, expected_returns
    from pypfopt import plotting
    HAS_PYPFOPT = True
except ImportError:
    HAS_PYPFOPT = False

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


def load_forecast_return(path="data/processed/future_forecast_summary.json"):
    """Use the Task 3 ARIMA forecast to inform TSLA's expected return instead of
    relying purely on historical mean, since the forecast is the whole point of Task 3."""
    try:
        summary = json.load(open(path))
        pct_change = summary["pct_change"]  # over ~12 months
        return pct_change / 100.0
    except Exception as e:
        print(f"WARNING: could not load forecast summary ({e}); falling back to historical TSLA return.")
        return None


def compute_mu_sigma(df, forecast_tsla_return=None):
    try:
        mu = expected_returns.mean_historical_return(df) if HAS_PYPFOPT else df.pct_change().dropna().mean() * 252
        sigma = risk_models.sample_cov(df) if HAS_PYPFOPT else df.pct_change().dropna().cov() * 252

        if forecast_tsla_return is not None:
            print(f"Overriding historical TSLA expected return ({mu['TSLA']:.4f}) "
                  f"with Task 3 forecast-implied return ({forecast_tsla_return:.4f})")
            mu["TSLA"] = forecast_tsla_return

        return mu, sigma
    except Exception as e:
        print(f"ERROR: failed to compute expected returns / covariance: {e}")
        sys.exit(1)


def optimize_portfolios(mu, sigma):
    results = {}
    try:
        ef_sharpe = EfficientFrontier(mu, sigma)
        try:
            ef_sharpe.max_sharpe(risk_free_rate=RISK_FREE_RATE)
        except Exception as e:
            print(f"WARNING: max_sharpe failed with a 0% risk-free rate fallback ({e}); retrying with rf=0.0")
            ef_sharpe = EfficientFrontier(mu, sigma)
            ef_sharpe.max_sharpe(risk_free_rate=0.0)
        w_sharpe = ef_sharpe.clean_weights()
        perf_sharpe = ef_sharpe.portfolio_performance(risk_free_rate=RISK_FREE_RATE)
        results["max_sharpe"] = {
            "weights": {k: round(v, 4) for k, v in w_sharpe.items()},
            "expected_annual_return": round(perf_sharpe[0], 4),
            "annual_volatility": round(perf_sharpe[1], 4),
            "sharpe_ratio": round(perf_sharpe[2], 4),
        }

        ef_minvol = EfficientFrontier(mu, sigma)
        ef_minvol.min_volatility()
        w_minvol = ef_minvol.clean_weights()
        perf_minvol = ef_minvol.portfolio_performance(risk_free_rate=RISK_FREE_RATE)
        results["min_volatility"] = {
            "weights": {k: round(v, 4) for k, v in w_minvol.items()},
            "expected_annual_return": round(perf_minvol[0], 4),
            "annual_volatility": round(perf_minvol[1], 4),
            "sharpe_ratio": round(perf_minvol[2], 4),
        }
    except Exception as e:
        print(f"ERROR: portfolio optimization failed: {e}")
        sys.exit(1)

    return results


def plot_efficient_frontier(mu, sigma, results, path="data/processed/efficient_frontier.png"):
    try:
        fig, ax = plt.subplots(figsize=(10, 7))
        ef_plot = EfficientFrontier(mu, sigma)
        plotting.plot_efficient_frontier(ef_plot, ax=ax, show_assets=True)

        ms = results["max_sharpe"]
        mv = results["min_volatility"]
        ax.scatter(ms["annual_volatility"], ms["expected_annual_return"],
                   marker="*", s=300, color="#e74c3c", label="Max Sharpe (Tangency)", zorder=5)
        ax.scatter(mv["annual_volatility"], mv["expected_annual_return"],
                   marker="*", s=300, color="#3498db", label="Min Volatility", zorder=5)
        ax.set_title("Efficient Frontier: TSLA / BND / SPY", fontweight="bold")
        ax.legend()
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Plot saved to {path}")
    except Exception as e:
        print(f"WARNING: could not save efficient frontier plot: {e}")


def save_results(results, path="data/processed/portfolio_optimization.json"):
    try:
        json.dump(results, open(path, "w"), indent=2)
        print(f"Results saved to {path}")
    except Exception as e:
        print(f"WARNING: could not save results: {e}")


def main():
    if not HAS_PYPFOPT:
        print("ERROR: pyportfolioopt is not installed. Run: pip install pyportfolioopt")
        sys.exit(1)

    df = load_data()
    print(f"Loaded prices for: {list(df.columns)}")

    forecast_return = load_forecast_return()
    mu, sigma = compute_mu_sigma(df, forecast_return)

    print("\nExpected annual returns:")
    print(mu.round(4))
    print("\nAnnualized covariance matrix:")
    print(sigma.round(5))

    results = optimize_portfolios(mu, sigma)
    print("\n=== Max Sharpe (Tangency) Portfolio ===")
    print(json.dumps(results["max_sharpe"], indent=2))
    print("\n=== Min Volatility Portfolio ===")
    print(json.dumps(results["min_volatility"], indent=2))

    save_results(results)
    plot_efficient_frontier(mu, sigma, results)


if __name__ == "__main__":
    main()
