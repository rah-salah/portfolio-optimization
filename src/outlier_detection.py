import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("data/processed/prices.csv", index_col=0, parse_dates=True)
returns = df.pct_change().dropna()

print("=== Outlier Detection (returns > 3 std devs) ===")
for col in returns.columns:
    mean, std = returns[col].mean(), returns[col].std()
    outliers = returns[col][np.abs(returns[col] - mean) > 3*std]
    print(f"{col}: {len(outliers)} outliers")
    if len(outliers) > 0:
        print(outliers.sort_values().head(3))

fig, axes = plt.subplots(3,1,figsize=(14,10))
for i, col in enumerate(returns.columns):
    mean, std = returns[col].mean(), returns[col].std()
    outliers = returns[col][np.abs(returns[col]-mean) > 3*std]
    axes[i].plot(returns.index, returns[col], color="#3498db", linewidth=0.7)
    axes[i].scatter(outliers.index, outliers, color="red", zorder=5, s=20, label="Outliers")
    axes[i].set_title(f"{col} Daily Returns with Outliers", fontweight="bold")
    axes[i].legend()
plt.tight_layout()
plt.savefig("data/processed/outliers.png", dpi=150, bbox_inches="tight")
plt.close()
print("Outlier plot saved")
