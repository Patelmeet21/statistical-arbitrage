"""Plots for the stat-arb strategy: spread, z-score bands, cumulative PnL."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_results(df, entry, exit, out_path, title="Pairs Trading"):
    """Three-panel plot: prices, z-score with bands, and cumulative PnL."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)

    # 1. Prices.
    axes[0].plot(df.index, df["A"], label="Asset A", lw=1.0)
    axes[0].plot(df.index, df["B"], label="Asset B", lw=1.0)
    axes[0].set_title(f"{title} - Price series")
    axes[0].legend(loc="upper left")
    axes[0].grid(alpha=0.3)

    # 2. Z-score with entry/exit bands and shaded positions.
    axes[1].plot(df.index, df["zscore"], color="#333", lw=0.9, label="z-score")
    for lvl, c in [(entry, "r"), (-entry, "r"), (exit, "g"), (-exit, "g")]:
        axes[1].axhline(lvl, color=c, ls="--", alpha=0.6)
    axes[1].axhline(0, color="grey", lw=0.6)
    long_mask = df["position"] == 1
    short_mask = df["position"] == -1
    axes[1].fill_between(df.index, df["zscore"].min(), df["zscore"].max(),
                         where=long_mask, color="green", alpha=0.08)
    axes[1].fill_between(df.index, df["zscore"].min(), df["zscore"].max(),
                         where=short_mask, color="red", alpha=0.08)
    axes[1].set_title("Spread z-score (green=long spread, red=short spread)")
    axes[1].legend(loc="upper left")
    axes[1].grid(alpha=0.3)

    # 3. Cumulative PnL.
    axes[2].plot(df.index, df["equity"], color="#1f77b4", lw=1.2)
    axes[2].set_title("Cumulative PnL / equity")
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
