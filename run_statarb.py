"""End-to-end pairs-trading / statistical-arbitrage demo.

Generates (or loads) a cointegrated pair, runs the Engle-Granger test, builds
the z-score signal, backtests with transaction costs, prints metrics, and saves
a diagnostic plot to ``results/``.

Usage
-----
    python run_statarb.py                       # synthetic cointegrated pair
    python run_statarb.py --csv data/sample_pair.csv
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

from statarb.backtest import backtest_pairs
from statarb.cointegration import engle_granger_test
from statarb.data import generate_cointegrated_pair
from statarb.plotting import plot_results
from statarb.signals import compute_spread, generate_positions, zscore


def load_or_generate(csv_path: str | None) -> pd.DataFrame:
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        print(f"Loaded pair from {csv_path} ({len(df)} rows)")
        return df
    df = generate_cointegrated_pair(n=1000, beta=1.5, gamma=10.0, kappa=0.05, seed=42)
    print("Generated synthetic cointegrated pair (n=1000, beta=1.5)")
    return df


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", default=None, help="CSV with columns A, B indexed by date")
    p.add_argument("--window", type=int, default=30, help="z-score rolling window")
    p.add_argument("--entry", type=float, default=2.0)
    p.add_argument("--exit", type=float, default=0.5)
    p.add_argument("--cost-bps", type=float, default=1.0)
    args = p.parse_args()

    df = load_or_generate(args.csv)

    # 1. Cointegration test + hedge ratio.
    result = engle_granger_test(df["B"], df["A"])
    print("\n--- Engle-Granger cointegration test ---")
    print(f"  Hedge ratio (beta)   : {result.hedge_ratio:.4f}")
    print(f"  Intercept (alpha)    : {result.intercept:.4f}")
    print(f"  ADF statistic        : {result.adf_stat:.4f}")
    print(f"  ADF p-value          : {result.adf_pvalue:.4g}")
    print(f"  Cointegrated (5%)    : {result.is_cointegrated}")
    print(f"  Spread half-life     : {result.half_life:.1f} days")

    if not result.is_cointegrated:
        print("\n[warn] Pair is not cointegrated; trading it is not advisable.")

    # 2. Signal construction.
    spread = compute_spread(df["B"], df["A"], result.hedge_ratio)
    z = zscore(spread, window=args.window)
    positions = generate_positions(z, entry=args.entry, exit=args.exit)

    # 3. Backtest.
    bt = backtest_pairs(
        df["A"], df["B"], spread, z, positions, result.hedge_ratio,
        cost_bps=args.cost_bps,
    )
    print("\n--- Backtest performance ---")
    labels = {
        "total_pnl": ("Total PnL (spread currency units)", "{:.4f}"),
        "annualized_return": ("Annualized return", "{:.2%}"),
        "annualized_volatility": ("Annualized volatility", "{:.2%}"),
        "sharpe_ratio": ("Sharpe ratio", "{:.2f}"),
        "max_drawdown": ("Max drawdown", "{:.2%}"),
        "num_trades": ("Number of trades", "{}"),
        "win_rate": ("Win rate", "{:.2%}"),
    }
    for k, (lab, fmt) in labels.items():
        print(f"  {lab:<32}: {fmt.format(bt.metrics[k])}")

    # 4. Persist artefacts.
    os.makedirs("results", exist_ok=True)
    if not (args.csv and os.path.exists(args.csv)):
        os.makedirs("data", exist_ok=True)
        df.to_csv("data/sample_pair.csv")
        print("\nSaved sample pair -> data/sample_pair.csv")
    plot_results(bt.frame, args.entry, args.exit, "results/pairs_trading.png")
    bt.frame.to_csv("results/backtest.csv")
    print("Saved results/pairs_trading.png and results/backtest.csv")


if __name__ == "__main__":
    main()
