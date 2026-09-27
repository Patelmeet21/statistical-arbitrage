r"""Pairs-trading backtest with transaction costs.

Given spread positions in :math:`\{-1, 0, +1\}`, the daily strategy return is
the position (held from the previous day, to avoid look-ahead) times the change
in the spread, normalised by the notional. Transaction costs are charged
whenever the position changes, proportional to the traded notional.

The backtest returns a DataFrame with the spread, z-score, positions, PnL and
cumulative equity, plus a metrics summary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


TRADING_DAYS = 252


@dataclass
class BacktestResult:
    frame: pd.DataFrame
    metrics: Dict[str, float]


def _metrics(returns: pd.Series, equity: pd.Series, positions: pd.Series) -> Dict[str, float]:
    returns = returns.fillna(0.0)
    ann_ret = returns.mean() * TRADING_DAYS
    ann_vol = returns.std(ddof=1) * np.sqrt(TRADING_DAYS)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    running_max = equity.cummax()
    dd = (equity - running_max) / running_max.replace(0.0, np.nan)
    max_dd = float(dd.min())
    trades = int((positions.diff().abs() > 0).sum())
    active = returns[positions.shift(1).fillna(0) != 0]
    win_rate = float((active > 0).mean()) if len(active) else 0.0
    return {
        "total_pnl": float(equity.iloc[-1] - equity.iloc[0]) if len(equity) else 0.0,
        "annualized_return": float(ann_ret),
        "annualized_volatility": float(ann_vol),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": max_dd,
        "num_trades": trades,
        "win_rate": win_rate,
    }


def backtest_pairs(
    price_a: pd.Series,
    price_b: pd.Series,
    spread: pd.Series,
    zscore: pd.Series,
    positions: pd.Series,
    hedge_ratio: float,
    cost_bps: float = 1.0,
    capital: float | None = None,
) -> BacktestResult:
    """Backtest a spread position series.

    Parameters
    ----------
    cost_bps : float
        Round-trip-agnostic per-trade transaction cost in basis points of the
        traded notional (charged on every position change).
    capital : float, optional
        Notional used to normalise PnL into returns. Defaults to the average
        gross notional of one spread unit, so returns are expressed relative to
        the capital actually tied up in the position.
    """
    df = pd.DataFrame(
        {"A": price_a, "B": price_b, "spread": spread, "zscore": zscore, "position": positions}
    ).dropna()

    # Notional of one unit of the spread (long 1 B, short beta A).
    notional = (df["B"].abs() + abs(hedge_ratio) * df["A"].abs())
    if capital is None:
        capital = float(notional.mean())

    # PnL from holding yesterday's position through today's spread change.
    spread_change = df["spread"].diff().fillna(0.0)
    gross_pnl = df["position"].shift(1).fillna(0.0) * spread_change

    # Transaction costs on position changes.
    turnover = df["position"].diff().abs().fillna(0.0)
    cost = turnover * notional * (cost_bps / 10_000.0)

    net_pnl = gross_pnl - cost
    df["pnl"] = net_pnl
    df["returns"] = net_pnl / capital
    df["equity"] = capital + net_pnl.cumsum()

    metrics = _metrics(df["returns"], df["equity"], df["position"])
    return BacktestResult(frame=df, metrics=metrics)
