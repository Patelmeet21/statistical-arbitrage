"""Tests for cointegration, hedge-ratio recovery, signals and backtest."""

import numpy as np
import pandas as pd
import pytest

from statarb.backtest import backtest_pairs
from statarb.cointegration import (
    engle_granger_test, estimate_half_life, estimate_hedge_ratio,
)
from statarb.data import generate_cointegrated_pair, generate_independent_pair, half_life
from statarb.signals import compute_spread, generate_positions, zscore


# ---------------------------- Cointegration ---------------------------- #
def test_cointegrated_pair_is_detected():
    df = generate_cointegrated_pair(n=1000, beta=1.5, seed=1)
    res = engle_granger_test(df["B"], df["A"])
    assert res.is_cointegrated
    assert res.adf_pvalue < 0.05


def test_independent_pair_not_cointegrated_usually():
    # Independent random walks should (almost always) fail the test.
    fails = 0
    for seed in range(10):
        df = generate_independent_pair(n=800, seed=seed)
        res = engle_granger_test(df["B"], df["A"])
        if not res.is_cointegrated:
            fails += 1
    assert fails >= 8  # overwhelming majority correctly rejected


def test_hedge_ratio_recovered():
    df = generate_cointegrated_pair(n=2000, beta=1.5, gamma=10.0, seed=3)
    beta, alpha, resid = estimate_hedge_ratio(df["B"], df["A"])
    assert beta == pytest.approx(1.5, abs=0.1)
    assert len(resid) == len(df)


def test_half_life_positive_for_mean_reverting():
    df = generate_cointegrated_pair(n=2000, kappa=0.05, seed=5)
    _, _, resid = estimate_hedge_ratio(df["B"], df["A"])
    hl = estimate_half_life(resid)
    assert 0 < hl < np.inf


def test_half_life_formula():
    assert half_life(np.log(2)) == pytest.approx(1.0)


# ------------------------------- Signals ------------------------------- #
def test_zscore_is_standardized():
    s = pd.Series(np.random.default_rng(0).normal(0, 1, 500))
    z = zscore(s, window=0).dropna()  # expanding, full sample
    # A standardized series has near-zero mean and unit-ish scale.
    assert abs(z.mean()) < 0.5


def test_positions_are_ternary():
    z = pd.Series([0, 3, 3, 0.2, -3, -0.1, 0])
    pos = generate_positions(z, entry=2.0, exit=0.5)
    assert set(pos.unique()).issubset({-1, 0, 1})


def test_positions_enter_and_exit():
    # Rises above +entry -> short (-1); reverts below exit -> flat (0).
    z = pd.Series([0.0, 2.5, 2.5, 0.3, 0.0])
    pos = generate_positions(z, entry=2.0, exit=0.5)
    assert pos.iloc[1] == -1
    assert pos.iloc[3] == 0


def test_long_spread_when_cheap():
    z = pd.Series([0.0, -2.5, -2.5, -0.3])
    pos = generate_positions(z, entry=2.0, exit=0.5)
    assert pos.iloc[1] == 1


# ------------------------------ Backtest ------------------------------- #
def test_backtest_runs_and_has_metrics():
    df = generate_cointegrated_pair(n=1000, seed=42)
    res = engle_granger_test(df["B"], df["A"])
    spread = compute_spread(df["B"], df["A"], res.hedge_ratio)
    z = zscore(spread, window=30)
    pos = generate_positions(z)
    bt = backtest_pairs(df["A"], df["B"], spread, z, pos, res.hedge_ratio, cost_bps=1.0)
    for key in ["sharpe_ratio", "max_drawdown", "num_trades", "win_rate", "total_pnl"]:
        assert key in bt.metrics
    assert bt.metrics["num_trades"] > 0


def test_transaction_costs_reduce_pnl():
    df = generate_cointegrated_pair(n=1000, seed=42)
    res = engle_granger_test(df["B"], df["A"])
    spread = compute_spread(df["B"], df["A"], res.hedge_ratio)
    z = zscore(spread, window=30)
    pos = generate_positions(z)
    free = backtest_pairs(df["A"], df["B"], spread, z, pos, res.hedge_ratio, cost_bps=0.0)
    costly = backtest_pairs(df["A"], df["B"], spread, z, pos, res.hedge_ratio, cost_bps=5.0)
    assert costly.metrics["total_pnl"] < free.metrics["total_pnl"]


def test_strategy_profitable_on_clean_synthetic_pair():
    # On a strongly mean-reverting, clearly cointegrated pair the strategy
    # should be profitable before/after modest costs.
    df = generate_cointegrated_pair(n=1500, beta=1.5, kappa=0.08, spread_sigma=1.0, seed=11)
    res = engle_granger_test(df["B"], df["A"])
    spread = compute_spread(df["B"], df["A"], res.hedge_ratio)
    z = zscore(spread, window=30)
    pos = generate_positions(z, entry=1.5, exit=0.3)
    bt = backtest_pairs(df["A"], df["B"], spread, z, pos, res.hedge_ratio, cost_bps=1.0)
    assert bt.metrics["total_pnl"] > 0
