r"""Synthetic data generation for pairs trading.

Two assets are *cointegrated* if a linear combination of their (individually
non-stationary) price levels is stationary. We generate such a pair explicitly:

.. math::

    P^A_t = \text{random walk}, \qquad
    P^B_t = \beta P^A_t + \gamma + u_t,

where the spread :math:`u_t` is a mean-reverting Ornstein-Uhlenbeck process.
This gives a ground-truth hedge ratio :math:`\beta` and half-life so tests can
check that the estimators recover the truth.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_cointegrated_pair(
    n: int = 1000,
    beta: float = 1.5,
    gamma: float = 10.0,
    spread_sigma: float = 1.0,
    kappa: float = 0.05,
    seed: int = 42,
    start_date: str = "2018-01-01",
    start_price: float = 50.0,
) -> pd.DataFrame:
    r"""Generate a cointegrated price pair (A, B).

    Parameters
    ----------
    n : int
        Number of observations.
    beta : float
        True hedge ratio (B on A).
    gamma : float
        Intercept of the cointegrating relationship.
    spread_sigma : float
        Innovation volatility of the OU spread.
    kappa : float
        OU mean-reversion speed (higher = faster reversion). The spread
        half-life is ``ln(2)/kappa``.
    seed : int
    """
    rng = np.random.default_rng(seed)

    # Asset A follows a random walk (unit root, non-stationary).
    a_returns = rng.normal(0.0, 1.0, size=n)
    price_a = start_price + np.cumsum(a_returns)
    price_a = np.maximum(price_a, 1.0)  # keep prices positive

    # Mean-reverting OU spread.
    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = spread[t - 1] - kappa * spread[t - 1] + rng.normal(0.0, spread_sigma)

    price_b = beta * price_a + gamma + spread
    price_b = np.maximum(price_b, 1.0)

    dates = pd.bdate_range(start=start_date, periods=n)
    return pd.DataFrame({"A": price_a, "B": price_b}, index=dates)


def generate_independent_pair(n: int = 1000, seed: int = 0, start_date="2018-01-01"):
    """Generate two *independent* random walks (NOT cointegrated).

    Useful as a negative control for the cointegration test.
    """
    rng = np.random.default_rng(seed)
    a = 50 + np.cumsum(rng.normal(0, 1, n))
    b = 50 + np.cumsum(rng.normal(0, 1, n))
    dates = pd.bdate_range(start=start_date, periods=n)
    return pd.DataFrame({"A": np.maximum(a, 1), "B": np.maximum(b, 1)}, index=dates)


def half_life(kappa: float) -> float:
    """Half-life of an OU process with mean-reversion speed ``kappa``."""
    return float(np.log(2.0) / kappa)
