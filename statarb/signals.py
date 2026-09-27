r"""Spread construction and z-score trading signals.

The spread is :math:`s_t = P^B_t - \beta P^A_t` (the OLS residual). Trading is
driven by its rolling **z-score**

.. math:: z_t = \frac{s_t - \mu_t}{\sigma_t},

where :math:`\mu_t, \sigma_t` are a rolling mean and standard deviation. The
rule is classic mean reversion:

* enter **short the spread** when :math:`z_t > z_{\text{entry}}` (spread rich),
* enter **long the spread** when :math:`z_t < -z_{\text{entry}}` (spread cheap),
* **exit** when :math:`|z_t|` falls below :math:`z_{\text{exit}}`.

"Long the spread" = long B, short :math:`\beta` units of A (and vice versa).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_spread(price_b: pd.Series, price_a: pd.Series, hedge_ratio: float) -> pd.Series:
    """Spread s = B - beta*A."""
    return price_b - hedge_ratio * price_a


def zscore(spread: pd.Series, window: int = 30) -> pd.Series:
    """Rolling z-score of the spread.

    A ``window`` of 0 uses the full-sample (static) mean and std instead.
    """
    if window and window > 1:
        mu = spread.rolling(window).mean()
        sigma = spread.rolling(window).std(ddof=0)
    else:
        mu = spread.expanding().mean()
        sigma = spread.expanding().std(ddof=0)
    return (spread - mu) / sigma.replace(0.0, np.nan)


def generate_positions(
    z: pd.Series, entry: float = 2.0, exit: float = 0.5
) -> pd.Series:
    """Convert a z-score series into spread positions in {-1, 0, +1}.

    +1 = long spread, -1 = short spread, 0 = flat. Positions are held until an
    exit threshold is crossed (stateful), avoiding whipsaw around the bands.
    """
    positions = np.zeros(len(z))
    current = 0
    z_values = z.values
    for i, zv in enumerate(z_values):
        if np.isnan(zv):
            positions[i] = current
            continue
        if current == 0:
            if zv > entry:
                current = -1  # spread too high -> short it
            elif zv < -entry:
                current = 1   # spread too low -> long it
        elif current == 1 and zv >= -exit:
            current = 0
        elif current == -1 and zv <= exit:
            current = 0
        positions[i] = current
    return pd.Series(positions, index=z.index)
