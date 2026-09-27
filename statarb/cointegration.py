r"""Cointegration testing and hedge-ratio estimation.

The **Engle-Granger** two-step procedure:

1. Estimate the cointegrating relationship by OLS,
   :math:`P^B_t = \alpha + \beta P^A_t + \varepsilon_t`, giving the hedge ratio
   :math:`\hat\beta`.
2. Test the residuals :math:`\hat\varepsilon_t` for a unit root with the
   Augmented Dickey-Fuller (ADF) test. Rejecting the unit root null implies the
   residual (the spread) is stationary, i.e. the pair is cointegrated.

We also expose the spread's **half-life** of mean reversion, estimated from an
AR(1) fit — a key input for sizing holding periods.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint


@dataclass
class CointegrationResult:
    """Outcome of an Engle-Granger cointegration test."""

    hedge_ratio: float
    intercept: float
    adf_stat: float
    adf_pvalue: float
    eg_pvalue: float
    is_cointegrated: bool
    half_life: float


def estimate_hedge_ratio(y: pd.Series, x: pd.Series):
    """OLS hedge ratio and intercept for ``y = alpha + beta*x + eps``.

    Returns
    -------
    (beta, alpha, residuals)
    """
    X = sm.add_constant(x.values)
    model = sm.OLS(y.values, X).fit()
    alpha, beta = model.params[0], model.params[1]
    residuals = pd.Series(y.values - (alpha + beta * x.values), index=y.index)
    return float(beta), float(alpha), residuals


def estimate_half_life(spread: pd.Series) -> float:
    r"""Estimate the mean-reversion half-life from an AR(1) fit.

    Regress :math:`\Delta s_t` on :math:`s_{t-1}`; if the slope is
    :math:`\lambda`, the half-life is :math:`-\ln 2 / \ln(1+\lambda)`.
    """
    spread = spread.dropna()
    lagged = spread.shift(1).dropna()
    delta = spread.diff().dropna()
    lagged = lagged.loc[delta.index]
    X = sm.add_constant(lagged.values)
    model = sm.OLS(delta.values, X).fit()
    lam = model.params[1]
    if lam >= 0:
        return float("inf")  # not mean-reverting
    return float(-np.log(2.0) / np.log(1.0 + lam))


def engle_granger_test(
    y: pd.Series, x: pd.Series, significance: float = 0.05
) -> CointegrationResult:
    """Run the full Engle-Granger cointegration test on two price series."""
    beta, alpha, residuals = estimate_hedge_ratio(y, x)

    # ADF test on the OLS residuals (the candidate spread).
    adf_stat, adf_p, *_ = adfuller(residuals.values, autolag="AIC", result_object=False)

    # statsmodels' direct cointegration test as a corroborating p-value.
    eg_stat, eg_p, _ = coint(y.values, x.values)

    hl = estimate_half_life(residuals)

    return CointegrationResult(
        hedge_ratio=beta,
        intercept=alpha,
        adf_stat=float(adf_stat),
        adf_pvalue=float(adf_p),
        eg_pvalue=float(eg_p),
        is_cointegrated=bool(adf_p < significance),
        half_life=hl,
    )
