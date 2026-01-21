# optimizer2.py
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def optimize_capm_portfolio(
    expected_returns: pd.Series,
    cov: pd.DataFrame,
    rf: float,
    allow_short: bool = False
) -> pd.Series:
    """
    Excel Solver 1–1:
    Max Sharpe = (w^T μ - rf) / sqrt(w^T Σ w)
    """

    mu = expected_returns.values
    Sigma = cov.values
    n = len(mu)

    def negative_sharpe(w):
        port_return = np.dot(w, mu)
        port_var = w.T @ Sigma @ w
        port_vol = np.sqrt(port_var)

        if port_vol <= 1e-10:
            return 1e6

        sharpe = (port_return - rf) / port_vol
        return -sharpe

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    ]

    bounds = None if allow_short else [(0.0, 1.0)] * n

    x0 = np.ones(n) / n  # giống Excel

    res = minimize(
        negative_sharpe,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 1000}
    )

    if not res.success:
        raise RuntimeError(res.message)

    w = res.x
    w = np.clip(w, 0, None)
    w = w / w.sum()

    return pd.Series(w, index=expected_returns.index)
