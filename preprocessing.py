# processing.py
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def calculate_log_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Log return giống Excel: ln(Pt / Pt-1)
    """
    log_returns = np.log(price_df / price_df.shift(1))
    return log_returns.dropna(how="any")


def estimate_betas(stock_returns: pd.DataFrame, market_returns: pd.Series) -> pd.Series:
    """
    Ước lượng beta từng cổ phiếu theo CAPM:
    Ri = alpha + beta * Rm
    """
    betas = {}

    X = market_returns.values.reshape(-1, 1)

    for symbol in stock_returns.columns:
        y = stock_returns[symbol].values
        model = LinearRegression()
        model.fit(X, y)
        betas[symbol] = model.coef_[0]

    return pd.Series(betas)


def estimate_market_parameters(market_returns: pd.Series):
    """
    Tính:
    - E(Rm): log return trung bình * 365
    - σ²(M): variance log return
    """
    expected_rm = market_returns.mean() * 365
    market_variance = market_returns.var()

    return expected_rm, market_variance


def capm_expected_returns(
    betas: pd.Series,
    expected_rm: float,
    rf: float
) -> pd.Series:
    """
    E(Ri) = rf + beta_i * (E(Rm) - rf)
    """
    excess_market_return = expected_rm - rf
    expected_returns = rf + betas * excess_market_return
    return expected_returns


def capm_covariance_matrix(
    betas: pd.Series,
    market_variance: float
) -> pd.DataFrame:
    """
    Σ = β βᵀ σ²(M)
    """
    beta_vec = betas.values.reshape(-1, 1)
    cov = market_variance * (beta_vec @ beta_vec.T)

    return pd.DataFrame(
        cov,
        index=betas.index,
        columns=betas.index
    )
