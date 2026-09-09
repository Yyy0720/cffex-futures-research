from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint


@dataclass
class ADFResult:
    statistic: float
    pvalue: float
    usedlag: int
    nobs: int
    critical_values: dict[str, float]
    stationary_5pct: bool

    def to_dict(self) -> dict:
        d = asdict(self)
        d["critical_values"] = {str(k): float(v) for k, v in self.critical_values.items()}
        return d


@dataclass
class EngleGrangerResult:
    beta: float
    intercept: float
    adf: ADFResult
    coint_pvalue: float
    residual: pd.Series

    def to_dict(self) -> dict:
        return {
            "beta": self.beta,
            "intercept": self.intercept,
            "coint_pvalue": self.coint_pvalue,
            "adf": self.adf.to_dict(),
        }


def adf_stationarity(series: pd.Series, maxlag: int | None = None) -> ADFResult:
    x = series.dropna().astype(float)
    res = adfuller(x, maxlag=maxlag, autolag="AIC")
    crit = {k: float(v) for k, v in res[4].items()}
    return ADFResult(
        statistic=float(res[0]),
        pvalue=float(res[1]),
        usedlag=int(res[2]),
        nobs=int(res[3]),
        critical_values=crit,
        stationary_5pct=float(res[1]) < 0.05,
    )


def engle_granger(
    y: pd.Series,
    x: pd.Series,
) -> EngleGrangerResult:
    """
    Engle-Granger two-step: OLS y ~ x, ADF on residual.
    Also reports statsmodels coint() p-value for cross-check.
    """
    df = pd.concat([y.rename("y"), x.rename("x")], axis=1, join="inner").dropna()
    X = sm.add_constant(df["x"])
    model = sm.OLS(df["y"], X).fit()
    resid = model.resid
    resid.name = "eg_residual"
    adf = adf_stationarity(resid)
    coint_stat = coint(df["y"], df["x"], trend="c")
    return EngleGrangerResult(
        beta=float(model.params["x"]),
        intercept=float(model.params["const"]),
        adf=adf,
        coint_pvalue=float(coint_stat[1]),
        residual=resid,
    )


def half_life(series: pd.Series) -> float:
    """Ornstein-Uhlenbeck style half-life from AR(1) on lagged levels."""
    x = series.dropna().astype(float)
    if len(x) < 10:
        return float("nan")
    lag = x.shift(1)
    delta = x - lag
    df = pd.concat([delta.rename("d"), lag.rename("l")], axis=1).dropna()
    if df["l"].var() == 0:
        return float("nan")
    beta = np.polyfit(df["l"].values, df["d"].values, 1)[0]
    if beta >= 0:
        return float("inf")
    return float(-np.log(2.0) / beta)
