from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from .cointegration import half_life


def rolling_adf_pvalue(series: pd.Series, window: int = 60) -> pd.Series:
    """Rolling ADF p-value; lower ⇒ stronger evidence of stationarity in-window."""
    x = series.dropna().astype(float)
    out = pd.Series(index=x.index, dtype=float, name="adf_pvalue")
    vals = x.values
    idx = x.index
    for i in range(window - 1, len(vals)):
        chunk = vals[i - window + 1 : i + 1]
        if np.std(chunk) < 1e-12:
            out.iloc[i] = np.nan
            continue
        try:
            out.iloc[i] = float(adfuller(chunk, autolag="AIC")[1])
        except Exception:
            out.iloc[i] = np.nan
    return out


def rolling_half_life(series: pd.Series, window: int = 60) -> pd.Series:
    x = series.dropna().astype(float)
    out = pd.Series(index=x.index, dtype=float, name="half_life")
    for i in range(window - 1, len(x)):
        chunk = x.iloc[i - window + 1 : i + 1]
        out.iloc[i] = half_life(chunk)
    return out
