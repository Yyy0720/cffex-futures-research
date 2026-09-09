from __future__ import annotations

import numpy as np
import pandas as pd


def compute_basis(
    futures: pd.Series,
    spot: pd.Series,
    *,
    mode: str = "relative",
) -> pd.Series:
    """
    Basis between futures and spot.

    mode='relative': futures/spot - 1  (scale-free, preferred)
    mode='absolute': futures - spot
    """
    aligned = pd.concat([futures.rename("f"), spot.rename("s")], axis=1, join="inner").dropna()
    if mode == "relative":
        basis = aligned["f"] / aligned["s"] - 1.0
    elif mode == "absolute":
        basis = aligned["f"] - aligned["s"]
    else:
        raise ValueError("mode must be 'relative' or 'absolute'")
    basis.name = "basis"
    return basis


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mu = series.rolling(window).mean()
    sd = series.rolling(window).std(ddof=0)
    z = (series - mu) / sd.replace(0.0, np.nan)
    z.name = "zscore"
    return z
