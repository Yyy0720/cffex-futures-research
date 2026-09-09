from __future__ import annotations

import numpy as np
import pandas as pd


def daily_log_returns(df: pd.DataFrame) -> pd.Series:
    close = df["close"].resample("1D").last().dropna()
    r = np.log(close).diff().dropna()
    r.name = "log_ret"
    return r


def rolling_volatility(
    df: pd.DataFrame,
    windows: list[int] | tuple[int, ...] = (20, 60),
    annualize: bool = True,
) -> pd.DataFrame:
    r = daily_log_returns(df)
    out = {}
    for w in windows:
        s = r.rolling(w).std()
        if annualize:
            s = s * np.sqrt(252)
        out[f"vol_{w}d"] = s
    return pd.DataFrame(out)


def rolling_correlation(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    windows: list[int] | tuple[int, ...] = (20, 60),
) -> pd.DataFrame:
    ra = daily_log_returns(df_a)
    rb = daily_log_returns(df_b)
    aligned = pd.concat([ra, rb], axis=1, join="inner")
    aligned.columns = ["a", "b"]
    out = {}
    for w in windows:
        out[f"corr_{w}d"] = aligned["a"].rolling(w).corr(aligned["b"])
    return pd.DataFrame(out, index=aligned.index)
