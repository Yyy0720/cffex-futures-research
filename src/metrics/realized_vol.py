from __future__ import annotations

import numpy as np
import pandas as pd


def _to_bars(df: pd.DataFrame, sampling_minutes: int) -> pd.DataFrame:
    if sampling_minutes <= 1:
        return df
    ohlc = df.resample(f"{sampling_minutes}min").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "open_interest": "last",
        }
    )
    return ohlc.dropna(subset=["open", "high", "low", "close"])


def daily_realized_variance(
    df: pd.DataFrame,
    sampling_minutes: int = 5,
) -> pd.Series:
    """Sum of squared log returns within each day (RV proxy)."""
    bars = _to_bars(df, sampling_minutes)
    log_ret = np.log(bars["close"]).diff()
    rv = log_ret.pow(2).groupby(bars.index.normalize()).sum()
    rv.name = "rv"
    return rv


def close_to_close_vol(df: pd.DataFrame, annualize: bool = True) -> pd.Series:
    """Daily close-to-close volatility from last minute close each day."""
    daily_close = df["close"].resample("1D").last().dropna()
    r = np.log(daily_close).diff()
    # Rolling 1-day absolute as series of daily vol estimates is not meaningful;
    # return daily |r| and optional annualization factor applied later by caller.
    vol = r.abs()
    if annualize:
        vol = vol * np.sqrt(252)
    vol.name = "ctc_vol"
    return vol


def garman_klass_vol(df: pd.DataFrame, annualize: bool = True) -> pd.Series:
    """
    Garman-Klass daily variance estimator using OHLC.
    sigma^2 = 0.5*(ln H/L)^2 - (2ln2 - 1)*(ln C/O)^2
    """
    daily = df.resample("1D").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna()
    log_hl = np.log(daily["high"] / daily["low"])
    log_co = np.log(daily["close"] / daily["open"])
    var = 0.5 * log_hl.pow(2) - (2.0 * np.log(2.0) - 1.0) * log_co.pow(2)
    var = var.clip(lower=0.0)
    vol = np.sqrt(var)
    if annualize:
        vol = vol * np.sqrt(252)
    vol.name = "gk_vol"
    return vol


def intraday_rv_profile(
    df: pd.DataFrame,
    sampling_minutes: int = 5,
) -> pd.DataFrame:
    """Average squared return by clock time (intraday RV pattern)."""
    bars = _to_bars(df, sampling_minutes)
    r2 = np.log(bars["close"]).diff().pow(2)
    tmp = pd.DataFrame({"r2": r2, "tod": bars.index.strftime("%H:%M")}).dropna()
    profile = tmp.groupby("tod")["r2"].agg(["mean", "median", "count"])
    profile = profile.rename(columns={"mean": "mean_r2", "median": "median_r2"})
    return profile
