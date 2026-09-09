from __future__ import annotations

import numpy as np
import pandas as pd


def amihud_illiquidity(
    df: pd.DataFrame,
    sampling_minutes: int = 5,
) -> pd.Series:
    """
    Amihud illiquidity: |return| / volume (per bar), then daily average.
    Higher = worse liquidity / larger price impact per unit volume.
    """
    if sampling_minutes > 1:
        bars = df.resample(f"{sampling_minutes}min").agg(
            {"close": "last", "volume": "sum"}
        ).dropna()
    else:
        bars = df[["close", "volume"]].dropna()

    ret = bars["close"].pct_change().abs()
    vol = bars["volume"].replace(0, np.nan)
    illiq = (ret / vol).replace([np.inf, -np.inf], np.nan)
    daily = illiq.groupby(bars.index.normalize()).mean()
    daily.name = "amihud"
    return daily


def intraday_volume_oi_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Mean volume and open interest by clock minute."""
    tmp = df.copy()
    tmp["tod"] = tmp.index.strftime("%H:%M")
    profile = tmp.groupby("tod").agg(
        volume_mean=("volume", "mean"),
        oi_mean=("open_interest", "mean"),
        n=("volume", "count"),
    )
    return profile
