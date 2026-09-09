from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED = ("open", "high", "low", "close", "volume", "open_interest")


def clean_minute_bars(
    df: pd.DataFrame,
    *,
    drop_auction: bool = True,
    max_return: float = 0.05,
    fill_limit: int = 3,
) -> pd.DataFrame:
    """
    Clean equity-index futures minute bars.

    - drop auction-ish bars near open/close if requested
    - drop rows with missing OHLCV / non-positive prices
    - flag and drop extreme jumps (likely bad ticks)
    - forward-fill small gaps within a session (limited)
    """
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        raise ValueError("Expected DatetimeIndex")

    missing = [c for c in REQUIRED if c not in out.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    out = out[list(REQUIRED)].astype(float)
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["open", "high", "low", "close"])
    out = out[(out[["open", "high", "low", "close"]] > 0).all(axis=1)]

    # OHLC consistency
    bad_ohlc = (
        (out["high"] < out[["open", "close"]].max(axis=1))
        | (out["low"] > out[["open", "close"]].min(axis=1))
        | (out["high"] < out["low"])
    )
    out = out.loc[~bad_ohlc]

    if drop_auction:
        t = out.index.time
        # Keep continuous auction window; drop first/last minute of each session half
        keep = ~(
            ((t == pd.Timestamp("09:30").time()) | (t == pd.Timestamp("11:30").time()))
            | ((t == pd.Timestamp("13:00").time()) | (t == pd.Timestamp("15:00").time()))
        )
        out = out.loc[keep]

    ret = out["close"].pct_change().abs()
    out = out.loc[ret.isna() | (ret <= max_return)]

    # Limited ffill for volume/OI only (prices already cleaned)
    out[["volume", "open_interest"]] = (
        out[["volume", "open_interest"]].ffill(limit=fill_limit).fillna(0.0)
    )
    return out
