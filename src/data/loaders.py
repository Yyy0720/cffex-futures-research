from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.common.config import project_root
from src.data.clean import clean_minute_bars
from src.data.io import load_minute_bars
from src.data.sample import write_sample_dataset
from src.data.spot import SPOT_FILES, load_spot_daily


def load_symbol_bars(symbol: str, use_sample: bool, cfg: dict) -> pd.DataFrame:
    """Load + clean 1-min bars from sample / processed / raw (futures only)."""
    root = project_root()
    if use_sample:
        path = root / cfg["data"]["sample_dir"] / f"{symbol}_1min.parquet"
        if not path.exists():
            write_sample_dataset()
    else:
        path = root / cfg["data"]["processed_dir"] / f"{symbol}_1min.parquet"
        if not path.exists():
            raw = root / cfg["data"]["raw_dir"] / f"{symbol}_1min.parquet"
            alt = root / cfg["data"]["raw_dir"] / f"{symbol}_1min.csv"
            path = raw if raw.exists() else alt
    return clean_minute_bars(load_minute_bars(path))


def daily_close(df: pd.DataFrame) -> pd.Series:
    s = df["close"].resample("1D").last().dropna()
    s.name = "close"
    return s


def load_daily_close(symbol: str, use_sample: bool, cfg: dict) -> pd.Series:
    """
    Daily close for futures (from minute bars) or spot indices (from daily files).
    Spot symbols: CSI300, SSE50, CSI500, CSI1000.
    """
    symbol = symbol.upper()
    if symbol in SPOT_FILES:
        if use_sample:
            # sample generator may have CSI300/SSE50 minute proxies
            try:
                return daily_close(load_symbol_bars(symbol, use_sample=True, cfg=cfg))
            except FileNotFoundError:
                pass
        spot = load_spot_daily(symbol)
        s = spot["close"].copy()
        s.name = "close"
        return s
    return daily_close(load_symbol_bars(symbol, use_sample=use_sample, cfg=cfg))
