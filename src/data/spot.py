"""Load and normalize spot index daily bars from teacher files."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.common.config import project_root
from src.data.io import save_parquet

# futures variety -> (spot symbol key, raw filename stem prefix)
SPOT_FILES = {
    "CSI300": "spot_CSI300_000300.parquet",
    "SSE50": "spot_SSE50_000016.parquet",
    "CSI500": "spot_CSI500_000905.parquet",
    "CSI1000": "spot_CSI1000_000852.parquet",
}

FUTURES_TO_SPOT = {
    "IF": "CSI300",
    "IH": "SSE50",
    "IC": "CSI500",
    "IM": "CSI1000",
}


def load_spot_daily(symbol: str, path: str | Path | None = None) -> pd.DataFrame:
    """Return DatetimeIndex OHLCV daily bars for a spot index symbol."""
    root = project_root()
    symbol = symbol.upper()
    if path is None:
        # prefer processed, else raw
        processed = root / "data" / "processed" / f"{symbol}_daily.parquet"
        if processed.exists():
            path = processed
        else:
            fname = SPOT_FILES.get(symbol)
            if not fname:
                raise KeyError(f"Unknown spot symbol: {symbol}")
            path = root / "data" / "raw" / fname
    path = Path(path)
    df = pd.read_parquet(path)
    # normalize columns
    cols = {c.lower(): c for c in df.columns}
    date_col = cols.get("date") or cols.get("datetime") or cols.get("tradingday")
    if date_col is None:
        raise ValueError(f"No date column in {path}")
    out = pd.DataFrame(
        {
            "open": df[cols.get("open", "open")].astype(float),
            "high": df[cols.get("high", "high")].astype(float),
            "low": df[cols.get("low", "low")].astype(float),
            "close": df[cols.get("close", "close")].astype(float),
            "volume": df[cols.get("volume", "volume")].astype(float)
            if "volume" in cols
            else 0.0,
        }
    )
    out.index = pd.to_datetime(df[date_col])
    out.index.name = "datetime"
    out = out.sort_index()
    if "open_interest" not in out.columns:
        out["open_interest"] = 0.0
    return out


def write_processed_spot() -> dict[str, Path]:
    out_dir = project_root() / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for sym in SPOT_FILES:
        df = load_spot_daily(sym)
        paths[sym] = save_parquet(df, out_dir / f"{sym}_daily.parquet")
        print(f"{sym}: {len(df):,} days | {df.index.min().date()} -> {df.index.max().date()}")
    return paths


if __name__ == "__main__":
    write_processed_spot()
