from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.common.config import project_root


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else project_root() / p


def load_minute_bars(path: str | Path) -> pd.DataFrame:
    """Load minute bars from parquet or csv. Expects a datetime column or index."""
    path = _resolve(path)
    if not path.exists():
        raise FileNotFoundError(f"Bar file not found: {path}")

    if path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime")
    elif not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Bars need a datetime column or DatetimeIndex")

    df = df.sort_index()
    df.index.name = "datetime"
    return df


def save_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    path = _resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    if isinstance(out.index, pd.DatetimeIndex):
        out = out.reset_index()
    out.to_parquet(path, index=False)
    return path
