from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import project_root
from src.data.io import save_parquet


def _session_minutes(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Build 1-min timestamps for China equity futures sessions."""
    morning = pd.date_range("09:30", "11:29", freq="1min").time
    afternoon = pd.date_range("13:00", "14:59", freq="1min").time
    stamps: list[pd.Timestamp] = []
    for d in dates:
        day = d.normalize()
        for t in list(morning) + list(afternoon):
            stamps.append(day + pd.Timedelta(hours=t.hour, minutes=t.minute))
    return pd.DatetimeIndex(stamps)


def generate_sample_bars(
    symbol: str = "IF",
    start: str = "2024-01-02",
    n_days: int = 60,
    seed: int = 42,
    start_price: float = 3600.0,
) -> pd.DataFrame:
    """
    Synthetic minute bars with:
    - intraday vol U-shape (higher near open/close)
    - mild vol clustering across days
    - volume / open interest patterns
    """
    rng = np.random.default_rng(seed + hash(symbol) % 10_000)
    biz_days = pd.bdate_range(start, periods=n_days)
    idx = _session_minutes(biz_days)
    n = len(idx)

    # Day-level vol factor (clustering)
    day_vol = np.exp(np.cumsum(rng.normal(0, 0.08, size=n_days)))
    day_vol = day_vol / day_vol.mean() * 0.0008
    day_map = {d.normalize(): day_vol[i] for i, d in enumerate(biz_days)}

    minutes_from_open = []
    session_len = 240  # 120 + 120
    for ts in idx:
        if ts.time() < pd.Timestamp("12:00").time():
            m = (ts.hour - 9) * 60 + ts.minute - 30
        else:
            m = 120 + (ts.hour - 13) * 60 + ts.minute
        minutes_from_open.append(m)
    mfo = np.asarray(minutes_from_open, dtype=float)
    # U-shape multiplier
    u = 1.0 + 1.2 * np.exp(-mfo / 25) + 1.0 * np.exp(-(session_len - mfo) / 30)

    base_vol = np.array([day_map[ts.normalize()] for ts in idx]) * u
    rets = rng.normal(0.0, base_vol)
    close = start_price * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[start_price], close[:-1]])
    noise = np.abs(rng.normal(0, base_vol * start_price * 0.3, size=n))
    high = np.maximum(open_, close) + noise
    low = np.minimum(open_, close) - noise

    # Volume: higher near open/close; OI drifts slowly
    vol_profile = 800 + 2200 * (np.exp(-mfo / 20) + 0.7 * np.exp(-(session_len - mfo) / 25))
    volume = rng.poisson(vol_profile).astype(float)
    oi = 120_000 + np.cumsum(rng.normal(0, 40, size=n))
    oi = np.maximum(oi, 50_000)

    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "open_interest": oi,
        },
        index=idx,
    )
    df.index.name = "datetime"
    return df


def _ou_basis(n_days: int, *, mean: float, kappa: float, sigma: float, seed: int) -> np.ndarray:
    """Daily OU basis path so z-score strategies can fire on sample data."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n_days)
    x[0] = mean
    for t in range(1, n_days):
        x[t] = x[t - 1] + kappa * (mean - x[t - 1]) + sigma * rng.normal()
    return x


def _spot_from_futures_with_basis(fut: pd.DataFrame, daily_basis: np.ndarray) -> pd.DataFrame:
    """Map daily relative basis onto minute bars: spot = futures / (1 + basis)."""
    days = pd.DatetimeIndex(sorted(fut.index.normalize().unique()))
    if len(daily_basis) != len(days):
        raise ValueError("daily_basis length must match number of sessions")
    basis_map = pd.Series(daily_basis, index=days)
    b = fut.index.normalize().map(basis_map).astype(float)
    spot = fut.copy()
    spot["close"] = fut["close"] / (1.0 + b)
    spot["open"] = spot["close"].shift(1).fillna(spot["close"].iloc[0])
    spot["high"] = np.maximum(spot["open"], spot["close"]) * 1.0003
    spot["low"] = np.minimum(spot["open"], spot["close"]) * 0.9997
    return spot


def write_sample_dataset(
    out_dir: str | Path | None = None,
    n_days: int = 252,
) -> dict[str, Path]:
    """Write IF / IH / spot sample parquet files for local development."""
    out = Path(out_dir) if out_dir else project_root() / "data" / "sample"
    out.mkdir(parents=True, exist_ok=True)

    if_bars = generate_sample_bars("IF", n_days=n_days, seed=1, start_price=3600)
    ih_bars = generate_sample_bars("IH", n_days=n_days, seed=2, start_price=2500)

    # Mean-reverting relative basis with enough amplitude for |z|>2 entries
    csi = _spot_from_futures_with_basis(
        if_bars,
        _ou_basis(n_days, mean=0.002, kappa=0.35, sigma=0.0012, seed=11),
    )
    sse = _spot_from_futures_with_basis(
        ih_bars,
        _ou_basis(n_days, mean=0.0015, kappa=0.40, sigma=0.0010, seed=22),
    )

    paths = {
        "IF": save_parquet(if_bars, out / "IF_1min.parquet"),
        "IH": save_parquet(ih_bars, out / "IH_1min.parquet"),
        "CSI300": save_parquet(csi, out / "CSI300_1min.parquet"),
        "SSE50": save_parquet(sse, out / "SSE50_1min.parquet"),
    }
    return paths
