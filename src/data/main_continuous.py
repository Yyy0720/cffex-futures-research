"""Build main-continuous 1-min bars from multi-contract minute data + main map."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.common.config import project_root
from src.data.io import save_parquet


RAW_COL_MAP = {
    "Time": "datetime",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume",
    "OpenInterest": "open_interest",
}


def load_raw_minute(path: str | Path | None = None) -> pd.DataFrame:
    root = project_root()
    path = Path(path) if path else root / "data" / "raw" / "min1_IF_IM_IC_IH.parquet"
    df = pd.read_parquet(path)
    df = df.rename(columns=RAW_COL_MAP)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["TradingDay"] = pd.to_datetime(df["TradingDay"])
    df["Variety"] = df["Variety"].astype(str).str.upper()
    df["Contract"] = df["Contract"].astype(str).str.upper()
    return df


def load_main_map(path: str | Path | None = None) -> pd.DataFrame:
    root = project_root()
    path = (
        Path(path)
        if path
        else root / "data" / "raw" / "fact_main_contract_20260727172524.parquet"
    )
    m = pd.read_parquet(path)
    m = m.rename(
        columns={
            "VarietyCode": "Variety",
            "ContractCode": "Contract",
            "StartDate": "StartDate",
            "EndDate": "EndDate",
        }
    )
    m["Variety"] = m["Variety"].astype(str).str.upper()
    m["Contract"] = m["Contract"].astype(str).str.upper()
    m["StartDate"] = pd.to_datetime(m["StartDate"])
    m["EndDate"] = pd.to_datetime(m["EndDate"])
    # keep CFFEX index futures only
    m = m[m["Variety"].isin(["IF", "IH", "IC", "IM"])].copy()
    # if overlapping segments exist, keep the one with latest EndDate then longest span
    m["span"] = (m["EndDate"] - m["StartDate"]).dt.days
    m = m.sort_values(["Variety", "StartDate", "EndDate", "span"])
    m = m.drop_duplicates(subset=["Variety", "StartDate", "EndDate", "Contract"], keep="last")
    return m[["Variety", "Contract", "StartDate", "EndDate"]]


def _daily_main_schedule(main_map: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
    """Expand main-map intervals to (Variety, TradingDay) -> Contract."""
    rows: list[pd.DataFrame] = []
    days = pd.DatetimeIndex(sorted(pd.to_datetime(days).unique()))
    for variety, g in main_map.groupby("Variety"):
        g = g.sort_values(["StartDate", "EndDate"])
        # build day->contract via interval coverage; later intervals overwrite earlier on overlap
        series = pd.Series(index=days, dtype="object")
        for _, r in g.iterrows():
            mask = (days >= r["StartDate"]) & (days <= r["EndDate"])
            series.loc[mask] = r["Contract"]
        part = series.dropna().rename("Contract").reset_index()
        part.columns = ["TradingDay", "Contract"]
        part["Variety"] = variety
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


def build_main_continuous(
    minute: pd.DataFrame | None = None,
    main_map: pd.DataFrame | None = None,
    varieties: list[str] | None = None,
    start: str | None = "2024-01-01",
    end: str | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Return dict variety -> DatetimeIndex OHLCV+OI main-continuous minute bars.
    Roll gaps: no adjustment (raw price splice); fine for RV/liquidity, noted in docs.
    """
    minute = load_raw_minute() if minute is None else minute
    main_map = load_main_map() if main_map is None else main_map
    varieties = varieties or ["IF", "IH", "IC", "IM"]

    m = minute[minute["Variety"].isin(varieties)].copy()
    if start:
        m = m[m["TradingDay"] >= pd.Timestamp(start)]
    if end:
        m = m[m["TradingDay"] <= pd.Timestamp(end)]

    schedule = _daily_main_schedule(main_map, m["TradingDay"].unique())
    merged = m.merge(schedule, on=["Variety", "TradingDay", "Contract"], how="inner")

    out: dict[str, pd.DataFrame] = {}
    for variety, g in merged.groupby("Variety"):
        bars = (
            g.sort_values("datetime")
            .drop_duplicates(subset=["datetime"], keep="last")
            .set_index("datetime")[
                ["open", "high", "low", "close", "volume", "open_interest"]
            ]
            .astype(float)
        )
        bars.index.name = "datetime"
        out[str(variety)] = bars
    return out


def write_processed_main_continuous(
    start: str = "2024-01-01",
    end: str | None = None,
    out_dir: str | Path | None = None,
) -> dict[str, Path]:
    out_dir = Path(out_dir) if out_dir else project_root() / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    series = build_main_continuous(start=start, end=end)
    paths: dict[str, Path] = {}
    for variety, df in series.items():
        paths[variety] = save_parquet(df, out_dir / f"{variety}_1min.parquet")
        print(f"{variety}: {len(df):,} bars | {df.index.min()} -> {df.index.max()} -> {paths[variety]}")
    return paths


if __name__ == "__main__":
    write_processed_main_continuous(start="2024-01-01")
