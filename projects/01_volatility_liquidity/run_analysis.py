"""
Project 1: Realized Volatility, Rolling Statistics & Liquidity Dynamics.

Usage (from repo root, venv active):
  # Step A — build main-continuous bars (once)
  python scripts/build_main_continuous.py --start 2024-01-01

  # Step B — run analysis on real data
  python projects/01_volatility_liquidity/run_analysis.py --real-data
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.config import load_config, project_root
from src.data.clean import clean_minute_bars
from src.data.loaders import load_symbol_bars
from src.metrics.liquidity import amihud_illiquidity, intraday_volume_oi_profile
from src.metrics.realized_vol import (
    close_to_close_vol,
    daily_realized_variance,
    garman_klass_vol,
    intraday_rv_profile,
)
from src.metrics.rolling import rolling_correlation, rolling_volatility


def run(use_sample: bool = False) -> Path:
    cfg = load_config()
    p1 = cfg["project_01"]
    varieties = list(p1.get("varieties", ["IF", "IH", "IC", "IM"]))
    minutes = int(p1["rv_sampling_minutes"])
    windows = list(p1["rolling_windows_days"])

    out_dir = project_root() / p1["results_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    bars = {v: load_symbol_bars(v, use_sample=use_sample, cfg=cfg) for v in varieties}
    # optional date filter on cleaned bars
    start = p1.get("analysis_start")
    if start and not use_sample:
        for v in list(bars):
            bars[v] = bars[v][bars[v].index >= pd.Timestamp(start)]

    # --- daily metrics ---
    daily_cols = {}
    for v, df in bars.items():
        daily_cols[f"{v}_rv"] = daily_realized_variance(df, sampling_minutes=minutes)
        daily_cols[f"{v}_gk"] = garman_klass_vol(df)
        daily_cols[f"{v}_ctc"] = close_to_close_vol(df)
        daily_cols[f"{v}_amihud"] = amihud_illiquidity(df, sampling_minutes=minutes)
    daily = pd.DataFrame(daily_cols)
    daily.to_csv(out_dir / "daily_metrics.csv")

    # --- rolling vol + pairwise corr (IF-IH, IC-IM if present) ---
    roll_parts = []
    for v, df in bars.items():
        roll_parts.append(rolling_volatility(df, windows=windows).add_prefix(f"{v}_"))
    pair_specs = [("IF", "IH"), ("IC", "IM")]
    corr_parts = []
    for a, b in pair_specs:
        if a in bars and b in bars:
            c = rolling_correlation(bars[a], bars[b], windows=windows)
            corr_parts.append(c.add_prefix(f"{a}{b}_"))
    rolling = pd.concat(roll_parts + corr_parts, axis=1)
    rolling.to_csv(out_dir / "rolling_stats.csv")

    # --- intraday profiles ---
    rv_profiles = {}
    vol_profiles = {}
    for v, df in bars.items():
        rv_profiles[v] = intraday_rv_profile(df, sampling_minutes=minutes)
        vol_profiles[v] = intraday_volume_oi_profile(df)
        rv_profiles[v].to_csv(out_dir / f"{v}_intraday_rv_profile.csv")
        vol_profiles[v].to_csv(out_dir / f"{v}_intraday_volume_oi_profile.csv")

    # --- figures ---
    fig, ax = plt.subplots(figsize=(10, 4))
    for v, prof in rv_profiles.items():
        ax.plot(range(len(prof)), prof["mean_r2"].values, label=v)
    ref = next(iter(rv_profiles.values()))
    step = max(len(ref) // 8, 1)
    ax.set_xticks(range(0, len(ref), step))
    ax.set_xticklabels(ref.index[::step], rotation=45, ha="right")
    ax.set_title(f"Intraday RV pattern ({minutes}-min squared returns)")
    ax.set_ylabel("Mean r²")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "intraday_rv_pattern.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    for v in varieties:
        col = f"{v}_vol_{windows[0]}d"
        if col in rolling:
            ax.plot(rolling.index, rolling[col], label=f"{v} {windows[0]}d")
    ax.set_title("Rolling volatility (close-to-close)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "rolling_volatility.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    plotted = False
    for a, b in pair_specs:
        col = f"{a}{b}_corr_{windows[0]}d"
        if col in rolling:
            ax.plot(rolling.index, rolling[col], label=col)
            plotted = True
    if plotted:
        ax.set_title("Rolling correlation")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / "rolling_correlation.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    for v in varieties:
        col = f"{v}_amihud"
        if col in daily:
            ax.plot(daily.index, daily[col], label=v)
    ax.set_title("Daily Amihud illiquidity (|r|/volume)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "amihud_illiquidity.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for v, prof in vol_profiles.items():
        axes[0].plot(range(len(prof)), prof["volume_mean"].values, label=v)
        axes[1].plot(range(len(prof)), prof["oi_mean"].values, label=v)
    refv = next(iter(vol_profiles.values()))
    step = max(len(refv) // 8, 1)
    axes[1].set_xticks(range(0, len(refv), step))
    axes[1].set_xticklabels(refv.index[::step], rotation=45, ha="right")
    axes[0].set_title("Intraday volume profile")
    axes[0].legend()
    axes[1].set_title("Intraday open interest profile")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "volume_oi_profile.png", dpi=150)
    plt.close(fig)

    # --- summary findings ---
    lines = [
        f"Project 1 findings (sample={use_sample})",
        f"Varieties: {', '.join(varieties)} | RV sampling: {minutes}min",
        "",
    ]
    for v, prof in rv_profiles.items():
        open_slice = prof.head(6)["mean_r2"].mean()
        day_mean = prof["mean_r2"].mean()
        ratio = open_slice / day_mean if day_mean > 0 else float("nan")
        amihud = daily[f"{v}_amihud"]
        lines.append(
            f"[{v}] early-session r² / day-mean ≈ {ratio:.2f}x | "
            f"bars={len(bars[v]):,} | "
            f"Amihud median={amihud.median():.3e} p90={amihud.quantile(0.9):.3e}"
        )
    if "IF" in bars and "IH" in bars:
        c = rolling_correlation(bars["IF"], bars["IH"], windows=windows)
        col = f"corr_{windows[0]}d"
        lines.append(
            f"[IF-IH] {windows[0]}d corr: mean={c[col].mean():.3f} "
            f"min={c[col].min():.3f} max={c[col].max():.3f}"
        )
    if "IC" in bars and "IM" in bars:
        c = rolling_correlation(bars["IC"], bars["IM"], windows=windows)
        col = f"corr_{windows[0]}d"
        lines.append(
            f"[IC-IM] {windows[0]}d corr: mean={c[col].mean():.3f} "
            f"min={c[col].min():.3f} max={c[col].max():.3f}"
        )
    lines += [
        "",
        f"Outputs: {out_dir}",
        f"Figures: {fig_dir}",
        "Note: main-continuous prices are unsmoothed at rolls (OK for RV/liquidity).",
    ]
    summary = "\n".join(lines) + "\n"
    (out_dir / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Project 1 volatility & liquidity")
    parser.add_argument("--use-sample", action="store_true", help="Use synthetic sample")
    parser.add_argument(
        "--real-data",
        action="store_true",
        default=True,
        help="Use data/processed main-continuous bars (default)",
    )
    args = parser.parse_args()
    use_sample = bool(args.use_sample) and not args.real_data
    # if user only passes --use-sample
    if args.use_sample:
        use_sample = True
    run(use_sample=use_sample)


if __name__ == "__main__":
    main()
