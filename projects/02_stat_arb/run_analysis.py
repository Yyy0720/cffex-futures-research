"""
Project 2: Statistical arbitrage with currently available data.

Default (no spot): cross-product pairs IF–IH and IC–IM
  - Engle–Granger hedge on log prices
  - ADF on residual spread + half-life
  - Cost-aware z-score backtest with train/test split
  - Rolling ADF / half-life regime diagnostics

If spot indices appear later, set project_02.mode: basis in configs/default.yaml.

Usage:
  python projects/02_stat_arb/run_analysis.py --real-data
  python projects/02_stat_arb/run_analysis.py --use-sample
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.config import load_config, project_root
from src.data.loaders import load_daily_close
from src.statarb.backtest import backtest_zscore
from src.statarb.basis import compute_basis, rolling_zscore
from src.statarb.cointegration import adf_stationarity, engle_granger, half_life
from src.statarb.regime import rolling_adf_pvalue, rolling_half_life


def _split_train_test(s: pd.Series, train_ratio: float) -> tuple[pd.Series, pd.Series]:
    n = len(s)
    cut = max(int(n * train_ratio), 1)
    cut = min(cut, n - 1) if n > 1 else n
    return s.iloc[:cut], s.iloc[cut:]


def _analyze_spread(
    name: str,
    spread: pd.Series,
    y: pd.Series,
    x: pd.Series,
    cfg_p2: dict,
    out_dir: Path,
    fig_dir: Path,
    eg_info: dict | None = None,
) -> dict:
    spread = spread.dropna()
    spread.name = "spread"
    spread.to_csv(out_dir / f"{name}_spread.csv", header=True)

    adf_s = adf_stationarity(spread)
    eg = engle_granger(y.reindex(spread.index), x.reindex(spread.index))
    hl = half_life(spread)

    train, test = _split_train_test(spread, float(cfg_p2["train_ratio"]))
    adf_train = adf_stationarity(train)
    adf_test = adf_stationarity(test) if len(test) >= 15 else None

    bt_kwargs = dict(
        z_window=int(cfg_p2["zscore_window"]),
        entry=float(cfg_p2["zscore_entry"]),
        exit=float(cfg_p2["zscore_exit"]),
        cost_bps_one_way=float(cfg_p2["cost_bps_one_way"]),
    )
    bt_all = backtest_zscore(spread, **bt_kwargs)
    bt_train = backtest_zscore(train, **bt_kwargs)
    bt_test = (
        backtest_zscore(test, **bt_kwargs)
        if len(test) >= bt_kwargs["z_window"] + 5
        else None
    )

    bt_all.equity.to_csv(out_dir / f"{name}_equity.csv", header=True)
    bt_all.trades.to_csv(out_dir / f"{name}_trades.csv", index=False)

    z = rolling_zscore(spread, int(cfg_p2["zscore_window"]))
    regime_w = int(cfg_p2["regime_window"])
    radf = rolling_adf_pvalue(spread, window=regime_w)
    rhl = rolling_half_life(spread, window=regime_w)
    pd.DataFrame({"zscore": z, "adf_pvalue": radf, "half_life": rhl}).to_csv(
        out_dir / f"{name}_regime.csv"
    )

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(spread.index, spread.values, label="spread")
    axes[0].set_title(f"{name} residual spread")
    axes[0].legend()
    axes[1].plot(z.index, z.values, label="zscore", color="C1")
    axes[1].axhline(cfg_p2["zscore_entry"], color="r", ls="--", lw=0.8)
    axes[1].axhline(-cfg_p2["zscore_entry"], color="r", ls="--", lw=0.8)
    axes[1].axhline(cfg_p2["zscore_exit"], color="gray", ls=":", lw=0.8)
    axes[1].axhline(-cfg_p2["zscore_exit"], color="gray", ls=":", lw=0.8)
    axes[1].set_title("Z-score & thresholds")
    axes[2].plot(bt_all.equity.index, bt_all.equity.values, label="net equity", color="C2")
    cut_date = train.index[-1]
    axes[2].axvline(cut_date, color="k", ls="--", lw=0.8, label="train/test split")
    axes[2].set_title("Cumulative PnL (net of costs)")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(fig_dir / f"{name}_spread_backtest.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    axes[0].plot(radf.index, radf.values, label="rolling ADF p")
    axes[0].axhline(0.05, color="r", ls="--", lw=0.8, label="5%")
    axes[0].set_title(f"{name} rolling ADF p-value ({regime_w}d)")
    axes[0].legend()
    finite_hl = rhl.replace([np.inf, -np.inf], np.nan)
    axes[1].plot(finite_hl.index, finite_hl.values, label="half-life (days)", color="C3")
    axes[1].set_title("Rolling half-life")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(fig_dir / f"{name}_regime.png", dpi=150)
    plt.close(fig)

    return {
        "pair": name,
        "n_obs": int(spread.shape[0]),
        "half_life_days": None if hl == float("inf") else hl,
        "adf_spread": adf_s.to_dict(),
        "engle_granger": eg.to_dict(),
        "eg_extra": eg_info,
        "adf_train": adf_train.to_dict(),
        "adf_test": adf_test.to_dict() if adf_test else None,
        "backtest_all": bt_all.metrics,
        "backtest_train": bt_train.metrics,
        "backtest_test": bt_test.metrics if bt_test else None,
        "params": bt_kwargs,
        "train_end": str(pd.Timestamp(cut_date).date()),
    }


def run(use_sample: bool = False) -> Path:
    cfg = load_config()
    p2 = cfg["project_02"]
    out_dir = project_root() / p2["results_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    mode = p2.get("mode", "pairs")
    start = p2.get("analysis_start")
    reports: list[dict] = []
    lines = [
        f"Project 2 — mode={mode} (sample={use_sample})",
        "Basis mode: relative futures/spot - 1 with ADF / Engle-Granger / cost-aware z-score."
        if mode == "basis"
        else "Pairs mode: cross-product log-price spreads (no spot required).",
        "",
    ]

    if mode == "pairs":
        needed = sorted({s for pair in p2["cross_pairs"] for s in pair})
        closes = {}
        for sym in needed:
            s = load_daily_close(sym, use_sample=use_sample, cfg=cfg)
            if start and not use_sample:
                s = s[s.index >= pd.Timestamp(start)]
            closes[sym] = s

        for y_sym, x_sym in p2["cross_pairs"]:
            name = f"{y_sym}_{x_sym}"
            y = np.log(closes[y_sym])
            x = np.log(closes[x_sym])
            eg = engle_granger(y, x)
            # residual in log space; backtest treats diffs as return-like units
            spread = eg.residual
            report = _analyze_spread(
                name,
                spread,
                y,
                x,
                p2,
                out_dir,
                fig_dir,
                eg_info={"y": y_sym, "x": x_sym, "price_space": "log"},
            )
            reports.append(report)
    else:
        # basis path: futures vs spot daily closes
        symbols = set(p2["basis_pairs"].keys()) | set(p2["basis_pairs"].values())
        closes = {}
        for sym in symbols:
            s = load_daily_close(sym, use_sample=use_sample, cfg=cfg)
            if start and not use_sample:
                s = s[s.index >= pd.Timestamp(start)]
            closes[sym] = s
        for fut_sym, spot_sym in p2["basis_pairs"].items():
            name = f"{fut_sym}_{spot_sym}"
            basis = compute_basis(
                closes[fut_sym], closes[spot_sym], mode=p2["basis_mode"]
            )
            report = _analyze_spread(
                name,
                basis,
                closes[fut_sym],
                closes[spot_sym],
                p2,
                out_dir,
                fig_dir,
                eg_info={"futures": fut_sym, "spot": spot_sym, "basis_mode": p2["basis_mode"]},
            )
            reports.append(report)

    for report in reports:
        m_all = report["backtest_all"]
        m_te = report["backtest_test"] or {}
        lines.append(
            f"[{report['pair']}] ADF p={report['adf_spread']['pvalue']:.4f} "
            f"stat={report['adf_spread']['stationary_5pct']} | "
            f"EG coint p={report['engle_granger']['coint_pvalue']:.4f} | "
            f"beta={report['engle_granger']['beta']:.4f} | "
            f"half-life={report['half_life_days']}"
        )
        lines.append(
            f"  full Sharpe={m_all.get('sharpe', float('nan')):.3f} "
            f"total_pnl={m_all.get('total_pnl', float('nan')):.6f} "
            f"maxDD={m_all.get('max_drawdown', float('nan')):.6f}"
        )
        if m_te:
            lines.append(
                f"  test Sharpe={m_te.get('sharpe', float('nan')):.3f} "
                f"total_pnl={m_te.get('total_pnl', float('nan')):.6f} "
                f"(train ends {report['train_end']})"
            )
        lines.append("")

    lines.append(
        "Honest note: PnL is in residual/log-spread units with a flat bps cost model — "
        "use for methodology demo, not as a claim of live edge."
    )
    summary = "\n".join(lines) + "\n"
    (out_dir / "summary.txt").write_text(summary, encoding="utf-8")
    with (out_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2, default=str)

    rows = []
    for r in reports:
        row = {"pair": r["pair"], "half_life": r["half_life_days"]}
        row.update({f"all_{k}": v for k, v in (r["backtest_all"] or {}).items()})
        if r["backtest_test"]:
            row.update({f"test_{k}": v for k, v in r["backtest_test"].items()})
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "metrics.csv", index=False)

    print(summary)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Project 2 statistical arbitrage")
    parser.add_argument("--use-sample", action="store_true")
    parser.add_argument("--real-data", action="store_true", default=True)
    args = parser.parse_args()
    use_sample = bool(args.use_sample)
    run(use_sample=use_sample)


if __name__ == "__main__":
    main()
