"""
Project 3: Portfolio exposure & margin stress testing.

Uses main-continuous daily closes from processed minute bars.
Margin rates are illustrative assumptions in configs/default.yaml
(not historical exchange adjustment records).

Usage:
  python projects/03_risk_margin/run_analysis.py --real-data
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.config import load_config, project_root
from src.data.loaders import daily_close, load_symbol_bars
from src.risk.exposure import (
    build_book,
    daily_portfolio_pnl,
    exposure_table,
    historical_scenario_table,
    historical_var,
    parametric_shock_table,
    portfolio_summary,
)


def run(use_sample: bool = False) -> Path:
    cfg = load_config()
    p3 = cfg["project_03"]
    markets = cfg["markets"]
    lots = {k: float(v) for k, v in p3["positions"].items()}
    start = p3.get("analysis_start")

    out_dir = project_root() / p3["results_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    closes = {}
    for v in lots:
        s = daily_close(load_symbol_bars(v, use_sample=use_sample, cfg=cfg))
        if start and not use_sample:
            s = s[s.index >= pd.Timestamp(start)]
        closes[v] = s
    px = pd.concat(closes, axis=1).dropna(how="any")
    px.columns = list(closes.keys())

    latest = {v: float(px[v].iloc[-1]) for v in px.columns}
    book = build_book(latest, lots, markets)
    exp = exposure_table(book)
    summary = portfolio_summary(book)
    exp.to_csv(out_dir / "exposure.csv", index=False)

    shocks = [float(x) for x in p3["shock_pct"]]
    para = parametric_shock_table(book, shocks)
    para.to_csv(out_dir / "parametric_stress.csv", index=False)

    hist = historical_scenario_table(
        px, lots, markets, n_worst=int(p3.get("n_worst_days", 5))
    )
    hist.to_csv(out_dir / "historical_scenarios.csv", index=False)

    pnl = daily_portfolio_pnl(px, lots, markets)
    pnl.to_csv(out_dir / "daily_portfolio_pnl.csv", header=True)
    var_levels = [float(c) for c in p3["var_confidence"]]
    var_rows = [{"confidence": c, "historical_var": historical_var(pnl, c)} for c in var_levels]
    var_df = pd.DataFrame(var_rows)
    var_df.to_csv(out_dir / "var.csv", index=False)

    # figures
    fig, ax = plt.subplots(figsize=(10, 4))
    equity = pnl.cumsum()
    ax.plot(equity.index, equity.values)
    ax.set_title("Cumulative portfolio MTM PnL (illustrative book)")
    fig.tight_layout()
    fig.savefig(fig_dir / "cumulative_pnl.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    labels = [f"{int(r.shock_pct*100):+d}%" for _, r in para.iterrows()]
    ax.bar(range(len(para)), para["portfolio_pnl"].values)
    ax.set_xticks(range(len(para)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title("Parametric parallel price shocks — portfolio PnL")
    ax.axhline(0, color="k", lw=0.8)
    fig.tight_layout()
    fig.savefig(fig_dir / "parametric_pnl.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(para)), para["margin_delta"].values, color="C1")
    ax.set_xticks(range(len(para)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title("Change in initial margin under shocks")
    ax.axhline(0, color="k", lw=0.8)
    fig.tight_layout()
    fig.savefig(fig_dir / "margin_delta.png", dpi=150)
    plt.close(fig)

    # simple alert rules (methodology note)
    worst_replay = float(hist["replay_on_latest_book_pnl"].min()) if len(hist) else float("nan")
    var99 = float(var_df.loc[var_df["confidence"] == 0.99, "historical_var"].iloc[0]) if (var_df["confidence"] == 0.99).any() else float("nan")
    alert_lines = [
        "Monitoring logic (illustrative):",
        f"- Alert if 1-day historical VaR(99%) > 30% of total initial margin "
        f"(VaR99={var99:,.0f}, margin={summary['total_initial_margin']:,.0f}, "
        f"ratio={var99 / summary['total_initial_margin'] if summary['total_initial_margin'] else float('nan'):.2%}).",
        f"- Alert if replay of worst historical day on current book < -25% of margin "
        f"(worst_replay={worst_replay:,.0f}).",
        "- Alert if |net|/gross notional exceeds desk limit (current "
        f"{summary['net_gross_ratio']:.2%}); limit itself should be set with mentor.",
        "- Margin rates are CONFIG ASSUMPTIONS, not exchange historical adjustments.",
    ]

    lines = [
        f"Project 3 — exposure & margin stress (sample={use_sample})",
        f"As-of prices: { {k: round(v,2) for k,v in latest.items()} }",
        f"Positions (lots): {lots}",
        "",
        "Exposure summary:",
        *[f"  {k}: {v:,.2f}" if isinstance(v, float) else f"  {k}: {v}" for k, v in summary.items()],
        "",
        "Historical VaR (daily PnL):",
        *[f"  {r['confidence']:.0%}: {r['historical_var']:,.2f}" for _, r in var_df.iterrows()],
        "",
        "Worst historical days (realized book PnL):",
    ]
    for _, r in hist.iterrows():
        lines.append(
            f"  {r['scenario_date']}: realized={r['realized_pnl_that_day']:,.0f} | "
            f"replay_now={r['replay_on_latest_book_pnl']:,.0f}"
        )
    lines += ["", *alert_lines, "", f"Outputs: {out_dir}"]
    text = "\n".join(lines) + "\n"
    (out_dir / "summary.txt").write_text(text, encoding="utf-8")
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {"latest_prices": latest, "positions": lots, "exposure": summary, "var": var_rows},
            f,
            indent=2,
        )
    print(text)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Project 3 risk / margin stress")
    parser.add_argument("--use-sample", action="store_true")
    parser.add_argument("--real-data", action="store_true", default=True)
    args = parser.parse_args()
    run(use_sample=bool(args.use_sample))


if __name__ == "__main__":
    main()
