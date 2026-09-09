from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .basis import rolling_zscore


@dataclass
class BacktestResult:
    equity: pd.Series
    positions: pd.Series
    trades: pd.DataFrame
    metrics: dict[str, float]


def backtest_zscore(
    series: pd.Series,
    *,
    z_window: int = 20,
    entry: float = 2.0,
    exit: float = 0.5,
    cost_bps_one_way: float = 2.5,
) -> BacktestResult:
    """
    Mean-reversion on a stationary spread/basis.

    Position convention (units of series change):
      +1 = long spread (profit when series rises)
      -1 = short spread
    Signal: enter short when z > entry; enter long when z < -entry;
            flatten when |z| < exit.
    PnL ≈ position.shift(1) * diff(series) - costs on position changes.
    Cost model: each unit position change pays cost_bps of notional=1
                (series treated as return-like / relative basis).
    """
    s = series.dropna().astype(float)
    z = rolling_zscore(s, z_window)
    pos = pd.Series(0.0, index=s.index, name="position")
    current = 0.0
    for t, zt in z.items():
        if np.isnan(zt):
            pos.loc[t] = current
            continue
        if current == 0.0:
            if zt > entry:
                current = -1.0
            elif zt < -entry:
                current = 1.0
        else:
            if abs(zt) < exit:
                current = 0.0
        pos.loc[t] = current

    ret = s.diff().fillna(0.0)
    gross = pos.shift(1).fillna(0.0) * ret
    turnover = pos.diff().abs().fillna(abs(pos.iloc[0]) if len(pos) else 0.0)
    costs = turnover * (cost_bps_one_way / 10_000.0)
    net = gross - costs
    equity = net.cumsum()
    equity.name = "equity"

    # trade log: each non-zero holding segment
    trades = _trade_log(pos, net)
    metrics = performance_summary(net, pos)
    return BacktestResult(equity=equity, positions=pos, trades=trades, metrics=metrics)


def _trade_log(pos: pd.Series, net: pd.Series) -> pd.DataFrame:
    rows = []
    if pos.empty:
        return pd.DataFrame()
    prev = 0.0
    entry_i = None
    for i, p in pos.items():
        if prev == 0.0 and p != 0.0:
            entry_i = i
        if prev != 0.0 and p != prev:
            # close or flip
            exit_i = i
            if entry_i is not None:
                seg = net.loc[entry_i:exit_i]
                rows.append(
                    {
                        "entry": entry_i,
                        "exit": exit_i,
                        "side": "long" if prev > 0 else "short",
                        "pnl": float(seg.sum()),
                        "bars": int(seg.shape[0]),
                    }
                )
            entry_i = i if p != 0.0 else None
        prev = p
    if prev != 0.0 and entry_i is not None:
        seg = net.loc[entry_i:]
        rows.append(
            {
                "entry": entry_i,
                "exit": pos.index[-1],
                "side": "long" if prev > 0 else "short",
                "pnl": float(seg.sum()),
                "bars": int(seg.shape[0]),
            }
        )
    return pd.DataFrame(rows)


def performance_summary(net_pnl: pd.Series, positions: pd.Series) -> dict[str, float]:
    x = net_pnl.dropna()
    if x.empty:
        return {}
    # Treat daily pnl as return units on notional=1
    ann_factor = 252.0
    mu = x.mean()
    sd = x.std(ddof=0)
    sharpe = float(mu / sd * np.sqrt(ann_factor)) if sd > 0 else float("nan")
    equity = x.cumsum()
    peak = equity.cummax()
    dd = equity - peak
    max_dd = float(dd.min()) if len(dd) else float("nan")
    wins = x[x > 0]
    n_days_in = int((positions.fillna(0.0) != 0).sum())
    return {
        "total_pnl": float(x.sum()),
        "ann_return": float(mu * ann_factor),
        "ann_vol": float(sd * np.sqrt(ann_factor)),
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "hit_rate": float((x > 0).mean()),
        "avg_daily_pnl": float(mu),
        "days_in_market": float(n_days_in),
        "n_obs": float(len(x)),
    }
