"""Portfolio exposure and margin stress helpers."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BookPosition:
    variety: str
    lots: float  # signed
    multiplier: float
    margin_rate: float
    price: float

    @property
    def notional(self) -> float:
        return self.lots * self.multiplier * self.price

    @property
    def margin(self) -> float:
        return abs(self.notional) * self.margin_rate


def build_book(
    prices: dict[str, float],
    lots: dict[str, float],
    markets: dict,
) -> list[BookPosition]:
    book = []
    for v, n in lots.items():
        mkt = markets[v]
        book.append(
            BookPosition(
                variety=v,
                lots=float(n),
                multiplier=float(mkt["multiplier"]),
                margin_rate=float(mkt["margin_rate"]),
                price=float(prices[v]),
            )
        )
    return book


def exposure_table(book: list[BookPosition]) -> pd.DataFrame:
    rows = []
    for p in book:
        rows.append(
            {
                "variety": p.variety,
                "lots": p.lots,
                "side": "long" if p.lots > 0 else ("short" if p.lots < 0 else "flat"),
                "multiplier": p.multiplier,
                "price": p.price,
                "notional": p.notional,
                "abs_notional": abs(p.notional),
                "margin_rate": p.margin_rate,
                "initial_margin": p.margin,
            }
        )
    df = pd.DataFrame(rows)
    return df


def portfolio_summary(book: list[BookPosition]) -> dict[str, float]:
    notionals = np.array([p.notional for p in book], dtype=float)
    margins = np.array([p.margin for p in book], dtype=float)
    return {
        "gross_notional": float(np.abs(notionals).sum()),
        "net_notional": float(notionals.sum()),
        "long_notional": float(notionals[notionals > 0].sum()) if (notionals > 0).any() else 0.0,
        "short_notional": float(notionals[notionals < 0].sum()) if (notionals < 0).any() else 0.0,
        "total_initial_margin": float(margins.sum()),
        "net_gross_ratio": float(
            abs(notionals.sum()) / np.abs(notionals).sum()
            if np.abs(notionals).sum() > 0
            else np.nan
        ),
    }


def daily_portfolio_pnl(
    closes: pd.DataFrame,
    lots: dict[str, float],
    markets: dict,
) -> pd.Series:
    """
    Approximate daily marked-to-market PnL in currency units:
      sum_i lots_i * multiplier_i * delta_price_i
    """
    pnl = pd.Series(0.0, index=closes.index, dtype=float)
    for v, n in lots.items():
        if v not in closes.columns:
            continue
        mult = float(markets[v]["multiplier"])
        pnl = pnl.add(float(n) * mult * closes[v].diff().fillna(0.0), fill_value=0.0)
    pnl.name = "portfolio_pnl"
    return pnl


def historical_var(pnl: pd.Series, conf: float = 0.95) -> float:
    """Historical simulation VaR on daily PnL (positive number = loss quantile)."""
    x = pnl.dropna()
    if x.empty:
        return float("nan")
    q = np.quantile(x, 1.0 - conf)
    return float(-q)  # loss amount


def parametric_shock_table(
    book: list[BookPosition],
    shocks: list[float],
) -> pd.DataFrame:
    """
    Parallel price shocks applied to all names (same signed % move).
    PnL ≈ sum lots * mult * price * shock
    Margin after ≈ abs(notional*(1+shock)) * rate  (simplified)
    """
    rows = []
    base_margin = sum(p.margin for p in book)
    for s in shocks:
        for sign in (+1.0, -1.0):
            shock = sign * abs(s)
            pnl = sum(p.lots * p.multiplier * p.price * shock for p in book)
            margin_after = sum(
                abs(p.lots * p.multiplier * p.price * (1.0 + shock)) * p.margin_rate
                for p in book
            )
            rows.append(
                {
                    "shock_pct": shock,
                    "portfolio_pnl": pnl,
                    "margin_before": base_margin,
                    "margin_after": margin_after,
                    "margin_delta": margin_after - base_margin,
                    "equity_vs_margin_buffer": pnl - (margin_after - base_margin),
                }
            )
    return pd.DataFrame(rows)


def historical_scenario_table(
    closes: pd.DataFrame,
    lots: dict[str, float],
    markets: dict,
    n_worst: int = 5,
) -> pd.DataFrame:
    pnl = daily_portfolio_pnl(closes, lots, markets)
    worst = pnl.nsmallest(n_worst)
    rows = []
    for dt, loss in worst.items():
        day = closes.loc[dt]
        move = {}
        prev_idx = closes.index.get_loc(dt)
        if prev_idx == 0:
            rets = {c: np.nan for c in closes.columns}
        else:
            prev = closes.iloc[prev_idx - 1]
            rets = {c: float(day[c] / prev[c] - 1.0) for c in closes.columns}
        # replay this day's % moves from latest prices as a stress scenario
        last = closes.iloc[-1]
        replay_pnl = 0.0
        for v, n in lots.items():
            if v not in last.index or np.isnan(rets.get(v, np.nan)):
                continue
            replay_pnl += float(n) * float(markets[v]["multiplier"]) * float(last[v]) * rets[v]
        rows.append(
            {
                "scenario_date": pd.Timestamp(dt).date(),
                "realized_pnl_that_day": float(loss),
                "replay_on_latest_book_pnl": float(replay_pnl),
                **{f"ret_{k}": v for k, v in rets.items()},
            }
        )
    return pd.DataFrame(rows)
