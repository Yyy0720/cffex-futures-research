# Project 3 — Portfolio Exposure & Margin Stress

Illustrative IF/IH/IC/IM book using main-continuous daily closes.

- Notional / initial margin (rates are **config assumptions**)
- Parametric parallel shocks (±5/10/15%)
- Historical worst-day replay + historical VaR

```bash
python projects/03_risk_margin/run_analysis.py --real-data
```

Does **not** reproduce official margin-ratio adjustment history (data not provided).
