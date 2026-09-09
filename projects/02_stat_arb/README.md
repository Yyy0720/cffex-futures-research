# Project 2 — Statistical Arbitrage

Default with spot data: **basis mean-reversion**

- IF–CSI300, IH–SSE50, IC–CSI500, IM–CSI1000
- Relative basis `F/S - 1`, ADF / Engle–Granger, cost-aware z-score, train/test

```bash
python scripts/process_spot.py
python projects/02_stat_arb/run_analysis.py --real-data
```

Cross-product pairs remain available via `project_02.mode: pairs` in `configs/default.yaml`.
