# Futures Research

Internship make-up research repo aligned to three resume bullets:

1. **Volatility & liquidity** — realized vol, rolling stats, Amihud / volume-OI profiles (IF / IH)
2. **Statistical arbitrage** — basis mean-reversion or IF–IH pairs (scaffold next)
3. **Risk / margin stress** — exposure + historical/parametric stress (scaffold next)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Desk market data is **not committed** (size + proprietary). Keep it local:

| Path | Purpose |
|------|---------|
| `data/raw/` | Teacher/desk files (minute, spot, tick, main map) |
| `data/processed/` | Main-continuous / spot daily built locally |
| `data/sample/` | Optional synthetic bars |

```bash
# after placing files in data/raw/
python scripts/build_main_continuous.py --start 2024-01-01
python scripts/process_spot.py
# optional smoke test without desk data:
python scripts/generate_sample_data.py
```

## Project 1

```bash
python projects/01_volatility_liquidity/run_analysis.py --use-sample
# after placing real files in data/raw or data/processed:
python projects/01_volatility_liquidity/run_analysis.py --real-data
```

Outputs land in `results/01_volatility_liquidity/` (CSV + figures + `summary.txt`).

## Project 2

Cross-product pairs (IF–IH, IC–IM) until spot arrives:

```bash
python projects/02_stat_arb/run_analysis.py --real-data
```

Outputs → `results/02_stat_arb/`.

## Project 3

Exposure + parametric/historical margin stress (illustrative margin rates):

```bash
python projects/03_risk_margin/run_analysis.py --real-data
```

Outputs → `results/03_risk_margin/`.

## Layout

```
configs/          shared YAML
src/data/         load / clean / sample generator
src/metrics/      RV, rolling, liquidity
projects/01_…     analysis entrypoints
projects/02_…     stat arb (next)
projects/03_…     risk / margin (next)
results/          artifacts per project
```
