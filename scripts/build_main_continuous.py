"""Build processed main-continuous 1-min bars from teacher raw data."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.main_continuous import write_processed_main_continuous


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2024-01-01", help="TradingDay start (inclusive)")
    p.add_argument("--end", default=None, help="TradingDay end (inclusive)")
    args = p.parse_args()
    write_processed_main_continuous(start=args.start, end=args.end)


if __name__ == "__main__":
    main()
