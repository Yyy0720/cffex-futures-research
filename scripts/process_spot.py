"""Normalize teacher spot index files into data/processed."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.spot import write_processed_spot

if __name__ == "__main__":
    write_processed_spot()
