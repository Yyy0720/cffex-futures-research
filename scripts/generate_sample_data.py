"""Generate synthetic IF/IH/spot minute bars for local development."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.sample import write_sample_dataset


if __name__ == "__main__":
    paths = write_sample_dataset(n_days=252)
    for k, v in paths.items():
        print(f"{k}: {v}")
