"""Standalone wrapper for dataset conversion without importing the package root."""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    target = Path(__file__).resolve().parents[1] / "molecular_force_field" / "cli" / "convert_dataset.py"
    runpy.run_path(str(target), run_name="__main__")
