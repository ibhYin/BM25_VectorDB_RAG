"""Compatibility entry point for scripts/kg/export.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.kg.export import *  # noqa: F403


if __name__ == "__main__":
    main()  # noqa: F405
