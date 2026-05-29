"""Pytest configuration: ensure the project root is importable.

Without this, ``from experiments...`` and ``from models...`` can fail when
pytest is invoked from arbitrary working directories.
"""

from pathlib import Path
import sys

# Repository root (parent of ``tests/``).
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
