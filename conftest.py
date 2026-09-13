import sys
from pathlib import Path

# Ensure the repo root is importable regardless of how pytest is invoked
# (plain `pytest` vs `python -m pytest`), since dspy_modules/ and
# trace_collection/ are top-level packages, not an installed distribution.
sys.path.insert(0, str(Path(__file__).resolve().parent))
