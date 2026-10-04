"""Central configuration for the ML package (paths and constants)."""
import os
from pathlib import Path

ROOT = Path(os.getenv("SG_ROOT", Path(__file__).resolve().parents[1]))
ARTIFACTS_DIR = Path(os.getenv("SG_ARTIFACTS_DIR", ROOT / "ml" / "models" / "artifacts"))
DATA_DIR = ROOT / "ml" / "data"
DOCS_DIR = Path(os.getenv("SG_DOCS_DIR", ROOT / "docs"))
MIN_POSTS = 5          # minimum posts needed to score an account
SEED = 42
