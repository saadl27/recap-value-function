"""Filesystem layout. Override the roots with RECAP_DATA_ROOT / RECAP_RUNS_ROOT."""

import os
from pathlib import Path

from recap_vf.constants import DATASET_NAME

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = Path(os.environ.get("RECAP_DATA_ROOT", REPO_ROOT / "data"))
RUNS_ROOT = Path(os.environ.get("RECAP_RUNS_ROOT", REPO_ROOT / "runs"))

RAW_DIR = DATA_ROOT / "raw" / DATASET_NAME
CACHE_DIR = DATA_ROOT / "cache"
FRAMES_DIR = CACHE_DIR / "frames"
FRAME_TABLE = CACHE_DIR / "frames.parquet"
EPISODE_TABLE = CACHE_DIR / "episodes.parquet"
SPLITS_FILE = CACHE_DIR / "splits.json"

REPORTS_DIR = REPO_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


def episode_frames_path(episode_index: int) -> Path:
    return FRAMES_DIR / f"ep_{episode_index:04d}.h5"


def returns_paths(tag: str) -> tuple[Path, Path]:
    """(parquet, json) for a returns table, e.g. tag='c2229'."""
    return CACHE_DIR / f"returns_{tag}.parquet", CACHE_DIR / f"returns_{tag}.json"


def run_dir(run_name: str) -> Path:
    return RUNS_ROOT / run_name
