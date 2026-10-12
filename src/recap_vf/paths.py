"""Filesystem layout. Override the roots with RECAP_DATA_ROOT / RECAP_RUNS_ROOT; pick the dataset with RECAP_DATASET."""

import os
from pathlib import Path

from recap_vf.constants import DATASET_NAME, DEFAULT_DATASET

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = Path(os.environ.get("RECAP_DATA_ROOT", REPO_ROOT / "data"))
RUNS_ROOT = Path(os.environ.get("RECAP_RUNS_ROOT", REPO_ROOT / "runs"))


def cache_dir_for(dataset_name: str) -> Path:
    """`data/cache` for the default dataset (the original layout), `data/cache/<name>` for any other."""
    return DATA_ROOT / "cache" if dataset_name == DEFAULT_DATASET else DATA_ROOT / "cache" / dataset_name


RAW_DIR = DATA_ROOT / "raw" / DATASET_NAME
CACHE_DIR = cache_dir_for(DATASET_NAME)
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
