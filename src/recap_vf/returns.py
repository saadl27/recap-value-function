"""Monte-Carlo rewards, returns-to-go and 201-bin value targets (RECAP, step 2).

Per episode of length T with success flag s (gamma = 1):
    r_t = -1 for t < T-1;  r_{T-1} = 0 if success else -C_fail
    G_t = -(T-1-t) - C_fail * 1[failure]
One dataset-wide constant Z = |min G_t| maps returns into [-1, 0]; the bin target comes from `recap_vf.bins`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from recap_vf import bins, paths

logger = logging.getLogger(__name__)

EPISODE_COLUMNS = ["episode_index", "length", "success", "dataset_from_index", "dataset_to_index"]
RETURNS_COLUMNS = [
    "index",
    "episode_index",
    "frame_index",
    "success",
    "episode_length",
    "reward",
    "return_raw",
    "return_norm",
    "bin_index",
]


def returns_tag(c_fail: float) -> str:
    """Tag used in file names, e.g. 2229 -> 'c2229'."""
    return f"c{int(c_fail)}"


def compute_episode_returns(length: int, success: bool, c_fail: float) -> dict[str, np.ndarray]:
    """Per-frame reward and undiscounted return-to-go for one episode (float64)."""
    if length < 1:
        raise ValueError(f"episode length must be >= 1, got {length}")
    reward = np.full(length, -1.0, dtype=np.float64)
    reward[-1] = 0.0 if success else -float(c_fail)
    # Reverse cumulative sum = sum_{t' >= t} r_{t'}; exact for these integer-valued rewards.
    return_raw = np.cumsum(reward[::-1])[::-1].copy()
    return {"reward": reward, "return_raw": return_raw}


def load_episodes(raw_dir: Path | None = None, prefer_cache: bool = True) -> pd.DataFrame:
    """Episode table (`episode_index, length, success, dataset_from_index, dataset_to_index`).

    Uses step 1's `data/cache/episodes.parquet` if it exists, else the raw LeRobot `meta/episodes` parquet(s).
    """
    if prefer_cache and paths.EPISODE_TABLE.exists():
        logger.info("reading episode table from %s", paths.EPISODE_TABLE)
        df = pd.read_parquet(paths.EPISODE_TABLE, columns=EPISODE_COLUMNS)
    else:
        raw_dir = Path(raw_dir) if raw_dir is not None else paths.RAW_DIR
        files = sorted((raw_dir / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
        if not files:
            raise FileNotFoundError(f"no meta/episodes parquet under {raw_dir}")
        logger.info("reading episode metadata from %d raw file(s) under %s", len(files), raw_dir)
        df = pd.concat([pd.read_parquet(f, columns=EPISODE_COLUMNS) for f in files], ignore_index=True)
    df = df.sort_values("episode_index").reset_index(drop=True)
    df["success"] = df["success"].astype(bool)
    for col in ("episode_index", "length", "dataset_from_index", "dataset_to_index"):
        df[col] = df[col].astype(np.int64)
    if not (df["dataset_to_index"] - df["dataset_from_index"] == df["length"]).all():
        raise ValueError("dataset_to_index - dataset_from_index != length for some episodes")
    return df


def compute_returns_table(episodes: pd.DataFrame, c_fail: float) -> tuple[pd.DataFrame, dict]:
    """Full per-frame returns table (contract schema) and its metadata dict."""
    episodes = episodes.sort_values("episode_index").reset_index(drop=True)
    lengths = episodes["length"].to_numpy(np.int64)
    successes = episodes["success"].to_numpy(bool)
    if (lengths < 1).any():
        raise ValueError("all episodes must have length >= 1")
    if successes.all():
        logger.warning("no failure episodes: C_fail has no effect")

    rewards, returns = [], []
    for length, success in zip(lengths, successes, strict=True):
        ep = compute_episode_returns(int(length), bool(success), c_fail)
        rewards.append(ep["reward"])
        returns.append(ep["return_raw"])
    reward = np.concatenate(rewards)
    return_raw = np.concatenate(returns)

    z = float(-return_raw.min())
    if z <= 0:
        raise ValueError(f"normaliser Z must be > 0, got {z} (all returns are 0?)")
    return_norm = return_raw / z

    from_index = episodes["dataset_from_index"].to_numpy(np.int64)
    frame_index = np.concatenate([np.arange(n, dtype=np.int64) for n in lengths])
    table = pd.DataFrame(
        {
            "index": np.repeat(from_index, lengths) + frame_index,
            "episode_index": np.repeat(episodes["episode_index"].to_numpy(np.int64), lengths),
            "frame_index": frame_index,
            "success": np.repeat(successes, lengths),
            "episode_length": np.repeat(lengths, lengths),
            "reward": reward.astype(np.float32),
            "return_raw": return_raw.astype(np.float32),
            "return_norm": return_norm.astype(np.float32),
            "bin_index": bins.value_to_bin_np(return_norm),
        }
    )
    table = table.sort_values("index", kind="stable").reset_index(drop=True)

    meta = {
        "tag": returns_tag(c_fail),
        "c_fail": float(c_fail),
        "Z": z,
        "num_bins": bins.NUM_BINS,
        "v_min": bins.V_MIN,
        "v_max": bins.V_MAX,
        "t_max": int(lengths.max()),
        "t_max_success": int(lengths[successes].max()) if successes.any() else 0,
        "t_max_failure": int(lengths[~successes].max()) if (~successes).any() else 0,
    }
    return table, meta


def verify_index(table: pd.DataFrame, frame_table: pd.DataFrame) -> None:
    """Check that (index, episode_index, frame_index) match the dataset's own frame table exactly."""
    cols = ["index", "episode_index", "frame_index"]
    ref = frame_table[cols].sort_values("index").reset_index(drop=True)
    ours = table[cols].sort_values("index").reset_index(drop=True)
    if len(ref) != len(ours):
        raise ValueError(f"row count mismatch: returns {len(ours)} vs frame table {len(ref)}")
    for col in cols:
        if not np.array_equal(ref[col].to_numpy(np.int64), ours[col].to_numpy(np.int64)):
            raise ValueError(f"column {col!r} differs between returns table and frame table")


def save_returns(table: pd.DataFrame, meta: dict, tag: str | None = None) -> tuple[Path, Path]:
    """Write `returns_{tag}.parquet` + `.json` to the cache directory."""
    tag = tag or meta["tag"]
    pq_path, json_path = paths.returns_paths(tag)
    pq_path.parent.mkdir(parents=True, exist_ok=True)
    table[RETURNS_COLUMNS].to_parquet(pq_path, index=False)
    json_path.write_text(json.dumps(meta, indent=2) + "\n")
    return pq_path, json_path


def load_returns(tag: str = "c2229") -> tuple[pd.DataFrame, dict]:
    """Load a returns table and its metadata (used by steps 4 and 6)."""
    pq_path, json_path = paths.returns_paths(tag)
    return pd.read_parquet(pq_path), json.loads(json_path.read_text())


def summarize_returns(table: pd.DataFrame, meta: dict) -> dict:
    """Key numbers for the report: frame balance, return_norm ranges and bin ranges per class."""
    out: dict = {"tag": meta["tag"], "c_fail": meta["c_fail"], "Z": meta["Z"], "num_frames": len(table)}
    for name, mask in (("success", table["success"]), ("failure", ~table["success"])):
        sub = table[mask]
        out[name] = {
            "episodes": int(sub["episode_index"].nunique()),
            "frames": len(sub),
            "frame_fraction": len(sub) / len(table),
            "return_norm_min": float(sub["return_norm"].min()) if len(sub) else float("nan"),
            "return_norm_max": float(sub["return_norm"].max()) if len(sub) else float("nan"),
            "bin_min": int(sub["bin_index"].min()) if len(sub) else -1,
            "bin_max": int(sub["bin_index"].max()) if len(sub) else -1,
            "bins_used": int(sub["bin_index"].nunique()),
        }
    counts = np.bincount(table["bin_index"].to_numpy(), minlength=bins.NUM_BINS)
    nonzero = counts[counts > 0]
    out["bins"] = {
        "used": int((counts > 0).sum()),
        "empty": int((counts == 0).sum()),
        "count_min_nonzero": int(nonzero.min()),
        "count_median_nonzero": float(np.median(nonzero)),
        "count_max": int(counts.max()),
        "argmax": int(counts.argmax()),
    }
    return out


def bin_counts(table: pd.DataFrame) -> pd.DataFrame:
    """Frames per bin, split by class (columns: bin_index, success, failure, total)."""
    b = table["bin_index"].to_numpy()
    s = table["success"].to_numpy(bool)
    succ = np.bincount(b[s], minlength=bins.NUM_BINS)
    fail = np.bincount(b[~s], minlength=bins.NUM_BINS)
    return pd.DataFrame({"bin_index": np.arange(bins.NUM_BINS), "success": succ, "failure": fail, "total": succ + fail})
