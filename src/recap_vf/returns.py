"""Monte-Carlo rewards, returns-to-go and 201-bin value targets (RECAP, step 2).

Per episode of length T with success flag s (gamma = 1):
    r_t = -1 for t < T-1;  r_{T-1} = 0 if success else -C_fail
    G_t = -(T-1-t) - C_fail * 1[failure]
One dataset-wide constant Z = |min G_t| maps returns into [-1, 0]; the bin target comes from `recap_vf.bins`.

Datasets with human interventions (DAgger rollouts, per-frame `intervention` flag) use the variant in
`compute_episode_returns_interventions`: the robot's last frame before each takeover gets -C_interv, every other frame
-1, and the return-to-go is cut at the next takeover, so each penalty only counts for the frames that led to it.
See docs/interventions.md.
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
# Extra columns of returns tables built with interventions (see `compute_episode_returns_interventions`).
INTERVENTION_COLUMNS = ["intervention", "stretch_index", "stretch_frame", "stretch_length"]


def returns_tag(c_fail: float, c_interv: float | None = None) -> str:
    """Tag used in file names, e.g. 2229 -> 'c2229'; with an intervention penalty 1500 -> 'i1500'."""
    return f"c{int(c_fail)}" if c_interv is None else f"i{int(c_interv)}"


def compute_episode_returns(length: int, success: bool, c_fail: float) -> dict[str, np.ndarray]:
    """Per-frame reward and undiscounted return-to-go for one episode (float64)."""
    if length < 1:
        raise ValueError(f"episode length must be >= 1, got {length}")
    reward = np.full(length, -1.0, dtype=np.float64)
    reward[-1] = 0.0 if success else -float(c_fail)
    # Reverse cumulative sum = sum_{t' >= t} r_{t'}; exact for these integer-valued rewards.
    return_raw = np.cumsum(reward[::-1])[::-1].copy()
    return {"reward": reward, "return_raw": return_raw}


def intervention_onsets(intervention: np.ndarray) -> np.ndarray:
    """Frames where a human takes over: `intervention[t]` and not `intervention[t-1]` (t >= 1).

    An episode that starts under human control has no onset at frame 0 (no robot action led to it).
    """
    h = np.asarray(intervention, dtype=bool)
    return np.flatnonzero(h[1:] & ~h[:-1]) + 1


def compute_episode_returns_interventions(
    length: int, success: bool, c_fail: float, intervention: np.ndarray, c_interv: float
) -> dict[str, np.ndarray]:
    """Rewards and returns-to-go of one episode with human interventions, cut at each takeover (gamma = 1).

    Every frame costs -1. The robot's last frame before each takeover gets -C_interv instead, and the last frame of
    the episode gets 0 (success) or -C_fail (failure). The episode splits into *stretches* that start at frame 0 or
    at a takeover: a stretch is the human correction followed by the robot's control until the next takeover (or the
    episode end). The return-to-go of a frame only sums its own stretch:
        G_t = -(end_s - 1 - t) - C_interv    if the stretch ends at a takeover
        G_t = -(T - 1 - t) - C_fail * 1[failure]    for the last stretch
    so a later intervention, caused by the robot after the human handed back, never counts against earlier frames.
    Without interventions this equals `compute_episode_returns`.
    """
    if length < 1:
        raise ValueError(f"episode length must be >= 1, got {length}")
    h = np.asarray(intervention, dtype=bool)
    if h.shape != (length,):
        raise ValueError(f"intervention has shape {h.shape}, expected ({length},)")
    onsets = intervention_onsets(h)
    reward = np.full(length, -1.0, dtype=np.float64)
    reward[onsets - 1] = -float(c_interv)
    reward[-1] = 0.0 if success else -float(c_fail)

    starts = np.r_[0, onsets].astype(np.int64)
    ends = np.r_[onsets, length].astype(np.int64)
    stretch = np.zeros(length, dtype=np.int64)
    stretch[onsets] = 1
    stretch = np.cumsum(stretch)
    # Return-to-go of the whole episode minus the part after the frame's stretch; exact for integer rewards.
    to_go = np.r_[np.cumsum(reward[::-1])[::-1], 0.0]
    return_raw = to_go[:-1] - to_go[ends[stretch]]
    return {
        "reward": reward,
        "return_raw": return_raw,
        "stretch_index": stretch,
        "stretch_frame": np.arange(length, dtype=np.int64) - starts[stretch],
        "stretch_length": (ends - starts)[stretch],
    }


def default_intervention_penalty(episodes: pd.DataFrame, interventions: pd.Series) -> int:
    """Longest stretch that ends in a success (frames): the C_interv analogue of C_fail = T_max.

    With C_interv >= this, every frame of a stretch that ends in a takeover has a lower return than every frame of a
    stretch that ends in success, so "a human will have to step in" always scores below "on track".
    """
    longest = 0
    for row in episodes.itertuples(index=False):
        if not row.success:
            continue
        flags = _episode_flags(interventions, int(row.dataset_from_index), int(row.length))
        onsets = intervention_onsets(flags)
        longest = max(longest, int(row.length - (onsets[-1] if len(onsets) else 0)))
    if longest == 0:
        raise ValueError("no successful episode: cannot derive C_interv, pass it explicitly")
    return longest


def _episode_flags(interventions: pd.Series, from_index: int, length: int) -> np.ndarray:
    flags = interventions.reindex(np.arange(from_index, from_index + length))
    if flags.isna().any():
        raise KeyError(f"intervention flags missing for global indices {from_index}..{from_index + length - 1}")
    return flags.to_numpy(bool)


def load_interventions(raw_dir: Path) -> pd.Series:
    """Per-frame `intervention` flags of a LeRobot snapshot, as a bool Series indexed by global `index`."""
    files = sorted((Path(raw_dir) / "data").glob("chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no data/chunk-*/file-*.parquet under {raw_dir}")
    df = pd.concat([pd.read_parquet(f, columns=["index", "intervention"]) for f in files], ignore_index=True)
    if df["index"].duplicated().any():
        raise ValueError("duplicated global index in the data parquet(s)")
    return df.set_index("index")["intervention"].astype(bool).sort_index()


def load_episodes(raw_dir: Path | None = None, prefer_cache: bool = True, assume_success: bool = False) -> pd.DataFrame:
    """Episode table (`episode_index, length, success, dataset_from_index, dataset_to_index`).

    Uses step 1's `data/cache/episodes.parquet` if it exists, else the raw LeRobot `meta/episodes` parquet(s).
    Pass `prefer_cache=False` for any dataset other than the default one (the cache belongs to it).
    `assume_success`: the metadata has no `success` column and every episode is known to succeed (e.g. DAgger
    rollouts, where the human finishes the task); without it a missing column is an error.
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
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        if "success" not in df.columns:
            if not assume_success:
                raise KeyError(
                    f"{raw_dir}: episode metadata has no 'success' column (pass assume_success=True if "
                    "every episode is known to succeed)"
                )
            logger.warning("no 'success' column: every episode is treated as a success")
            df["success"] = True
        df = df[EPISODE_COLUMNS]
    df = df.sort_values("episode_index").reset_index(drop=True)
    df["success"] = df["success"].astype(bool)
    for col in ("episode_index", "length", "dataset_from_index", "dataset_to_index"):
        df[col] = df[col].astype(np.int64)
    if not (df["dataset_to_index"] - df["dataset_from_index"] == df["length"]).all():
        raise ValueError("dataset_to_index - dataset_from_index != length for some episodes")
    return df


def compute_returns_table(
    episodes: pd.DataFrame,
    c_fail: float,
    *,
    interventions: pd.Series | None = None,
    c_interv: float | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Full per-frame returns table (contract schema) and its metadata dict.

    With `interventions` (bool per global `index`, e.g. from `load_interventions`) and `c_interv`, returns are cut
    at each takeover (`compute_episode_returns_interventions`) and the table gains `INTERVENTION_COLUMNS`.
    """
    if (interventions is None) != (c_interv is None):
        raise ValueError("pass both `interventions` and `c_interv`, or neither")
    episodes = episodes.sort_values("episode_index").reset_index(drop=True)
    lengths = episodes["length"].to_numpy(np.int64)
    successes = episodes["success"].to_numpy(bool)
    if (lengths < 1).any():
        raise ValueError("all episodes must have length >= 1")
    if successes.all():
        logger.warning("no failure episodes: C_fail has no effect")

    from_index = episodes["dataset_from_index"].to_numpy(np.int64)
    rewards, returns = [], []
    extra: dict[str, list[np.ndarray]] = {c: [] for c in INTERVENTION_COLUMNS}
    for start, length, success in zip(from_index, lengths, successes, strict=True):
        if interventions is None:
            ep = compute_episode_returns(int(length), bool(success), c_fail)
        else:
            flags = _episode_flags(interventions, int(start), int(length))
            ep = compute_episode_returns_interventions(int(length), bool(success), c_fail, flags, c_interv)
            extra["intervention"].append(flags)
            for c in INTERVENTION_COLUMNS[1:]:
                extra[c].append(ep[c])
        rewards.append(ep["reward"])
        returns.append(ep["return_raw"])
    reward = np.concatenate(rewards)
    return_raw = np.concatenate(returns)

    z = float(-return_raw.min())
    if z <= 0:
        raise ValueError(f"normaliser Z must be > 0, got {z} (all returns are 0?)")
    return_norm = return_raw / z

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
    if interventions is not None:
        for c in INTERVENTION_COLUMNS:
            table[c] = np.concatenate(extra[c])
    table = table.sort_values("index", kind="stable").reset_index(drop=True)

    meta = {
        "tag": returns_tag(c_fail, c_interv),
        "c_fail": float(c_fail),
        "Z": z,
        "num_bins": bins.NUM_BINS,
        "v_min": bins.V_MIN,
        "v_max": bins.V_MAX,
        "t_max": int(lengths.max()),
        "t_max_success": int(lengths[successes].max()) if successes.any() else 0,
        "t_max_failure": int(lengths[~successes].max()) if (~successes).any() else 0,
    }
    if interventions is not None:
        first = table["stretch_frame"].to_numpy() == 0
        stretches = table.loc[first, ["episode_index", "stretch_index", "stretch_length"]]
        last = stretches.groupby("episode_index")["stretch_index"].transform("max") == stretches["stretch_index"]
        ep_success = dict(zip(episodes["episode_index"], successes, strict=True))
        ends_success = last & stretches["episode_index"].map(ep_success).astype(bool)
        n_int = stretches.groupby("episode_index").size() - 1
        meta.update(
            {
                "c_interv": float(c_interv),
                "num_interventions": int(n_int.sum()),
                "num_stretches": int(len(stretches)),
                "episodes_by_num_interventions": {str(k): int(v) for k, v in n_int.value_counts().sort_index().items()},
                "intervention_frame_fraction": float(table["intervention"].mean()),
                "stretch_length_max_success": int(stretches.loc[ends_success, "stretch_length"].max())
                if ends_success.any()
                else 0,
                "stretch_length_max_takeover": int(stretches.loc[~last, "stretch_length"].max())
                if (~last).any()
                else 0,
            }
        )
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


def _returns_paths(tag: str, cache_dir: Path | None) -> tuple[Path, Path]:
    if cache_dir is None:
        return paths.returns_paths(tag)
    return Path(cache_dir) / f"returns_{tag}.parquet", Path(cache_dir) / f"returns_{tag}.json"


def save_returns(
    table: pd.DataFrame, meta: dict, tag: str | None = None, cache_dir: Path | None = None
) -> tuple[Path, Path]:
    """Write `returns_{tag}.parquet` + `.json` to `cache_dir` (default: the cache directory)."""
    tag = tag or meta["tag"]
    pq_path, json_path = _returns_paths(tag, cache_dir)
    pq_path.parent.mkdir(parents=True, exist_ok=True)
    table[RETURNS_COLUMNS + [c for c in INTERVENTION_COLUMNS if c in table.columns]].to_parquet(pq_path, index=False)
    json_path.write_text(json.dumps(meta, indent=2) + "\n")
    return pq_path, json_path


def load_returns(tag: str = "c2229", cache_dir: Path | None = None) -> tuple[pd.DataFrame, dict]:
    """Load a returns table and its metadata (used by steps 4 and 6)."""
    pq_path, json_path = _returns_paths(tag, cache_dir)
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
