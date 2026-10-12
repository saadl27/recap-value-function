"""Reproducible episode splits: a stratified held-out set and 5 stratified cross-fitting folds.

Strata = success flag x episode-length quantile bin (computed within each success class), so that held-out and
every fold have the same success rate and a representative length distribution.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from recap_vf import paths

HELDOUT_FRAC = 0.15
NUM_FOLDS = 5
LENGTH_BINS = 4


def length_strata(episodes: pd.DataFrame, n_length_bins: int = LENGTH_BINS) -> pd.Series:
    """Stratum label per episode: `success * n_length_bins + length quantile bin within that success class`."""
    labels = pd.Series(0, index=episodes.index, dtype=np.int64)
    for s, grp in episodes.groupby("success"):
        k = min(n_length_bins, len(grp))
        # rank first so ties never collapse bins; deterministic (ties broken by episode order)
        q = pd.qcut(grp["length"].rank(method="first"), k, labels=False) if k > 1 else 0
        labels.loc[grp.index] = int(s) * n_length_bins + np.asarray(q, dtype=np.int64)
    return labels


def _largest_remainder(sizes: np.ndarray, frac: float, total: int) -> np.ndarray:
    """Integer quotas ~ sizes * frac that sum to `total` (Hamilton / largest-remainder rounding)."""
    raw = sizes * frac
    quota = np.floor(raw).astype(np.int64)
    order = np.argsort(-(raw - quota), kind="stable")
    for i in order[: total - quota.sum()]:
        quota[i] += 1
    return np.minimum(quota, sizes)


def make_splits(
    episodes: pd.DataFrame,
    seed: int = 0,
    heldout_frac: float = HELDOUT_FRAC,
    n_folds: int = NUM_FOLDS,
    n_length_bins: int = LENGTH_BINS,
) -> dict:
    """Build the `splits.json` dict from an episode table with columns `episode_index, length, success`.

    heldout: round(heldout_frac * E) episodes, allocated to success classes then to length bins by largest remainder.
    folds: all episodes, independent of heldout; members of each stratum are shuffled and dealt round-robin with a
    counter that continues across strata, so fold sizes differ by <= 1 overall, per success class and per stratum.
    """
    eps = episodes[["episode_index", "length", "success"]].reset_index(drop=True).copy()
    eps["success"] = eps["success"].astype(bool)
    eps["stratum"] = length_strata(eps, n_length_bins)
    rng = np.random.default_rng(seed)

    # ---- held-out: success class first, then length bins inside each class
    n_heldout = int(round(heldout_frac * len(eps)))
    classes = sorted(eps["success"].unique())
    class_sizes = np.array([(eps["success"] == c).sum() for c in classes])
    class_quota = _largest_remainder(class_sizes, heldout_frac, n_heldout)
    heldout: list[int] = []
    for c, cq in zip(classes, class_quota, strict=True):
        strata = sorted(eps.loc[eps["success"] == c, "stratum"].unique())
        sizes = np.array([(eps["stratum"] == s).sum() for s in strata])
        for s, q in zip(strata, _largest_remainder(sizes, cq / sizes.sum(), int(cq)), strict=True):
            members = eps.loc[eps["stratum"] == s, "episode_index"].to_numpy()
            heldout.extend(rng.permutation(members)[:q].tolist())
    heldout_set = set(heldout)
    train = [int(e) for e in eps["episode_index"] if e not in heldout_set]

    # ---- folds over all episodes
    order: list[int] = []
    for s in sorted(eps["stratum"].unique()):
        members = eps.loc[eps["stratum"] == s, "episode_index"].to_numpy()
        order.extend(rng.permutation(members).tolist())
    folds: dict[str, list[int]] = {str(k): [] for k in range(n_folds)}
    for pos, e in enumerate(order):
        folds[str(pos % n_folds)].append(int(e))

    return {
        "seed": int(seed),
        "heldout": sorted(int(e) for e in heldout),
        "train": sorted(train),
        "folds": {k: sorted(v) for k, v in folds.items()},
        "heldout_frac": heldout_frac,
        "n_length_bins": n_length_bins,
        "stratified_on": "success x length quantile (within success class)",
    }


def split_summary(episodes: pd.DataFrame, splits: dict) -> pd.DataFrame:
    """One row per split: #episodes, #success, success rate, length quartiles, #frames."""
    eps = episodes.set_index("episode_index")
    groups = {"all": list(eps.index), "train": splits["train"], "heldout": splits["heldout"]}
    groups.update({f"fold{k}": v for k, v in splits["folds"].items()})
    rows = []
    for name, members in groups.items():
        sub = eps.loc[members]
        q = sub["length"].quantile([0.25, 0.5, 0.75]).to_numpy()
        rows.append(
            {
                "split": name,
                "episodes": len(sub),
                "success": int(sub["success"].sum()),
                "success_rate": float(sub["success"].mean()),
                "len_q25": float(q[0]),
                "len_median": float(q[1]),
                "len_q75": float(q[2]),
                "frames": int(sub["length"].sum()),
            }
        )
    return pd.DataFrame(rows).set_index("split")


def save_splits(splits: dict, path: Path = paths.SPLITS_FILE) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(splits, indent=1) + "\n")
    tmp.replace(path)


def load_splits(path: Path = paths.SPLITS_FILE) -> dict:
    """The `splits.json` dict: keys `seed`, `heldout`, `train`, `folds` ("0".."4"), plus metadata."""
    return json.loads(Path(path).read_text())
