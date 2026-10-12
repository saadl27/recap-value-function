"""Returns for a dataset with human interventions: penalty before each takeover, return-to-go cut at the next one.

    RECAP_DATASET=actuator_unboxing_1h_HQ_dagger_rollouts env -u PYTHONPATH uv run python \
        scripts/02_compute_intervention_returns.py

Reads the snapshot's `meta/episodes` and the per-frame `intervention` flag of `data/*.parquet` (no videos needed) and
writes `<out-dir>/returns_i{C}.parquet` + `.json` (default: the dataset's cache dir) plus an example figure.
C_interv defaults to the longest stretch that ends in success (see `returns.default_intervention_penalty`).
The reward is explained in docs/interventions.md.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from recap_vf import bins, paths
from recap_vf.constants import DATASETS
from recap_vf.returns import (
    compute_returns_table,
    default_intervention_penalty,
    load_episodes,
    load_interventions,
    save_returns,
)

logger = logging.getLogger("02_compute_intervention_returns")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", type=Path, default=paths.RAW_DIR, help="snapshot with an `intervention` column")
    p.add_argument("--out-dir", type=Path, default=None, help="default: the cache dir of the raw-dir's dataset")
    p.add_argument("--c-interv", default="auto", help="penalty before each takeover: 'auto' or a number of frames")
    p.add_argument(
        "--c-fail", type=float, default=None, help="penalty at the end of a failed episode (default C_interv)"
    )
    p.add_argument(
        "--assume-success",
        action="store_true",
        help="metadata has no success column: all episodes succeed (implied for datasets marked so in constants)",
    )
    p.add_argument("--fig-dir", type=Path, default=paths.FIGURES_DIR / "interventions")
    p.add_argument("--no-figures", action="store_true")
    return p.parse_args()


def check_frames(table: pd.DataFrame, raw_dir: Path) -> None:
    """(index, episode_index, frame_index) of the returns table must match the snapshot's frame table exactly."""
    cols = ["index", "episode_index", "frame_index"]
    files = sorted((raw_dir / "data").glob("chunk-*/file-*.parquet"))
    ref = pd.concat([pd.read_parquet(f, columns=cols) for f in files]).sort_values("index").reset_index(drop=True)
    if not ref.equals(table[cols].reset_index(drop=True)):
        raise ValueError("returns table and data parquet disagree on (index, episode_index, frame_index)")


def plot_examples(table: pd.DataFrame, meta: dict, out: Path) -> None:
    """g_t for a few episodes with 0, 1 and 2+ interventions; human control shaded, penalty frames marked."""
    import matplotlib.pyplot as plt

    n_int = table.groupby("episode_index")["stretch_index"].max()
    picks = [e for k in (0, 1, 2) for e in n_int.index[n_int.clip(upper=2) == k][:2]]
    fig, axes = plt.subplots(len(picks), 1, figsize=(8, 1.5 * len(picks)), dpi=110, sharex=True, squeeze=False)
    for ax, e in zip(axes[:, 0], picks, strict=True):
        sub = table[table["episode_index"] == e]
        t = sub["frame_index"].to_numpy() / 50
        h = sub["intervention"].to_numpy(bool)
        ax.fill_between(t, -1, 0, where=h, color="#eb6834", alpha=0.18, lw=0, step="mid", label="human control")
        ax.plot(t, sub["return_norm"], color="#2a78d6", lw=1.4, label="g_t (cut at each takeover)")
        pen = sub["reward"].to_numpy() <= -meta["c_interv"]
        ax.plot(t[pen], sub["return_norm"].to_numpy()[pen], "v", color="#eb6834", ms=5, label="−C_interv")
        ax.set_ylim(-1.02, 0.02)
        ax.set_ylabel(f"ep {e}", fontsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=7, ncol=3, loc="lower right")
    axes[-1, 0].set_xlabel("time [s]")
    fig.suptitle(f"{meta['tag']}: C_interv = {meta['c_interv']:.0f}, Z = {meta['Z']:.0f}", fontsize=9)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    raw_dir = args.raw_dir
    out_dir = args.out_dir or paths.cache_dir_for(raw_dir.name)
    assume_success = args.assume_success or DATASETS.get(raw_dir.name, {}).get("all_succeed", False)
    episodes = load_episodes(raw_dir, prefer_cache=False, assume_success=assume_success)
    interventions = load_interventions(raw_dir)
    c_interv = (
        default_intervention_penalty(episodes, interventions) if args.c_interv == "auto" else float(args.c_interv)
    )
    c_fail = float(args.c_fail) if args.c_fail is not None else float(c_interv)
    table, meta = compute_returns_table(episodes, c_fail, interventions=interventions, c_interv=c_interv)
    check_frames(table, raw_dir)
    meta.update(
        {
            "dataset": raw_dir.name,
            "c_interv_rule": "auto: longest stretch ending in success" if args.c_interv == "auto" else "given",
            "assume_success": bool(assume_success),
            "num_episodes": int(len(episodes)),
            "num_frames": int(len(table)),
        }
    )
    pq_path, json_path = save_returns(table, meta, cache_dir=out_dir)
    logger.info("wrote %s (%d rows) and %s", pq_path, len(table), json_path)
    logger.info("metadata:\n%s", json.dumps(meta, indent=2))

    last = table["stretch_index"] == table.groupby("episode_index")["stretch_index"].transform("max")
    g = table["return_norm"]
    logger.info(
        "return_norm: stretches ending in a takeover %.3f..%.3f, ending in success %.3f..%.3f; bins used %d/%d",
        g[~last].min(),
        g[~last].max(),
        g[last].min(),
        g[last].max(),
        table["bin_index"].nunique(),
        bins.NUM_BINS,
    )
    if not args.no_figures:
        args.fig_dir.mkdir(parents=True, exist_ok=True)
        out = args.fig_dir / f"{raw_dir.name}_{meta['tag']}_examples.png"
        plot_examples(table, meta, out)
        logger.info("figure: %s", out)


if __name__ == "__main__":
    main()
