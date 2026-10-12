"""Step 2: per-frame rewards, Monte-Carlo returns and 201-bin value targets for one or more C_fail values.

Full scale:   env -u PYTHONPATH uv run python scripts/02_compute_returns.py
Smoke scale:  env -u PYTHONPATH uv run python scripts/02_compute_returns.py --max-episodes 10 --no-figures
Writes data/cache/returns_{tag}.parquet + .json (tag = c{int(C_fail)}, suffixed `_smoke{N}` with --max-episodes)
and figures under reports/figures/step_02/.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from recap_vf import bins, paths
from recap_vf.returns import (
    bin_counts,
    compute_returns_table,
    load_episodes,
    returns_tag,
    save_returns,
    summarize_returns,
    verify_index,
)

logger = logging.getLogger("02_compute_returns")

DEFAULT_CONFIG = paths.REPO_ROOT / "configs" / "returns.yaml"
FIG_DIR = paths.FIGURES_DIR / "step_02"
COLORS = {"success": "#2a78d6", "failure": "#eb6834"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--c-fail", type=float, nargs="+", default=None, help="override configs/returns.yaml c_fail")
    p.add_argument("--max-episodes", type=int, default=None, help="smoke scale: only the first N episodes")
    p.add_argument("--raw-dir", type=Path, default=None, help="LeRobot snapshot (default: paths.RAW_DIR)")
    p.add_argument("--no-cache", action="store_true", help="ignore data/cache/episodes.parquet, read raw meta")
    p.add_argument("--no-verify", action="store_true", help="skip the check against the frame table")
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def load_frame_index(raw_dir: Path) -> pd.DataFrame | None:
    """(index, episode_index, frame_index) from step 1's frame table, else the raw LeRobot data parquet(s)."""
    cols = ["index", "episode_index", "frame_index"]
    if paths.FRAME_TABLE.exists():
        return pd.read_parquet(paths.FRAME_TABLE, columns=cols)
    files = sorted((raw_dir / "data").glob("chunk-*/file-*.parquet"))
    if not files:
        return None
    return pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)


def plot_histogram(table: pd.DataFrame, meta: dict, out: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=110)
    edges = np.linspace(bins.V_MIN - bins.BIN_WIDTH / 2, bins.V_MAX + bins.BIN_WIDTH / 2, bins.NUM_BINS + 1)
    for name, mask in (("success", table["success"]), ("failure", ~table["success"])):
        ax.hist(
            table.loc[mask, "return_norm"],
            bins=edges,
            color=COLORS[name],
            alpha=0.85,
            label=f"{name} frames ({int(mask.sum()):,})",
        )
    ax.set_xlim(-1.02, 0.02)
    ax.set_xlabel("normalised return-to-go  g_t = G_t / Z  (one bar = one value bin)")
    ax.set_ylabel("frames")
    ax.set_title(f"{meta['tag']}: C_fail = {meta['c_fail']:.0f}, Z = {meta['Z']:.0f}", fontsize=10)
    _style(ax)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_examples(table: pd.DataFrame, meta: dict, n: int, seed: int, out: Path) -> None:
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(8, 3.8), dpi=110)
    ep = table.groupby("episode_index")["success"].first()
    for name, flag in (("success", True), ("failure", False)):
        pool = ep.index[ep == flag].to_numpy()
        chosen = rng.choice(pool, size=min(n, len(pool)), replace=False) if len(pool) else []
        for i, e in enumerate(sorted(chosen)):
            sub = table[table["episode_index"] == e]
            ax.plot(
                sub["frame_index"] / sub["episode_length"],
                sub["return_norm"],
                color=COLORS[name],
                lw=1.4,
                alpha=0.8,
                label=name if i == 0 else None,
            )
    ax.set_xlim(0, 1)
    ax.set_ylim(-1.02, 0.02)
    ax.set_xlabel("t / T")
    ax.set_ylabel("g_t")
    ax.set_title(f"{meta['tag']}: g_t for {n} random episodes per class (seed {seed})", fontsize=10)
    _style(ax)
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        ["success\nepisodes", "failure\nepisodes"],
        frameon=False,
        fontsize=9,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
    )
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_bin_occupancy(counts: pd.DataFrame, meta: dict, out: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(7.5, 5), dpi=110, sharex=True)
    for ax, log in zip(axes, (False, True), strict=True):
        ax.bar(counts["bin_index"], counts["success"], width=1.0, color=COLORS["success"], label="success")
        ax.bar(
            counts["bin_index"],
            counts["failure"],
            width=1.0,
            bottom=counts["success"],
            color=COLORS["failure"],
            label="failure",
        )
        if log:
            ax.set_yscale("log")
            ax.set_ylabel("frames (log)")
        else:
            ax.set_ylabel("frames")
        _style(ax)
    used = int((counts["total"] > 0).sum())
    axes[0].set_title(
        f"{meta['tag']}: frames per value bin ({used}/{bins.NUM_BINS} bins used, "
        f"max {int(counts['total'].max()):,} in bin {int(counts['total'].idxmax())})",
        fontsize=10,
    )
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    axes[1].set_xlabel("bin index (0 → g = −1, 200 → g = 0)")
    axes[1].set_xlim(-1, bins.NUM_BINS)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def _style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#dddddd", lw=0.6)
    ax.set_axisbelow(True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text()) if args.config.exists() else {}
    c_fails = args.c_fail or cfg.get("c_fail", [2229])
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 0))
    verify = cfg.get("verify_frame_table", True) and not args.no_verify
    figures = cfg.get("figures", True) and not args.no_figures
    n_examples = int(cfg.get("num_example_episodes", 10))
    raw_dir = args.raw_dir or paths.RAW_DIR

    episodes = load_episodes(raw_dir, prefer_cache=not args.no_cache)
    suffix = ""
    if args.max_episodes is not None:
        episodes = episodes.head(args.max_episodes)
        suffix = f"_smoke{args.max_episodes}"
    logger.info(
        "%d episodes (%d success / %d failure), %d frames, seed %d",
        len(episodes),
        int(episodes["success"].sum()),
        int((~episodes["success"]).sum()),
        int(episodes["length"].sum()),
        seed,
    )

    frame_index = load_frame_index(raw_dir) if verify else None
    if verify and frame_index is None:
        logger.warning("no frame table found; skipping index verification")
    if frame_index is not None:
        frame_index = frame_index[frame_index["episode_index"].isin(episodes["episode_index"])]

    if figures:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
    summaries = []
    for c_fail in c_fails:
        table, meta = compute_returns_table(episodes, c_fail)
        tag = returns_tag(c_fail) + suffix
        meta["tag"] = tag
        meta["seed"] = seed
        meta["num_episodes"] = len(episodes)
        meta["num_frames"] = len(table)
        if frame_index is not None:
            verify_index(table, frame_index)
            logger.info("[%s] index/episode_index/frame_index match the frame table (%d rows)", tag, len(table))
        pq_path, json_path = save_returns(table, meta, tag)
        logger.info("[%s] wrote %s and %s", tag, pq_path, json_path)

        summary = summarize_returns(table, meta)
        summaries.append(summary)
        logger.info("[%s] summary:\n%s", tag, json.dumps(summary, indent=2))

        if figures:
            counts = bin_counts(table)
            counts.to_csv(FIG_DIR / f"bin_counts_{tag}.csv", index=False)
            plot_histogram(table, meta, FIG_DIR / f"return_hist_{tag}.png")
            plot_examples(table, meta, n_examples, seed, FIG_DIR / f"return_examples_{tag}.png")
            plot_bin_occupancy(counts, meta, FIG_DIR / f"bin_occupancy_{tag}.png")
            logger.info("[%s] figures written to %s", tag, FIG_DIR)

    if figures:
        (FIG_DIR / f"summary{suffix}.json").write_text(json.dumps(summaries, indent=2) + "\n")


if __name__ == "__main__":
    main()
