"""Step 7 — fixed-seed selection of the 20 evaluation episodes.

From the held-out split: 10 failures at random, the 5 successes with the deepest dip in V (peak-to-later-trough
drawdown of a 0.5 s rolling-median V), 5 further successes at random.

    env -u PYTHONPATH uv run python scripts/07_select_episodes.py --run vf_c2229_s0
    # after cross-fitting every episode is scored by a model that never saw it -> any episode may be used:
    env -u PYTHONPATH uv run python scripts/07_select_episodes.py --run vf_c2229_cf --pool all

Writes runs/<run>/eval_episodes.json: {"seed", "pool", ..., "episodes": [{episode_index, success, length, group,
dip_depth, dip_peak_frame, dip_trough_frame}, ...]}.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from recap_vf import paths
from recap_vf.data.frames import parse_episodes
from recap_vf.data.splits import load_splits
from recap_vf.viz import select_eval_episodes

log = logging.getLogger("07_select_episodes")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run name under runs/ (or a path to a run directory)")
    ap.add_argument("--values", default="values.parquet", help="values file in the run dir")
    ap.add_argument(
        "--pool", default="heldout", help="'heldout' (default), 'train', 'all', or an explicit episode spec like 0-9"
    )
    ap.add_argument(
        "--require-frames",
        action="store_true",
        help="restrict the pool to episodes present in the local frame store (laptop dev)",
    )
    ap.add_argument("--n-fail", type=int, default=10)
    ap.add_argument("--n-dip", type=int, default=5)
    ap.add_argument("--n-success", type=int, default=5)
    ap.add_argument("--smooth", type=int, default=25, help="rolling-median window [frames] before the dip search")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, help="default: runs/<run>/eval_episodes.json")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    run_dir = Path(args.run) if Path(args.run).is_dir() else paths.run_dir(args.run)
    episodes = pd.read_parquet(paths.EPISODE_TABLE, columns=["episode_index", "length", "success"])
    if args.pool in ("heldout", "train"):
        pool = load_splits()[args.pool]
    elif args.pool == "all":
        pool = episodes["episode_index"].tolist()
    else:
        pool = parse_episodes(args.pool)
    if args.pool != "heldout":
        log.warning("pool=%s: only meaningful with cross-fitted values (otherwise memorisation flatters V)", args.pool)
    if args.require_frames:
        pool = [e for e in pool if paths.episode_frames_path(e).is_file()]
    log.info("pool: %d episodes", len(pool))

    vals = pq.read_table(
        run_dir / args.values,
        columns=["episode_index", "frame_index", "value"],
        filters=[("episode_index", "in", sorted(pool))],
    ).to_pandas()
    sel = select_eval_episodes(
        episodes,
        pool,
        vals,
        seed=args.seed,
        n_fail=args.n_fail,
        n_dip=args.n_dip,
        n_success=args.n_success,
        smooth=args.smooth,
    )
    out = args.out or run_dir / "eval_episodes.json"
    out.write_text(
        json.dumps(
            {
                "run": str(run_dir),
                "values": args.values,
                "seed": args.seed,
                "pool": args.pool,
                "pool_size": len(pool),
                "require_frames": args.require_frames,
                "n_fail": args.n_fail,
                "n_dip": args.n_dip,
                "n_success": args.n_success,
                "smooth": args.smooth,
                "episodes": sel,
            },
            indent=2,
        )
    )
    for x in sel:
        log.info(
            "%-20s ep %04d len %4d %s",
            x["group"],
            x["episode_index"],
            x["length"],
            f"dip {x['dip_depth']:.3f} @ {x['dip_trough_frame']}" if "dip_depth" in x else "",
        )
    log.info("wrote %s (%d episodes)", out, len(sel))


if __name__ == "__main__":
    main()
