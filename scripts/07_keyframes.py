"""Step 7.2 — keyframe strips at the k most negative / most positive advantages of an episode.

    env -u PYTHONPATH uv run python scripts/07_keyframes.py --run vf_c2229_s0 --episode 0 --k 3
    env -u PYTHONPATH uv run python scripts/07_keyframes.py --run vf_c2229_s0 \
        --episodes-file runs/vf_c2229_s0/eval_episodes.json

For each selected time t (picks pairwise >= --min-sep-s apart) saves a PNG with the three cameras (rows) at
t-1 s, t, t+1 s, t+2 s (columns), V and A printed above each column and the episode's V/g/A panel below, to
<out-dir>/ep_XXXX_{drop,rise}_{i}.png (default out-dir: runs/<run>/keyframes; copy only the few best stills
into reports/figures/step_07). Also writes
<out-dir>/keyframes.json with the selected times and values.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from recap_vf import paths
from recap_vf.constants import CAMERAS, FPS
from recap_vf.data.dataset import FrameDataset
from recap_vf.data.frames import parse_episodes
from recap_vf.viz import (
    episode_success,
    load_advantage_meta,
    load_episode_series,
    read_episode_list,
    render_keyframe_strip,
    select_keyframes,
)

log = logging.getLogger("07_keyframes")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run name under runs/ (or a path to a run directory)")
    ap.add_argument("--episode", "--episodes", dest="episodes", help="e.g. 0 | 0-9 | 0,3,5-7")
    ap.add_argument("--episodes-file", type=Path, help="JSON written by scripts/07_select_episodes.py")
    ap.add_argument("--advantages", default="advantages_50_0.4")
    ap.add_argument("--returns-tag", default="c2229")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--min-sep-s", type=float, default=2.0, help="minimum spacing between picks [s]")
    ap.add_argument("--out-dir", type=Path, default=None, help="default: runs/<run>/keyframes")
    ap.add_argument("--seed", type=int, default=0, help="unused (deterministic); recorded for the conventions")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    run_dir = Path(args.run) if Path(args.run).is_dir() else paths.run_dir(args.run)
    episodes = set(parse_episodes(args.episodes)) if args.episodes else set()
    if args.episodes_file:
        episodes |= set(read_episode_list(args.episodes_file))
    if not episodes:
        ap.error("give --episode(s) and/or --episodes-file")
    meta = load_advantage_meta(run_dir, args.advantages)
    if args.out_dir is None:
        args.out_dir = run_dir / "keyframes"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    min_sep = int(round(args.min_sep_s * FPS))

    ds = FrameDataset(sorted(episodes), CAMERAS)
    index_path = args.out_dir / "keyframes.json"
    summary = json.loads(index_path.read_text()) if index_path.is_file() else {}
    for e in sorted(episodes):
        series = load_episode_series(run_dir, e, args.advantages, args.returns_tag)
        success = episode_success(e)
        adv = series["advantage"].to_numpy()
        for old in args.out_dir.glob(f"ep_{e:04d}_*_*.png"):  # stale strips from an earlier run
            if old.stem.split("_")[2] in ("drop", "rise"):
                old.unlink()
        picks = []
        for mode in ("drop", "rise"):
            for i, t in enumerate(select_keyframes(adv, args.k, min_sep, mode)):
                row = series.iloc[t]
                label = f"· {mode} #{i}: A {row['advantage']:+.4f} at {t / FPS:.2f} s (frame {t})"
                out = args.out_dir / f"ep_{e:04d}_{mode}_{i}.png"
                render_keyframe_strip(ds, e, series, t, out, label=label, success=success, epsilon=meta.get("epsilon"))
                picks.append(
                    {
                        "mode": mode,
                        "rank": i,
                        "frame": t,
                        "time_s": round(t / FPS, 2),
                        "value": float(row["value"]),
                        "advantage": float(row["advantage"]),
                        "indicator": bool(row["indicator"]),
                        "png": str(out),
                    }
                )
                log.info("ep %04d %s #%d: frame %d A=%+.4f -> %s", e, mode, i, t, row["advantage"], out)
        summary[f"{e:04d}"] = {
            "run": str(run_dir),
            "advantages": args.advantages,
            "success": success,
            "length": len(series),
            "min_sep_frames": min_sep,
            "picks": picks,
        }
    ds.close()
    index_path.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
