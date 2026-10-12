"""Step 7.1 — render evaluation mp4s: three cameras + V(t), g(t), A(t), I_t shading and a moving cursor.

    env -u PYTHONPATH uv run python scripts/07_render_episode.py --run vf_c2229_s0 --episode 0
    env -u PYTHONPATH uv run python scripts/07_render_episode.py --run vf_c2229_s0 \
        --episodes-file runs/vf_c2229_s0/eval_episodes.json            # the 20 episodes from 07_select_episodes.py

Writes runs/<run>/videos/ep_XXXX.mp4 (every 2nd frame at 25 fps = real time, H.264 CRF 28) and
runs/<run>/videos/render_log.json (time and size per episode).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from recap_vf import paths
from recap_vf.constants import CAMERAS
from recap_vf.data.dataset import FrameDataset
from recap_vf.data.frames import parse_episodes
from recap_vf.viz import (
    episode_success,
    load_advantage_meta,
    load_episode_series,
    read_episode_list,
    render_episode_video,
)

log = logging.getLogger("07_render_episode")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run name under runs/ (or a path to a run directory)")
    ap.add_argument("--episode", "--episodes", dest="episodes", help="e.g. 0 | 0-9 | 0,3,5-7")
    ap.add_argument("--episodes-file", type=Path, help="JSON written by scripts/07_select_episodes.py")
    ap.add_argument(
        "--advantages",
        default="advantages_50_0.4",
        help="advantages file stem in the run dir; 'none' = values.parquet only (no A / I)",
    )
    ap.add_argument("--returns-tag", default="c2229")
    ap.add_argument("--out-dir", type=Path, help="default: runs/<run>/videos")
    ap.add_argument("--stride", type=int, default=2, help="write every n-th frame")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--crf", type=int, default=28)
    ap.add_argument("--preset", default="medium", help="libx264 preset")
    ap.add_argument("--panel-height", type=int, default=300)
    ap.add_argument("--max-frames", type=int, help="debug: stop after this many written frames")
    ap.add_argument("--seed", type=int, default=0, help="unused (deterministic); recorded for the conventions")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    run_dir = Path(args.run) if Path(args.run).is_dir() else paths.run_dir(args.run)
    episodes = set(parse_episodes(args.episodes)) if args.episodes else set()
    if args.episodes_file:
        episodes |= set(read_episode_list(args.episodes_file))
    if not episodes:
        ap.error("give --episode(s) and/or --episodes-file")
    advantages = None if args.advantages.lower() == "none" else args.advantages
    meta = load_advantage_meta(run_dir, advantages)
    out_dir = args.out_dir or run_dir / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = FrameDataset(sorted(episodes), CAMERAS)
    results = []
    for e in sorted(episodes):
        series = load_episode_series(run_dir, e, advantages, args.returns_tag)
        res = render_episode_video(
            ds,
            e,
            series,
            out_dir / f"ep_{e:04d}.mp4",
            success=episode_success(e),
            epsilon=meta.get("epsilon"),
            stride=args.stride,
            fps=args.fps,
            crf=args.crf,
            panel_height=args.panel_height,
            max_frames=args.max_frames,
            preset=args.preset,
        )
        log.info(
            "ep %04d: %d frames in %.1f s -> %s (%.2f MB)",
            e,
            res["frames_written"],
            res["seconds"],
            res["path"],
            res["bytes"] / 1e6,
        )
        results.append(res)
    ds.close()

    log_path = out_dir / "render_log.json"
    old = json.loads(log_path.read_text()) if log_path.is_file() else {}
    old.update(
        {
            f"{r['episode']:04d}": r
            | {
                "advantages": advantages,
                "returns_tag": args.returns_tag,
                "stride": args.stride,
                "fps": args.fps,
                "crf": args.crf,
            }
            for r in results
        }
    )
    log_path.write_text(json.dumps(old, indent=2))
    tot_s = sum(r["seconds"] for r in results)
    tot_b = sum(r["bytes"] for r in results)
    log.info(
        "%d episodes, %.1f s total (%.1f s/episode), %.1f MB total",
        len(results),
        tot_s,
        tot_s / len(results),
        tot_b / 1e6,
    )


if __name__ == "__main__":
    main()
