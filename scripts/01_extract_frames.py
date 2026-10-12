"""Steps 1.2 + 1.3 — write episodes.parquet / frames.parquet, then decode the mp4s into data/cache/frames/ep_XXXX.h5.

Laptop smoke test (streams the first episodes from the HF hub, no video download):
    env -u PYTHONPATH uv run python scripts/01_extract_frames.py --source url --episodes 0-9
Workstation (after `scripts/01_download.py`):
    uv run python scripts/01_extract_frames.py --source local --episodes all

Resumable: episodes whose .h5 already holds `length` frames per camera are skipped (use --overwrite to redo them).
Tables always cover all 800 episodes (they come from the small tabular files); the frame store only the requested ones.
"""

import argparse
import json
import logging
import time
from pathlib import Path

from recap_vf import paths
from recap_vf.constants import CAMERAS
from recap_vf.data.frames import JPEG_QUALITY, extract_frames, parse_episodes, write_tables

log = logging.getLogger("01_extract_frames")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", default="all", help="'all', '0-9', '0,5,10-12' (inclusive ranges)")
    ap.add_argument("--source", default="local", help="'local' (data/raw mp4s), 'url' (stream from HF) or a dir")
    ap.add_argument("--cameras", nargs="+", default=list(CAMERAS), choices=list(CAMERAS))
    ap.add_argument("--quality", type=int, default=JPEG_QUALITY)
    ap.add_argument("--overwrite", action="store_true", help="re-extract episodes that are already complete")
    ap.add_argument("--skip-tables", action="store_true", help="do not rebuild episodes/frames.parquet")
    ap.add_argument("--tables-only", action="store_true")
    ap.add_argument("--queue-size", type=int, default=4, help="episodes buffered per camera process")
    ap.add_argument("--max-retries", type=int, default=5, help="reconnects on transient HTTP errors (url source)")
    ap.add_argument("--revision", default="main", help="HF revision for --source url")
    ap.add_argument("--seed", type=int, default=0, help="unused (deterministic); recorded for uniformity")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    record: dict = {"args": vars(args), "start": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if not args.skip_tables:
        t0 = time.time()
        record["tables"] = write_tables(paths.RAW_DIR, paths.CACHE_DIR)
        log.info("tables OK in %.1fs: %s", time.time() - t0, json.dumps(record["tables"]))
    if args.tables_only:
        return

    import pandas as pd

    n_eps = len(pd.read_parquet(paths.EPISODE_TABLE, columns=["episode_index"]))
    episodes = parse_episodes(args.episodes, n_eps)
    log.info("extracting %d episodes (%s) from %s", len(episodes), args.episodes, args.source)
    stats = extract_frames(
        episodes,
        source=args.source,
        cameras=args.cameras,
        quality=args.quality,
        overwrite=args.overwrite,
        queue_size=args.queue_size,
        max_retries=args.max_retries,
        revision=args.revision,
    )
    record["extract"] = stats
    gb = stats["store_file_bytes"] / 1e9
    log.info(
        "done: wrote %d episodes / %d frames in %.1fs (%.0f frames/s per camera); store for requested episodes: "
        "%d episodes, %.3f GB, %.0f JPEG bytes/image, %.0f file bytes/frame (3 cams)",
        stats["written_episodes"],
        stats["written_frames"],
        stats["wall_time_s"],
        stats.get("frames_per_s_per_camera", 0.0),
        stats["store_episodes"],
        gb,
        stats["mean_jpeg_bytes_per_image"],
        stats["mean_file_bytes_per_frame_3cams"],
    )
    out = paths.CACHE_DIR / "logs" / f"01_extract_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1, default=str) + "\n")
    log.info("run record: %s", Path(out))


if __name__ == "__main__":
    main()
