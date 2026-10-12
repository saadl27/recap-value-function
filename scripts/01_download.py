"""Step 1.1 — download the HF dataset snapshot into data/raw/ and validate it against meta/info.json.

Laptop:       env -u PYTHONPATH uv run python scripts/01_download.py --meta-only    (~60 MB: meta/* + data/*)
Workstation:  uv run python scripts/01_download.py                                   (full snapshot, ~5.3 GB)
Files already present (same etag) are skipped by huggingface_hub, so the script can be re-run safely.
"""

import argparse
import json
import logging
import time
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from recap_vf import paths
from recap_vf.constants import CAMERAS, DATASET_REPO_ID
from recap_vf.data.frames import load_raw_episodes, validate_info, video_spans

log = logging.getLogger("01_download")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meta-only", action="store_true", help="only meta/* and data/* (no videos)")
    ap.add_argument("--local-dir", type=Path, default=paths.RAW_DIR)
    ap.add_argument("--revision", default=None, help="dataset git revision (default: main)")
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0, help="unused; recorded for uniformity")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    allow = ["meta/*", "data/*"] if args.meta_only else None
    t0 = time.time()
    snapshot_download(
        repo_id=DATASET_REPO_ID,
        repo_type="dataset",
        local_dir=args.local_dir,
        allow_patterns=allow,
        revision=args.revision,
        max_workers=args.max_workers,
    )
    log.info("snapshot ready in %.1fs at %s", time.time() - t0, args.local_dir)

    info = validate_info(args.local_dir)
    log.info(
        "meta/info.json OK: %d episodes, %d frames, %d fps", info["total_episodes"], info["total_frames"], info["fps"]
    )
    raw_eps = load_raw_episodes(args.local_dir)
    if len(raw_eps) != info["total_episodes"]:
        raise SystemExit(f"meta/episodes has {len(raw_eps)} rows, info.json says {info['total_episodes']}")

    videos = {}
    for cam in CAMERAS:
        rel = video_spans(raw_eps, cam, info).video_path
        p = args.local_dir / rel
        videos[cam] = {"path": rel, "present": p.is_file(), "bytes": p.stat().st_size if p.is_file() else None}
        if not args.meta_only and not p.is_file():
            raise SystemExit(f"missing video {p}")
        if p.is_file():
            import av

            with av.open(str(p)) as c:
                n = c.streams.video[0].frames
            videos[cam]["frames"] = n
            if n != info["total_frames"]:
                raise SystemExit(f"{rel}: {n} frames in the stream, info.json says {info['total_frames']}")
    log.info("videos: %s", videos)

    try:
        sha = HfApi().dataset_info(DATASET_REPO_ID, revision=args.revision).sha
    except Exception as err:  # noqa: BLE001 - offline is fine, the record is informative
        sha = f"unknown ({err.__class__.__name__})"
    record = {
        "repo_id": DATASET_REPO_ID,
        "revision": args.revision or "main",
        "sha": sha,
        "meta_only": args.meta_only,
        "local_dir": str(args.local_dir),
        "videos": videos,
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out = paths.CACHE_DIR / "logs" / "01_download.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1) + "\n")
    log.info("dataset sha %s; record written to %s", sha, out)


if __name__ == "__main__":
    main()
