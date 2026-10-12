"""Step 8.1 — blind contact sheets (cameras only) for the failure-time / fumble annotations.

    # laptop smoke test (local frame store = episodes 0-9)
    env -u PYTHONPATH uv run python scripts/08_contact_sheets.py --episodes 0-9 --stills
    # workstation, CPU only, low priority (the GPU and the training dataloaders are busy)
    CUDA_VISIBLE_DEVICES="" nice -n 19 uv run python scripts/08_contact_sheets.py --split heldout --workers 4 --stills
    # zoom: every 0.2 s around a candidate event (one window for all --episodes, or per-episode --zoom EP:T0:T1)
    ... scripts/08_contact_sheets.py --episodes 409 --window 3 8 --every 0.2
    ... scripts/08_contact_sheets.py --zoom 409:3:8 --zoom 588:4:9.5 --every 0.2

Overview: one tile per second (+ the last frame), 4 tiles per row at 0.75 scale. Each tile is
top | left wrist | right wrist with a label "ep N · t = …s · outcome". The sheets never show V, g, A or labels: the
annotations must be made blind (docs/steps/08_failure_annotations.md, 8.2). Pages of --rows-per-page rows are written
to <out-dir>/ep_XXXX_overview_pN.jpg and <out-dir>/ep_XXXX_zoom_<t0>-<t1>_pN.jpg (default out-dir:
runs/annotation_sheets).

--stills also writes <out-dir>/stills.csv: candidate `stall` intervals (> --still-min-s) where every joint of both
arms moves less than --still-range-rad within a centred 1 s window, from the joint states in frames.parquet. It is
an aid for the visual check, not an annotation.
"""

from __future__ import annotations

import argparse
import io
import logging
import math
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image, ImageDraw

from recap_vf import paths
from recap_vf.constants import CAMERAS, FPS, IMAGE_SIZE
from recap_vf.data.frames import parse_episodes
from recap_vf.data.splits import load_splits
from recap_vf.viz import CAMERA_LABELS, get_font

log = logging.getLogger("08_contact_sheets")

HEADER_H = 24
LABEL_H = 16
GAP_X = 8
GAP_Y = 4
BG = (24, 24, 23)
FG = (240, 239, 236)
OUTCOME_RGB = {True: (64, 200, 140), False: (240, 96, 96)}


@dataclass(frozen=True)
class Job:
    episode: int
    kind: str  # "overview" | "zoom"
    t0: float
    t1: float  # inclusive; inf = to the end
    every: float


def tile_frames(length: int, t0: float, t1: float, every: float, add_last: bool) -> list[int]:
    """Frame indices at t0, t0+every, ... <= t1, clipped to the episode (optionally + its last frame)."""
    last = length - 1
    t1 = min(t1, last / FPS)
    n = int(math.floor((t1 - t0) / every + 1e-6)) + 1 if t1 >= t0 else 0
    frames = [min(last, max(0, int(round((t0 + i * every) * FPS)))) for i in range(n)]
    if add_last:
        frames.append(last)
    return sorted(set(frames))


def _decode(data: np.ndarray, size: int) -> Image.Image:
    img = Image.open(io.BytesIO(data.tobytes())).convert("RGB")
    return img if img.width == size else img.resize((size, size), Image.BILINEAR)


def render_job(
    job: Job, frames_dir: Path, out_dir: Path, per_row: int, rows_per_page: int, scale: float, quality: int
) -> list[str]:
    """Render one overview / zoom job of one episode -> list of written JPG paths."""
    path = frames_dir / paths.episode_frames_path(job.episode).name
    with h5py.File(path, "r") as f:
        length = int(f.attrs["length"])
        success = bool(f.attrs["success"])
        frames = tile_frames(length, job.t0, job.t1, job.every, add_last=job.kind == "overview")
        cam = int(round(IMAGE_SIZE * scale))
        tile_w, tile_h = cam * len(CAMERAS), cam + LABEL_H
        page_w = per_row * tile_w + (per_row - 1) * GAP_X
        per_page = per_row * rows_per_page
        pages = [frames[i : i + per_page] for i in range(0, len(frames), per_page)] or [[]]
        outcome = "success" if success else "failure"
        decimals = 1 if job.every >= 0.1 - 1e-9 else 2
        font, font_b = get_font(12), get_font(14, bold=True)
        if job.kind == "overview":
            stem = f"ep_{job.episode:04d}_overview"
        else:
            stem = f"ep_{job.episode:04d}_zoom_{job.t0:.1f}-{min(job.t1, (length - 1) / FPS):.1f}"
        for old in out_dir.glob(f"{stem}_p*.jpg"):
            old.unlink()
        written = []
        for p, page in enumerate(pages, start=1):
            rows = max(1, math.ceil(len(page) / per_row))
            page_h = HEADER_H + rows * tile_h + (rows - 1) * GAP_Y + GAP_Y
            sheet = Image.new("RGB", (page_w, page_h), BG)
            d = ImageDraw.Draw(sheet)
            cams = " | ".join(CAMERA_LABELS[c] for c in CAMERAS)
            span = f"{job.kind} every {job.every:g} s" + (
                "" if job.kind == "overview" else f" in [{job.t0:g}, {min(job.t1, (length - 1) / FPS):g}] s"
            )
            head = f"ep {job.episode}  ·  {outcome}  ·  T = {length / FPS:.2f} s ({length} frames)  ·  {span}"
            head += f"  ·  page {p}/{len(pages)}  ·  tile = {cams}"
            d.text((8, 4), head, font=font_b, fill=OUTCOME_RGB[success])
            for k, fi in enumerate(page):
                r, c = divmod(k, per_row)
                x, y = c * (tile_w + GAP_X), HEADER_H + r * (tile_h + GAP_Y)
                for j, cname in enumerate(CAMERAS):
                    sheet.paste(_decode(f[cname][fi], cam), (x + j * cam, y + LABEL_H))
                end = "  (last frame)" if fi == length - 1 else ""
                label = f"ep {job.episode} · t = {fi / FPS:.{decimals}f} s · {outcome}{end}"
                d.text((x + 3, y + 1), label, font=font, fill=FG)
                d.rectangle((x + tile_w - 10, y + 4, x + tile_w - 3, y + 11), fill=OUTCOME_RGB[success])
            out = out_dir / f"{stem}_p{p}.jpg"
            sheet.save(out, quality=quality)
            written.append(str(out))
    return written


def _run(args: tuple) -> list[str]:
    return render_job(*args)


def detect_stills(episodes: list[int], min_s: float, range_rad: float, frame_table: Path) -> pd.DataFrame:
    """Intervals > min_s where all 14 joints stay within range_rad over a centred 1 s window (stall candidates)."""
    t = pq.read_table(
        frame_table, columns=["episode_index", "frame_index", "state"], filters=[("episode_index", "in", episodes)]
    ).to_pandas()
    rows = []
    for e, g in t.groupby("episode_index"):
        st = pd.DataFrame(np.stack(g.sort_values("frame_index")["state"].to_numpy()))
        roll = st.rolling(FPS, center=True, min_periods=1)
        still = ((roll.max() - roll.min()).to_numpy().max(axis=1) < range_rad).astype(np.int8)
        d = np.diff(np.r_[0, still, 0])
        for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1), strict=True):
            if (b - a) / FPS > min_s:
                rows.append(
                    {
                        "episode_index": int(e),
                        "t_start_s": a / FPS,
                        "t_end_s": (b - 1) / FPS,
                        "duration_s": (b - a) / FPS,
                    }
                )
    return pd.DataFrame(rows, columns=["episode_index", "t_start_s", "t_end_s", "duration_s"])


def select_episodes(args: argparse.Namespace) -> list[int]:
    eps: set[int] = set()
    if args.episodes:
        eps |= set(parse_episodes(args.episodes))
    if args.split == "heldout":
        eps |= set(load_splits(args.splits)["heldout"])
    elif args.split == "val_train":
        f = paths.run_dir(args.eval_run) / "eval" / "val_train_step000250.parquet"
        eps |= set(pq.read_table(f, columns=["episode_index"]).column(0).unique().to_pylist())
    return sorted(int(e) for e in eps)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", help="e.g. 0-9 | 0,3,5-7")
    ap.add_argument("--split", choices=["heldout", "val_train"], help="heldout (splits.json) | val_train (step 4)")
    ap.add_argument("--eval-run", default="vf_c2229_s0", help="run whose eval/val_train_*.parquet lists val_train")
    ap.add_argument("--splits", type=Path, default=paths.SPLITS_FILE)
    ap.add_argument("--window", nargs=2, type=float, metavar=("T0", "T1"), help="zoom window [s] for --episodes")
    ap.add_argument("--zoom", action="append", default=[], metavar="EP:T0:T1", help="per-episode zoom (repeatable)")
    ap.add_argument("--every", type=float, default=None, help="seconds between tiles (default 1.0, zoom 0.2)")
    ap.add_argument("--per-row", type=int, default=4)
    ap.add_argument("--rows-per-page", type=int, default=6)
    ap.add_argument("--scale", type=float, default=0.75)
    ap.add_argument("--quality", type=int, default=85)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--frames-dir", type=Path, default=paths.FRAMES_DIR)
    ap.add_argument("--out-dir", type=Path, default=paths.RUNS_ROOT / "annotation_sheets")
    ap.add_argument("--stills", action="store_true", help="also write <out-dir>/stills.csv (stall candidates)")
    ap.add_argument("--still-min-s", type=float, default=3.0)
    ap.add_argument("--still-range-rad", type=float, default=0.04)
    ap.add_argument("--seed", type=int, default=0, help="unused (deterministic); recorded for the conventions")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    zooms = []
    for z in args.zoom:
        e, a, b = z.split(":")
        zooms.append((int(e), float(a), float(b)))
    episodes = select_episodes(args)
    if not episodes and not zooms:
        ap.error("give --episodes, --split or --zoom")
    zoom_mode = bool(args.window or zooms)
    every = args.every if args.every is not None else (0.2 if zoom_mode else 1.0)
    if args.window:
        zooms += [(e, args.window[0], args.window[1]) for e in episodes]
    jobs = (
        [Job(e, "zoom", a, b, every) for e, a, b in zooms]
        if zoom_mode
        else [Job(e, "overview", 0.0, math.inf, every) for e in episodes]
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    task = [
        (j, args.frames_dir, args.out_dir, args.per_row, args.rows_per_page, args.scale, args.quality) for j in jobs
    ]
    n_pages = 0
    if args.workers > 1 and len(task) > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for job, out in zip(jobs, ex.map(_run, task), strict=True):
                n_pages += len(out)
                log.info("ep %04d %s -> %d page(s)", job.episode, job.kind, len(out))
    else:
        for job, t in zip(jobs, task, strict=True):
            out = _run(t)
            n_pages += len(out)
            log.info("ep %04d %s -> %d page(s)", job.episode, job.kind, len(out))
    log.info("%d job(s), %d page(s) in %.1f s -> %s", len(jobs), n_pages, time.time() - t0, args.out_dir)

    if args.stills:
        done = sorted({j.episode for j in jobs})
        stills = detect_stills(done, args.still_min_s, args.still_range_rad, paths.FRAME_TABLE)
        out = args.out_dir / "stills.csv"
        if out.is_file():  # keep the stills of other episodes written earlier
            old = pd.read_csv(out)
            stills = pd.concat([old[~old["episode_index"].isin(done)], stills])
        stills.sort_values(["episode_index", "t_start_s"]).to_csv(out, index=False, float_format="%.2f")
        log.info("%d still interval(s) -> %s", len(stills), out)


if __name__ == "__main__":
    main()
