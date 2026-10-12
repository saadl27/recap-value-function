"""Step 1.5 — benchmark FrameDataset: JPEG decoders, then DataLoader items/s for several worker counts.

    env -u PYTHONPATH uv run python scripts/01_benchmark_dataset.py --episodes 0-9 --workers 4 8 12
Workstation (full store): uv run python scripts/01_benchmark_dataset.py --episodes all --workers 4 8 12 16
Note: on a small store the files sit in the page cache; a 40 GB store may not, so measure there too.
"""

import argparse
import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from recap_vf import paths
from recap_vf.data.dataset import FrameDataset, decode_jpeg
from recap_vf.data.frames import parse_episodes


def bench_decoders(ds: FrameDataset, n: int, rng: np.random.Generator) -> dict:
    f = ds._file(int(ds.episode_index[0]))
    raw = [f["top"][int(i)] for i in rng.integers(0, len(f["top"]), n)]
    out = {}
    for dec in ("torchvision", "pil"):
        decode_jpeg(raw[0], dec)
        t0 = time.perf_counter()
        for r in raw:
            decode_jpeg(r, dec)
        out[dec] = (time.perf_counter() - t0) / n * 1e3
    return out


def bench_loader(ds: FrameDataset, workers: int, batch_size: int, n_batches: int, warmup: int, seed: int) -> dict:
    g = torch.Generator().manual_seed(seed)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        generator=g,
        drop_last=True,
        persistent_workers=False,
        prefetch_factor=4 if workers else None,
    )
    it = iter(dl)
    t_start = time.perf_counter()
    for _ in range(warmup):
        next(it)
    t0 = time.perf_counter()
    n = 0
    for _ in range(n_batches):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl)
            batch = next(it)
        n += batch["index"].shape[0]
    dt = time.perf_counter() - t0
    del it
    return {"workers": workers, "items_per_s": n / dt, "items": n, "seconds": dt, "startup_s": t0 - t_start}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", default="all", help="episodes to sample from (must be in the store)")
    ap.add_argument("--workers", type=int, nargs="+", default=[4, 8, 12])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--batches", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--decoder", default="torchvision", choices=["torchvision", "pil"])
    ap.add_argument("--max-open-files", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import pandas as pd

    n_eps = len(pd.read_parquet(paths.EPISODE_TABLE, columns=["episode_index"]))
    eps = [e for e in parse_episodes(args.episodes, n_eps) if paths.episode_frames_path(e).is_file()]
    ds = FrameDataset(eps, decoder=args.decoder, max_open_files=args.max_open_files)
    print(f"{len(eps)} episodes, {len(ds)} items, decoder={args.decoder}, max_open_files={args.max_open_files}")
    rng = np.random.default_rng(args.seed)
    dec = bench_decoders(ds, 500, rng)
    print("decode ms/image:", {k: round(v, 3) for k, v in dec.items()})
    t0 = time.perf_counter()
    for i in rng.integers(0, len(ds), 200):
        ds[int(i)]
    single = 200 / (time.perf_counter() - t0)
    print(f"main process (0 workers): {single:.0f} items/s")
    ds.close()
    results = []
    for w in args.workers:
        r = bench_loader(ds, w, args.batch_size, args.batches, args.warmup, args.seed)
        results.append(r)
        print(f"workers={w:3d}: {r['items_per_s']:7.0f} items/s ({r['items']} items in {r['seconds']:.1f}s)")
    record = {
        "args": vars(args),
        "episodes": len(eps),
        "items": len(ds),
        "decode_ms_per_image": dec,
        "items_per_s_0_workers": single,
        "loader": results,
        "cpu_count": torch.multiprocessing.cpu_count(),
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out = paths.CACHE_DIR / "logs" / f"01_benchmark_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
