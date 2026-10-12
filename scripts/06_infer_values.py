"""Step 6.1: score every frame with the trained value model -> runs/<run>/values.parquet.

    # single model (step 5 decided: no memorisation)
    env -u PYTHONPATH uv run python scripts/06_infer_values.py --run vf_c2229_s0 --ckpt best
    # cross-fitting: model k = runs/vf_c2229_cf{k}/checkpoints/last scores the episodes of fold k
    env -u PYTHONPATH uv run python scripts/06_infer_values.py --run vf_c2229_cf --crossfit 5 --ckpt last
    # laptop smoke test: tiny randomly-initialised model on the local 10-episode frame store, CPU
    env -u PYTHONPATH uv run python scripts/06_infer_values.py --run dev_infer_tiny --tiny-random-model \
        --episodes 0-9 --device cpu --batch-size 32 --num-workers 2

Eval mode, no grad, bf16 autocast (CUDA), no augmentation, every frame (frame_stride 1), episode-ordered.
Output schema: index, episode_index, frame_index, value, value_std, model_id (+ values.json with timings).
Asserts 634 838 rows / every index once (or exactly the frames of --episodes), and with --crossfit that every
episode was scored by the model of its own fold.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import logging
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from recap_vf import paths
from recap_vf.advantage import VALUE_COLUMNS
from recap_vf.constants import NUM_EPISODES, NUM_FRAMES
from recap_vf.data.dataset import FrameDataset

logger = logging.getLogger("06_infer_values")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="output run directory (and checkpoint run unless --crossfit)")
    p.add_argument("--ckpt", default="best", help="checkpoint name under runs/<run>/checkpoints/ (best, last, ...)")
    p.add_argument("--ckpt-path", type=Path, default=None, help="explicit checkpoint directory (single model)")
    p.add_argument("--crossfit", type=int, default=0, help="K folds: model k scores fold k (0 = single model)")
    p.add_argument("--fold-run", default="{run}{k}", help="run name of fold model k (format string)")
    p.add_argument("--episodes", default="all", help="all | heldout | train | fold:K | 0-9 | 1,5,7 (comma-joined)")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--device", default="auto", help="auto | cuda | cuda:1 | cpu")
    p.add_argument("--no-bf16", action="store_true", help="disable bf16 autocast")
    p.add_argument("--splits", type=Path, default=paths.SPLITS_FILE)
    p.add_argument("--frames-dir", type=Path, default=paths.FRAMES_DIR)
    p.add_argument("--out", type=Path, default=None, help="default runs/<run>/values.parquet")
    p.add_argument("--tiny-random-model", action="store_true", help="smoke test: tiny random model, no checkpoint")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# ---------------------------------------------------------------------------------------------------------------------
def parse_episodes(spec: str, splits: dict | None, all_episodes: np.ndarray) -> list[int]:
    """Episode selection: comma-joined parts of `all`, `heldout`, `train`, `fold:K`, `A-B` (inclusive), or `N`."""
    out: set[int] = set()
    for part in (s.strip() for s in spec.split(",") if s.strip()):
        if part == "all":
            out |= set(all_episodes.tolist())
        elif part in ("heldout", "train") or part.startswith("fold:"):
            if splits is None:
                raise FileNotFoundError(f"--episodes {part} needs the splits file")
            out |= set(splits["folds"][part.split(":", 1)[1]] if part.startswith("fold:") else splits[part])
        elif "-" in part:
            a, b = (int(x) for x in part.split("-", 1))
            out |= set(range(a, b + 1))
        else:
            out.add(int(part))
    if unknown := sorted(out - set(all_episodes.tolist())):
        raise KeyError(f"unknown episodes {unknown[:10]}")
    return sorted(out)


def tiny_random_model(seed: int):
    """CPU-friendly randomly-initialised ValueModel (same tiny configs as tests/test_model.py)."""
    from transformers import Gemma3TextConfig, SiglipVisionConfig

    from recap_vf.model import ValueModel, ValueModelConfig

    torch.manual_seed(seed)
    vision = SiglipVisionConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4, image_size=224, patch_size=14
    )
    text = Gemma3TextConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=8,
        query_pre_attn_scalar=8,
        sliding_window=16,
        layer_types=["sliding_attention", "full_attention"],
        max_position_embeddings=2048,
        pad_token_id=0,
    )
    model = ValueModel.from_backbone_configs(ValueModelConfig(), vision, text)
    with torch.no_grad():  # zero head = constant -0.5; randomise it so values vary with the input
        model.head.weight.normal_(std=0.5)
        model.head.bias.normal_(std=0.5)
    return model


def _find_key(doc, key: str):
    """First value of `key` anywhere in a nested dict/list (None if absent)."""
    if isinstance(doc, dict):
        if key in doc:
            return doc[key]
        doc = list(doc.values())
    if isinstance(doc, list):
        for v in doc:
            if (found := _find_key(v, key)) is not None:
                return found
    return None


def load_model(ckpt_dir: Path, expect_fold: int | None):
    import yaml

    from recap_vf.model import CONFIG_NAME, ValueModel

    if not (ckpt_dir / CONFIG_NAME).is_file():
        raise FileNotFoundError(f"no checkpoint at {ckpt_dir} (expected {CONFIG_NAME} + model.safetensors)")
    if expect_fold is not None:  # best effort: the step-4 config may record the held-out fold of a cross-fit model
        fold = _find_key(yaml.safe_load((ckpt_dir / CONFIG_NAME).read_text()), "fold")
        if fold is not None and int(fold) != expect_fold:
            raise ValueError(f"{ckpt_dir} was trained for fold {fold}, but would score fold {expect_fold}")
    return ValueModel.from_pretrained(ckpt_dir)


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=paths.REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:  # noqa: BLE001 - informative only
        return None


def score(model, dataset: FrameDataset, model_id: str, args, device: torch.device, bf16: bool) -> pd.DataFrame:
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,  # dataset is ordered by global index = episode by episode
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    autocast = torch.autocast(device_type=device.type, dtype=torch.bfloat16) if bf16 else contextlib.nullcontext()
    model.eval().to(device)
    cols: dict[str, list[np.ndarray]] = {k: [] for k in ("index", "episode_index", "frame_index", "value", "std")}
    with torch.inference_mode(), autocast:
        for batch in tqdm(loader, desc=model_id, unit="batch", mininterval=10):
            images = {k: v.to(device, non_blocking=True) for k, v in batch["images"].items()}
            out = model.predict(images, batch["state"].to(device, non_blocking=True))
            cols["value"].append(out["value"].float().cpu().numpy())
            cols["std"].append(out["value_std"].float().cpu().numpy())
            for k in ("index", "episode_index", "frame_index"):
                cols[k].append(batch[k].numpy())
    model.to("cpu")
    cat = {k: np.concatenate(v) if v else np.zeros(0) for k, v in cols.items()}
    return pd.DataFrame(
        {
            "index": cat["index"].astype(np.int64),
            "episode_index": cat["episode_index"].astype(np.int64),
            "frame_index": cat["frame_index"].astype(np.int64),
            "value": cat["value"].astype(np.float32),
            "value_std": cat["std"].astype(np.float32),
            "model_id": model_id,
        }
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    torch.manual_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    bf16 = not args.no_bf16 and device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    episodes_df = pd.read_parquet(paths.EPISODE_TABLE)
    splits = json.loads(args.splits.read_text()) if args.splits.exists() else None
    selected = parse_episodes(args.episodes, splits, episodes_df["episode_index"].to_numpy())
    full = len(selected) == NUM_EPISODES

    # (model_id, checkpoint dir or None for tiny random, episodes, expected fold)
    groups: list[tuple[str, Path | None, list[int], int | None]] = []
    if args.crossfit:
        if splits is None:
            raise FileNotFoundError(f"--crossfit needs {args.splits}")
        if len(splits["folds"]) != args.crossfit:
            raise ValueError(f"--crossfit {args.crossfit} but splits.json has {len(splits['folds'])} folds")
        fold_of = {int(e): int(k) for k, eps in splits["folds"].items() for e in eps}
        for k in range(args.crossfit):
            eps = [e for e in selected if fold_of[e] == k]
            if not eps:
                continue
            fold_run = args.fold_run.format(run=args.run, k=k)
            if args.tiny_random_model:
                groups.append((f"tiny-random:fold{k}", None, eps, k))
            else:
                groups.append(
                    (f"{fold_run}/{args.ckpt}:fold{k}", paths.run_dir(fold_run) / "checkpoints" / args.ckpt, eps, k)
                )
    elif args.tiny_random_model:
        groups.append((f"tiny-random:seed{args.seed}", None, selected, None))
    else:
        ckpt = args.ckpt_path or paths.run_dir(args.run) / "checkpoints" / args.ckpt
        try:
            mid = str(Path(ckpt).resolve().relative_to(paths.RUNS_ROOT.resolve())).replace("/checkpoints/", "/")
        except ValueError:
            mid = str(ckpt)
        groups.append((mid, Path(ckpt), selected, None))

    logger.info(
        "%d episodes (%s), %d model(s), device %s, bf16 %s, batch %d, workers %d",
        len(selected),
        "all" if full else args.episodes,
        len(groups),
        device,
        bf16,
        args.batch_size,
        args.num_workers,
    )
    t0 = time.perf_counter()
    parts, timings = [], []
    for model_id, ckpt_dir, eps, fold in groups:
        tg = time.perf_counter()
        model = tiny_random_model(args.seed + (fold or 0)) if ckpt_dir is None else load_model(ckpt_dir, fold)
        dataset = FrameDataset(eps, frame_stride=1, frames_dir=args.frames_dir)
        t_load = time.perf_counter() - tg
        ts = time.perf_counter()
        part = score(model, dataset, model_id, args, device, bf16)
        dt_s = time.perf_counter() - ts
        parts.append(part)
        timings.append(
            {
                "model_id": model_id,
                "checkpoint": str(ckpt_dir) if ckpt_dir else None,
                "episodes": len(eps),
                "frames": len(part),
                "load_s": t_load,
                "score_s": dt_s,
                "frames_per_s": len(part) / dt_s,
            }
        )
        logger.info(
            "%s: %d frames of %d episodes in %.1f s (%.0f frames/s)",
            model_id,
            len(part),
            len(eps),
            dt_s,
            len(part) / dt_s,
        )
        del model, dataset
        if device.type == "cuda":
            torch.cuda.empty_cache()
    wall = time.perf_counter() - t0

    values = pd.concat(parts, ignore_index=True).sort_values("index", kind="stable").reset_index(drop=True)
    values = values[VALUE_COLUMNS]
    # --- assertions --------------------------------------------------------------------------------------------------
    assert values["index"].is_unique, "some frames were scored twice"
    sel = episodes_df[episodes_df["episode_index"].isin(selected)]
    expected = np.concatenate(
        [np.arange(a, b) for a, b in zip(sel["dataset_from_index"], sel["dataset_to_index"], strict=True)]
    )
    if full:
        assert len(values) == NUM_FRAMES, f"{len(values)} rows, expected {NUM_FRAMES}"
    assert np.array_equal(values["index"].to_numpy(), np.sort(expected)), "scored frames != frames of the episodes"
    assert np.isfinite(values["value"]).all(), "non-finite values"
    if args.crossfit:
        by_ep = values.groupby("episode_index")["model_id"].agg(["unique"])
        for e, (ids,) in by_ep.iterrows():
            assert len(ids) == 1 and ids[0].endswith(f"fold{fold_of[int(e)]}"), f"episode {e} scored by {ids}"
    logger.info(
        "checks passed: %d rows, every index once%s",
        len(values),
        ", every episode scored by its own fold model" if args.crossfit else "",
    )

    out = args.out or paths.run_dir(args.run) / "values.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    values.to_parquet(out, index=False)
    meta = {
        "run": args.run,
        "ckpt": args.ckpt,
        "crossfit": args.crossfit,
        "episodes": args.episodes,
        "num_episodes": len(selected),
        "num_frames": len(values),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "bf16_autocast": bf16,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "wall_time_s": wall,
        "frames_per_s": len(values) / wall,
        "groups": timings,
        "value_mean": float(values["value"].mean()),
        "value_std_mean": float(values["value_std"].mean()),
        "seed": args.seed,
        "torch": torch.__version__,
        "git_commit": git_commit(),
        "created_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    logger.info("wrote %s (%d rows) in %.1f s wall (%.0f frames/s)", out, len(values), wall, len(values) / wall)


if __name__ == "__main__":
    main()
