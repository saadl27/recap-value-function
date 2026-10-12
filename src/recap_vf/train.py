"""Value-function training (step 4): Monte-Carlo bin targets, cross-entropy, held-out checkpoint selection.

`TrainConfig` is loaded from YAML (`configs/train_default.yaml`) and patched from the CLI (`--set key=value`, dotted
keys for the nested model config, e.g. `--set model.freeze_vision=true`). `train()` runs the whole job:

- train set: every frame of the train episodes, uniform and shuffled by a steps-based infinite sampler; targets
  (`target_bin`, `return_norm`) joined on the global frame `index` from `returns_{tag}.parquet`;
- two val sets evaluated every `eval_every` steps: `val_heldout` (held-out episodes) and `val_train` (a fixed random
  subset of train episodes), both at `val_frame_stride`; their gap is step 5's memorisation signal;
- `runs/<run>/`: `config.yaml` (resolved), `train.log`, `train_log.jsonl`, `metrics.jsonl` (one line per eval),
  `tb/`, `plots/`, `eval/<set>_step<N>.parquet` (per-frame predictions), `checkpoints/{best,last}/`
  (`best` = min val_heldout CE; `last` also holds `trainer_state.pt` for `--resume`).
- early stopping after `early_stop_patience` evals without a val_heldout CE improvement.
See docs/steps/04_training.md.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import math
import os
import platform
import random
import shutil
import subprocess
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Sampler

from recap_vf import constants, paths
from recap_vf.metrics import FAIL_DELTA, compute_metrics, summarize_logits

log = logging.getLogger(__name__)

TRAINER_STATE = "trainer_state.pt"
VAL_SETS = ("val_heldout", "val_train")


# ======================================================================================================================
# Config
# ======================================================================================================================
def _default_model() -> dict[str, Any]:
    return {"language_backbone": constants.LANGUAGE_BACKBONE_MIRROR, "gradient_checkpointing": True}


@dataclass
class TrainConfig:
    run_name: str = "vf_c2229_s0"
    seed: int = 0
    # --- data ---------------------------------------------------------------------------------------------------
    returns_tag: str = "c2229"
    splits_file: str | None = None  # None -> paths.SPLITS_FILE
    frames_dir: str | None = None  # None -> paths.FRAMES_DIR
    frame_table: str | None = None  # None -> paths.FRAME_TABLE
    returns_dir: str | None = None  # None -> paths.CACHE_DIR (via returns.load_returns)
    stats_file: str | None = None  # None -> <raw>/meta/stats.json (state q01/q99)
    train_episodes: Any = None  # None -> splits["train"]; or "1-9", "1,3,5", [1, 2]
    val_heldout_episodes: Any = None  # None -> splits["heldout"]
    val_train_num_episodes: int = 120
    val_train_seed: int = 0  # fixed independently of `seed`, so runs with different seeds share val_train
    val_frame_stride: int = 10
    num_workers: int = 12
    eval_num_workers: int = 8
    prefetch_factor: int = 4
    mp_context: str = "forkserver"  # fork | forkserver | spawn (fork warns: torch is multi-threaded)
    # --- model (ValueModelConfig fields) ------------------------------------------------------------------------
    model: dict[str, Any] = field(default_factory=_default_model)
    # --- optimisation -------------------------------------------------------------------------------------------
    steps: int = 2100  # cap on optimizer steps; early stopping may end the run sooner
    global_batch: int = 256
    micro_batch: int = 32
    lr_backbone: float = 5e-5
    lr_new: float = 1e-4
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    warmup_steps: int = 200  # linear warm-up, then constant
    grad_clip: float = 1.0
    precision: str = "bf16"  # bf16 (autocast, fp32 master weights) | fp32
    tf32: bool = True
    # --- eval / logging -----------------------------------------------------------------------------------------
    eval_every: int = 250  # also the checkpoint interval
    eval_batch: int = 64
    early_stop_patience: int = 3  # evals without val_heldout CE improvement before stopping; 0 disables
    log_every: int = 10
    plot_heldout_per_class: int = 4
    plot_train_episodes: int = 4

    @property
    def grad_accum(self) -> int:
        return self.global_batch // self.micro_batch

    def validate(self) -> None:
        from recap_vf.model import ValueModelConfig

        if self.global_batch % self.micro_batch:
            raise ValueError(f"global_batch {self.global_batch} must be a multiple of micro_batch {self.micro_batch}")
        known = {f.name for f in dataclasses.fields(ValueModelConfig)}
        if unknown := set(self.model) - known:
            raise ValueError(f"unknown model config keys {sorted(unknown)}; known: {sorted(known)}")
        if self.precision not in ("bf16", "fp32"):
            raise ValueError(f"precision={self.precision!r}, expected bf16 | fp32")
        if self.mp_context not in ("fork", "forkserver", "spawn"):
            raise ValueError(f"mp_context={self.mp_context!r}, expected fork | forkserver | spawn")
        if self.steps < 1 or self.eval_every < 1 or self.log_every < 1:
            raise ValueError("steps, eval_every and log_every must be >= 1")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrainConfig:
        known = {f.name for f in dataclasses.fields(cls)}
        if unknown := set(d) - known:
            raise ValueError(f"unknown train config keys: {sorted(unknown)}")
        d = dict(d)
        if "betas" in d:
            d["betas"] = tuple(d["betas"])
        if "model" in d:
            d["model"] = {**_default_model(), **(d["model"] or {})}
        return cls(**d)

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(dataclasses.asdict(self)))  # tuples -> lists, YAML-safe


def apply_overrides(d: dict[str, Any], overrides: Sequence[str]) -> dict[str, Any]:
    """Apply `key=value` strings (value parsed as YAML; dotted keys address nested dicts) to a config dict."""
    d = json.loads(json.dumps(d))
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override {item!r} is not key=value")
        key, raw = item.split("=", 1)
        node = d
        *parents, leaf = key.strip().split(".")
        for p in parents:
            node = node.setdefault(p, {})
            if not isinstance(node, dict):
                raise ValueError(f"cannot set {key}: {p} is not a mapping")
        node[leaf] = yaml.safe_load(raw)
    return d


def load_config(path: Path | str | None, overrides: Sequence[str] = ()) -> TrainConfig:
    d = yaml.safe_load(Path(path).read_text()) if path else {}
    cfg = TrainConfig.from_dict(apply_overrides(d or {}, overrides))
    cfg.validate()
    return cfg


def resolve_config(
    config_path: Path | str | None, overrides: Sequence[str] = (), resume: bool = False, run_name: str | None = None
) -> TrainConfig:
    """Config for a new run (YAML + overrides) or, with `resume`, the run's saved config + the same overrides."""
    overrides = list(overrides) + ([f"run_name={run_name}"] if run_name else [])
    cfg = load_config(config_path, overrides)
    if resume:
        saved = paths.run_dir(cfg.run_name) / "config.yaml"
        if not saved.is_file():
            raise FileNotFoundError(f"--resume: no {saved}")
        cfg = TrainConfig.from_dict(apply_overrides(yaml.safe_load(saved.read_text())["train"], overrides))
        cfg.validate()
    return cfg


def parse_episodes(spec: Any) -> list[int] | None:
    """None | int | list | "1-9,12" -> sorted unique episode list (None stays None)."""
    if spec is None:
        return None
    if isinstance(spec, bool):
        raise ValueError(f"bad episode spec {spec!r}")
    if isinstance(spec, int):
        return [spec]
    if isinstance(spec, list | tuple):
        return sorted({int(x) for x in spec})
    out: set[int] = set()
    for part in str(spec).split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        elif part:
            out.add(int(part))
    return sorted(out)


# ======================================================================================================================
# Data
# ======================================================================================================================
@dataclass
class TargetTable:
    """Returns-table columns as lookup arrays indexed by the global frame `index` (join on index, not row order)."""

    episode_index: np.ndarray
    frame_index: np.ndarray
    target_bin: np.ndarray
    return_norm: np.ndarray
    episode_length: np.ndarray
    success: np.ndarray

    @classmethod
    def from_returns(cls, table: pd.DataFrame) -> TargetTable:
        idx = table["index"].to_numpy(np.int64)
        if len(np.unique(idx)) != len(idx):
            raise ValueError("returns table has duplicate `index` values")
        n = int(idx.max()) + 1

        def lut(col: str, dtype, fill) -> np.ndarray:
            a = np.full(n, fill, dtype=dtype)
            a[idx] = table[col].to_numpy(dtype)
            return a

        return cls(
            episode_index=lut("episode_index", np.int64, -1),
            frame_index=lut("frame_index", np.int64, -1),
            target_bin=lut("bin_index", np.int64, -1),
            return_norm=lut("return_norm", np.float32, np.nan),
            episode_length=lut("episode_length", np.int64, -1),
            success=lut("success", bool, False),
        )

    def check_covers(self, index: np.ndarray, episode_index: np.ndarray, frame_index: np.ndarray) -> None:
        """Every dataset frame has a target and agrees on (episode_index, frame_index)."""
        index = np.asarray(index, dtype=np.int64)
        if len(index) and (index.min() < 0 or index.max() >= len(self.target_bin)):
            raise KeyError("dataset frame index outside the returns table")
        if (self.target_bin[index] < 0).any():
            raise KeyError("some dataset frames have no returns-table row")
        if not np.array_equal(self.episode_index[index], episode_index) or not np.array_equal(
            self.frame_index[index], frame_index
        ):
            raise ValueError("returns table and frame dataset disagree on (episode_index, frame_index) for an index")


class TargetDataset(Dataset):
    """FrameDataset item + `target_bin` / `return_norm` looked up by the item's global `index`."""

    def __init__(self, base: Dataset, targets: TargetTable) -> None:
        targets.check_covers(base.index, base.episode_index, base.frame_index)
        self.base = base
        self.target_bin = targets.target_bin  # only the two arrays the workers need
        self.return_norm = targets.return_norm

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int) -> dict:
        item = self.base[i]
        k = int(item["index"])
        item["target_bin"] = torch.tensor(self.target_bin[k], dtype=torch.int64)
        item["return_norm"] = torch.tensor(self.return_norm[k], dtype=torch.float32)
        return item


class InfiniteSampler(Sampler[int]):
    """Endless stream of dataset positions, one fresh seeded permutation per pass.

    Fully determined by (seed, start), so a resumed run continues the same stream with `start = samples_seen`.
    """

    def __init__(self, n: int, seed: int, start: int = 0) -> None:
        if n < 1:
            raise ValueError("empty dataset")
        self.n, self.seed, self.start = n, int(seed), int(start)

    def __iter__(self) -> Iterator[int]:
        k, off = divmod(self.start, self.n)
        while True:
            g = torch.Generator().manual_seed(self.seed * 1_000_003 + k)
            yield from torch.randperm(self.n, generator=g)[off:].tolist()
            k, off = k + 1, 0

    def __len__(self) -> int:  # nominal; the stream is infinite
        return self.n


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_info() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=paths.REPO_ROOT, capture_output=True, text=True, timeout=10).stdout

    try:
        commit = run("rev-parse", "HEAD").strip()
        dirty = bool(run("status", "--porcelain", "--untracked-files=no").strip())
        return {"commit": commit or None, "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


@dataclass
class DataBundle:
    train: TargetDataset
    val: dict[str, TargetDataset]
    targets: TargetTable
    returns_meta: dict
    splits_file: Path
    train_episodes: list[int]
    val_episodes: dict[str, list[int]]
    plot_episodes: dict[str, list[int]]


def _pick(rng: np.random.Generator, pool: Sequence[int], k: int) -> list[int]:
    pool = sorted(pool)
    k = min(k, len(pool))
    return sorted(int(x) for x in rng.choice(pool, size=k, replace=False)) if k else []


def build_data(cfg: TrainConfig) -> DataBundle:
    from recap_vf.data.dataset import FrameDataset
    from recap_vf.returns import load_returns

    splits_file = Path(cfg.splits_file) if cfg.splits_file else paths.SPLITS_FILE
    splits = json.loads(splits_file.read_text())
    heldout_all = {int(e) for e in splits["heldout"]}
    train_eps = parse_episodes(cfg.train_episodes) or sorted(int(e) for e in splits["train"])
    val_eps = parse_episodes(cfg.val_heldout_episodes)
    val_eps = sorted(heldout_all) if val_eps is None else val_eps
    # held-out episodes must never reach the train sampler
    if leak := sorted(set(train_eps) & heldout_all):
        raise ValueError(f"train episodes include held-out episodes: {leak[:10]}")
    if leak := sorted(set(train_eps) & set(val_eps)):
        raise ValueError(f"train and val_heldout episodes overlap: {leak[:10]}")

    if cfg.returns_dir:
        rdir = Path(cfg.returns_dir)
        table = pd.read_parquet(rdir / f"returns_{cfg.returns_tag}.parquet")
        meta = json.loads((rdir / f"returns_{cfg.returns_tag}.json").read_text())
    else:
        table, meta = load_returns(cfg.returns_tag)
    targets = TargetTable.from_returns(table)

    ds_kwargs = {
        "frames_dir": cfg.frames_dir or paths.FRAMES_DIR,
        "frame_table": cfg.frame_table or paths.FRAME_TABLE,
    }
    val_train_eps = _pick(np.random.default_rng(cfg.val_train_seed), train_eps, cfg.val_train_num_episodes)
    train = TargetDataset(FrameDataset(train_eps, **ds_kwargs), targets)
    val = {
        name: TargetDataset(FrameDataset(eps, frame_stride=cfg.val_frame_stride, **ds_kwargs), targets)
        for name, eps in (("val_heldout", val_eps), ("val_train", val_train_eps))
    }
    if set(train.base.index.tolist()) & set(val["val_heldout"].base.index.tolist()):
        raise AssertionError("val_heldout frames found in the train set")

    ep_success = {int(e): bool(s) for e, s in table.groupby("episode_index")["success"].first().items()}
    prng = np.random.default_rng(cfg.val_train_seed + 1)
    n = cfg.plot_heldout_per_class
    plot_heldout = _pick(prng, [e for e in val_eps if ep_success[e]], n) + _pick(
        prng, [e for e in val_eps if not ep_success[e]], n
    )
    half = cfg.plot_train_episodes // 2
    plot_train = _pick(prng, [e for e in val_train_eps if ep_success[e]], cfg.plot_train_episodes - half) + _pick(
        prng, [e for e in val_train_eps if not ep_success[e]], half
    )
    return DataBundle(
        train=train,
        val=val,
        targets=targets,
        returns_meta=meta,
        splits_file=splits_file,
        train_episodes=train_eps,
        val_episodes={"val_heldout": val_eps, "val_train": val_train_eps},
        plot_episodes={"val_heldout": plot_heldout, "val_train": plot_train},
    )


def make_loader(
    ds: Dataset,
    batch_size: int,
    num_workers: int,
    mp_context: str,
    *,
    sampler: Sampler | None = None,
    persistent: bool = False,
    prefetch_factor: int = 4,
    pin_memory: bool = False,
) -> DataLoader:
    kwargs: dict[str, Any] = {}
    if num_workers > 0:
        kwargs = {"multiprocessing_context": mp_context, "persistent_workers": persistent}
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(
        ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=pin_memory, **kwargs
    )


def to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, dict):
            out[k] = {kk: vv.to(device, non_blocking=True) for kk, vv in v.items()}
        else:
            out[k] = v.to(device, non_blocking=True) if isinstance(v, Tensor) else v
    return out


# ======================================================================================================================
# Model / optimisation
# ======================================================================================================================
def build_model(cfg: TrainConfig) -> torch.nn.Module:
    """ValueModel from pretrained backbones (HF cache) with the dataset's state statistics."""
    from recap_vf.model import ValueModel, ValueModelConfig, load_state_stats

    stats = Path(cfg.stats_file) if cfg.stats_file else paths.RAW_DIR / "meta" / "stats.json"
    q01, q99 = load_state_stats(stats)
    return ValueModel(ValueModelConfig.from_dict(cfg.model), q01, q99)


def lr_lambda(warmup_steps: int) -> Callable[[int], float]:
    """Linear warm-up over `warmup_steps`, then constant (RLinf's schedule; lets a run be extended by resuming)."""

    def f(step: int) -> float:
        return (step + 1) / warmup_steps if step < warmup_steps else 1.0

    return f


def _backbone_lr(opt: torch.optim.Optimizer) -> float:
    """Current lr of the pretrained-weights groups (falls back to the first group if the backbones are frozen)."""
    return next(
        (g["lr"] for g in opt.param_groups if g.get("name", "").startswith("backbone")), opt.param_groups[0]["lr"]
    )


# ======================================================================================================================
# GPU utilisation monitor
# ======================================================================================================================
class GpuMonitor(threading.Thread):
    """Polls `nvidia-smi` for this GPU's utilisation (no extra dependency)."""

    def __init__(self, interval: float = 2.0) -> None:
        super().__init__(daemon=True)
        self.interval = interval
        self.samples: list[tuple[float, float]] = []  # (time, util %)
        self._stop_evt = threading.Event()
        vis = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
        self.gpu = vis if vis.isdigit() else str(torch.cuda.current_device())
        self.ok = shutil.which("nvidia-smi") is not None

    def run(self) -> None:
        cmd = ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits", "-i", self.gpu]
        while self.ok and not self._stop_evt.is_set():
            try:
                out = subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout
                self.samples.append((time.time(), float(out.strip())))
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
            self._stop_evt.wait(self.interval)

    def stop(self) -> None:
        self._stop_evt.set()

    def mean_util(self, since: float = 0.0) -> float:
        u = [s[1] for s in self.samples if s[0] >= since]
        return float(np.mean(u)) if u else float("nan")


# ======================================================================================================================
# Evaluation, plots, checkpoints
# ======================================================================================================================
@torch.no_grad()
def predict_dataset(
    model: torch.nn.Module, loader: DataLoader, device: torch.device, amp_dtype: torch.dtype | None
) -> dict[str, np.ndarray]:
    """Eval-mode logits for a whole loader: {"index", "episode_index", "frame_index", "logits"}."""
    was_training = model.training
    model.eval()
    out: dict[str, list] = {"index": [], "episode_index": [], "frame_index": [], "logits": []}
    try:
        for batch in loader:
            b = to_device(batch, device)
            with torch.autocast(device.type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
                logits = model(b["images"], b["state"])
            out["logits"].append(logits.float().cpu().numpy())
            for k in ("index", "episode_index", "frame_index"):
                out[k].append(batch[k].numpy())
    finally:
        model.train(was_training)
    return {k: np.concatenate(v) for k, v in out.items()}


def evaluate_predictions(
    pred: dict[str, np.ndarray], targets: TargetTable, fail_value: float
) -> tuple[dict[str, float], pd.DataFrame]:
    """Metrics + a per-frame prediction table for one val set."""
    idx = pred["index"].astype(np.int64)
    tb, rn = targets.target_bin[idx], targets.return_norm[idx]
    length, succ = targets.episode_length[idx], targets.success[idx]
    m = compute_metrics(
        pred["logits"], tb, rn, pred["episode_index"], pred["frame_index"], length, succ, fail_value=fail_value
    )
    s = summarize_logits(pred["logits"], tb)
    df = pd.DataFrame(
        {
            "index": idx,
            "episode_index": pred["episode_index"].astype(np.int64),
            "frame_index": pred["frame_index"].astype(np.int64),
            "value": s["value"].astype(np.float32),
            "value_std": s["value_std"].astype(np.float32),
            "ce": s["ce"].astype(np.float32),
            "return_norm": rn,
            "target_bin": tb,
            "success": succ,
        }
    )
    return m, df


def plot_vcurves(
    preds: dict[str, pd.DataFrame],
    plot_episodes: dict[str, list[int]],
    targets: TargetTable,
    fail_value: float,
    step: int,
    out_path: Path,
):
    """E[V] (line, val frames, ±1 std band) and the true return_norm (dashed, every frame) vs time."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [(name, e) for name in VAL_SETS for e in plot_episodes.get(name, [])]
    if not panels:
        return None
    ncol = 4
    nrow = math.ceil(len(panels) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2.8 * nrow), squeeze=False, sharey=True)
    for ax in axes.flat[len(panels) :]:
        ax.axis("off")
    for ax, (name, e) in zip(axes.flat, panels, strict=False):
        rows = np.flatnonzero(targets.episode_index == e)
        rows = rows[np.argsort(targets.frame_index[rows])]
        succ = bool(targets.success[rows[0]])
        color = "tab:blue" if succ else "tab:orange"
        t_true = targets.frame_index[rows] / constants.FPS
        ax.plot(t_true, targets.return_norm[rows], "--", color="0.3", lw=1, label="true g_t")
        p = preds[name]
        p = p[p["episode_index"] == e].sort_values("frame_index")
        t = p["frame_index"].to_numpy() / constants.FPS
        v, sd = p["value"].to_numpy(), p["value_std"].to_numpy()
        ax.plot(t, v, color=color, lw=1.5, label="E[V]")
        ax.fill_between(t, v - sd, v + sd, color=color, alpha=0.15, lw=0)
        ax.axhline(fail_value + FAIL_DELTA, color="tab:red", lw=0.7, ls=":")
        kind = "held-out" if name == "val_heldout" else "train"
        outcome = "success" if succ else "failure"
        ax.set_title(f"{kind} ep {e} · {outcome} · T={int(targets.episode_length[rows[0]])}", fontsize=9)
        ax.set_ylim(-1.02, 0.02)
        ax.grid(alpha=0.3)
    for ax in axes[:, 0]:
        ax.set_ylabel("value (normalised)")
    for ax in axes[-1, :]:
        ax.set_xlabel("time [s]")
    axes.flat[0].legend(loc="lower right", fontsize=8)
    fig.suptitle(f"V curves at step {step} (red dotted: failure-confidence threshold)", fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=90)
    return fig


def plot_curves(run_dir: Path) -> None:
    """Train CE (train_log.jsonl) and val CE / MAE (metrics.jsonl) vs step -> plots/curves.png."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train = _read_jsonl(run_dir / "train_log.jsonl")
    evals = _read_jsonl(run_dir / "metrics.jsonl")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    if train:
        axes[0].plot([r["step"] for r in train], [r["train_ce"] for r in train], color="0.6", lw=0.8, label="train")
    for name, color in (("val_heldout", "tab:red"), ("val_train", "tab:green")):
        for ax, key in ((axes[0], "ce"), (axes[1], "mae")):
            pts = [(r["step"], r[name][key]) for r in evals if name in r]
            if pts:
                ax.plot(*zip(*pts, strict=True), "o-", color=color, label=name)
    axes[0].set(xlabel="step", ylabel="cross-entropy", title="CE")
    axes[1].set(xlabel="step", ylabel="MAE |E[V] - g|", title="MAE")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)
    fig.savefig(run_dir / "plots" / "curves.png", dpi=90)
    plt.close(fig)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _append_jsonl(path: Path, row: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def save_checkpoint(
    ckpt_dir: Path, model: torch.nn.Module, extra: dict[str, Any], trainer_state: dict | None = None
) -> None:
    """Write `<dir>.tmp`, then swap it in, so a crash never leaves a half-written checkpoint."""
    tmp = ckpt_dir.with_name(ckpt_dir.name + ".tmp")
    old = ckpt_dir.with_name(ckpt_dir.name + ".old")
    for d in (tmp, old):
        if d.exists():
            shutil.rmtree(d)
    model.save_pretrained(tmp, extra=extra)
    if trainer_state is not None:
        torch.save(trainer_state, tmp / TRAINER_STATE)
    if ckpt_dir.exists():
        ckpt_dir.rename(old)
    tmp.rename(ckpt_dir)
    if old.exists():
        shutil.rmtree(old)


# ======================================================================================================================
# Training loop
# ======================================================================================================================
def train(
    cfg: TrainConfig,
    *,
    resume: bool = False,
    model_factory: Callable[[TrainConfig], torch.nn.Module] | None = None,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Run (or resume from `checkpoints/last`) a training job; returns a summary dict (also in summary.json)."""
    cfg.validate()
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    run_dir = paths.run_dir(cfg.run_name)
    if not resume and (run_dir / "checkpoints").exists():
        raise FileExistsError(f"{run_dir} already has checkpoints; pass --resume or choose another run_name")
    run_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(run_dir / "train.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.addHandler(fh)
    if root.level > logging.INFO:
        root.setLevel(logging.INFO)
    try:
        return _train(cfg, run_dir, device, resume, model_factory or build_model)
    finally:
        root.removeHandler(fh)
        fh.close()


def _train(
    cfg: TrainConfig,
    run_dir: Path,
    device: torch.device,
    resume: bool,
    model_factory: Callable[[TrainConfig], torch.nn.Module],
) -> dict[str, Any]:
    t_start = time.time()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    cuda = device.type == "cuda"
    if cuda and cfg.tf32:
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.allow_tf32 = True
    amp_dtype = torch.bfloat16 if cfg.precision == "bf16" else None

    # --- data -------------------------------------------------------------------------------------------------------
    data = build_data(cfg)
    meta = data.returns_meta
    fail_value = -float(meta["c_fail"]) / float(meta["Z"])
    log.info(
        "train: %d eps / %d frames | val_heldout: %d eps / %d frames | val_train: %d eps / %d frames",
        len(data.train_episodes),
        len(data.train),
        len(data.val_episodes["val_heldout"]),
        len(data.val["val_heldout"]),
        len(data.val_episodes["val_train"]),
        len(data.val["val_train"]),
    )

    # --- model / optimiser ------------------------------------------------------------------------------------------
    model = model_factory(cfg).to(device)
    counts = model.parameter_counts()
    log.info("parameters: %s", counts["all"])
    groups = model.param_groups(cfg.lr_backbone, cfg.lr_new, cfg.weight_decay)
    opt = torch.optim.AdamW(groups, betas=tuple(cfg.betas), eps=cfg.eps, fused=cuda)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(cfg.warmup_steps))

    step, best_ce, best_step, bad_evals = 0, math.inf, -1, 0
    ckpt_root = run_dir / "checkpoints"
    if resume:
        from safetensors.torch import load_file

        last = ckpt_root / "last"
        state = torch.load(last / TRAINER_STATE, map_location="cpu", weights_only=False)
        model.load_state_dict(load_file(str(last / "model.safetensors")), strict=True)
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        step, best_ce, best_step, bad_evals = state["step"], state["best_ce"], state["best_step"], state["bad_evals"]
        torch.set_rng_state(state["rng"]["torch"])
        np.random.set_state(state["rng"]["numpy"])
        random.setstate(state["rng"]["python"])
        if cuda and state["rng"]["cuda"] is not None:
            torch.cuda.set_rng_state(state["rng"]["cuda"], device)
        for name in ("metrics.jsonl", "train_log.jsonl"):  # drop rows logged after the checkpoint
            rows = [r for r in _read_jsonl(run_dir / name) if r["step"] <= step]
            (run_dir / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
        log.info("resumed from %s at step %d (best val_heldout CE %.4f @ %d)", last, step, best_ce, best_step)
        del state

    resolved = {
        "train": cfg.to_dict(),
        "resolved": {
            "returns": meta,
            "fail_value": fail_value,
            "splits_file": str(data.splits_file),
            "splits_sha256": file_sha256(data.splits_file),
            "git": git_info(),
            "train_episodes": len(data.train_episodes),
            "train_frames": len(data.train),
            "val_episodes": data.val_episodes,
            "val_frames": {k: len(v) for k, v in data.val.items()},
            "plot_episodes": data.plot_episodes,
            "grad_accum": cfg.grad_accum,
            "parameter_counts": counts,
            "host": platform.node(),
            "device": torch.cuda.get_device_name(device) if cuda else "cpu",
            "torch": torch.__version__,
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "resumed_at_step": step if resume else None,
        },
    }
    resolved = json.loads(json.dumps(resolved, default=str))  # plain YAML-safe types (e.g. torch.__version__)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False))

    # --- loaders / logging ------------------------------------------------------------------------------------------
    sampler = InfiniteSampler(len(data.train), cfg.seed, start=step * cfg.global_batch)
    train_loader = make_loader(
        data.train,
        cfg.micro_batch,
        cfg.num_workers,
        cfg.mp_context,
        sampler=sampler,
        persistent=True,
        prefetch_factor=cfg.prefetch_factor,
        pin_memory=cuda,
    )
    train_iter = iter(train_loader)
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(str(run_dir / "tb"), purge_step=step if resume else None)
    monitor = GpuMonitor() if cuda else None
    if monitor is not None:
        monitor.start()

    def run_eval(at_step: int) -> tuple[dict[str, dict[str, float]], float]:
        t0 = time.time()
        results, preds = {}, {}
        for name, ds in data.val.items():
            loader = make_loader(ds, cfg.eval_batch, cfg.eval_num_workers, cfg.mp_context, pin_memory=cuda)
            results[name], preds[name] = evaluate_predictions(
                predict_dataset(model, loader, device, amp_dtype), data.targets, fail_value
            )
            (run_dir / "eval").mkdir(exist_ok=True)
            preds[name].to_parquet(run_dir / "eval" / f"{name}_step{at_step:06d}.parquet", index=False)
        seconds = time.time() - t0
        png = run_dir / "plots" / f"vcurves_{at_step:06d}.png"
        fig = plot_vcurves(preds, data.plot_episodes, data.targets, fail_value, at_step, png)
        for name, m in results.items():
            for k, v in m.items():
                if not k.startswith("num_") and math.isfinite(v):
                    writer.add_scalar(f"{name}/{k}", v, at_step)
        writer.add_scalar("eval/seconds", seconds, at_step)
        if fig is not None:
            writer.add_figure("vcurves", fig, at_step, close=True)
        return results, seconds

    def trainer_state() -> dict:
        return {
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "step": step,
            "best_ce": best_ce,
            "best_step": best_step,
            "bad_evals": bad_evals,
            "rng": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state(device) if cuda else None,
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
        }

    # --- loop -------------------------------------------------------------------------------------------------------
    model.train()
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    accum = cfg.grad_accum
    log_t0, data_wait, log_steps = time.time(), 0.0, 0
    ce_acc = torch.zeros((), device=device)
    interval_ce: list[float] = []
    early_stopped = False
    try:
        while step < cfg.steps:
            for _ in range(accum):
                t_d = time.time()
                batch = next(train_iter)
                data_wait += time.time() - t_d
                b = to_device(batch, device)
                with torch.autocast(device.type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
                    logits = model(b["images"], b["state"])
                loss = F.cross_entropy(logits, b["target_bin"])
                (loss / accum).backward()
                ce_acc += loss.detach() / accum
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            log_steps += 1

            if step % cfg.log_every == 0 or step == cfg.steps:
                ce = ce_acc.item() / log_steps
                gn = float(grad_norm)
                if not (math.isfinite(ce) and math.isfinite(gn)):
                    raise FloatingPointError(f"non-finite loss/grad norm at step {step}: ce={ce} grad_norm={gn}")
                dt = time.time() - log_t0
                rec = {
                    "step": step,
                    "train_ce": ce,
                    "grad_norm": gn,
                    "lr_backbone": _backbone_lr(opt),
                    "samples_per_s": log_steps * cfg.global_batch / dt,
                    "data_wait_frac": data_wait / dt,
                    "gpu_util": monitor.mean_util(log_t0) if monitor else float("nan"),
                    "max_mem_gib": torch.cuda.max_memory_allocated(device) / 2**30 if cuda else 0.0,
                    "elapsed_h": (time.time() - t_start) / 3600,
                }
                log.info(
                    "step %d/%d ce %.4f gn %.2f lr %.2e | %.1f samples/s, data wait %.0f%%, gpu %.0f%%, "
                    "max mem %.1f GiB | ETA %.2f h excl. evals",
                    step,
                    cfg.steps,
                    ce,
                    gn,
                    rec["lr_backbone"],
                    rec["samples_per_s"],
                    100 * rec["data_wait_frac"],
                    rec["gpu_util"],
                    rec["max_mem_gib"],
                    (cfg.steps - step) * dt / log_steps / 3600,
                )
                for k, v in rec.items():
                    if k != "step" and math.isfinite(v):
                        writer.add_scalar(f"train/{k}", v, step)
                _append_jsonl(run_dir / "train_log.jsonl", rec)
                interval_ce.append(ce)
                ce_acc.zero_()
                log_t0, data_wait, log_steps = time.time(), 0.0, 0

            if step % cfg.eval_every == 0 or step == cfg.steps:
                results, seconds = run_eval(step)
                held_ce = results["val_heldout"]["ce"]
                improved = held_ce < best_ce
                if improved:
                    best_ce, best_step, bad_evals = held_ce, step, 0
                else:
                    bad_evals += 1
                row = {
                    "step": step,
                    "samples_seen": step * cfg.global_batch,
                    "elapsed_h": (time.time() - t_start) / 3600,
                    "train_ce": float(np.mean(interval_ce)) if interval_ce else float("nan"),
                    "eval_seconds": seconds,
                    "best_step": best_step,
                    "best_ce": best_ce,
                    **results,
                }
                _append_jsonl(run_dir / "metrics.jsonl", row)
                interval_ce = []
                h, t = results["val_heldout"], results["val_train"]
                log.info(
                    "eval step %d (%.0f s): heldout ce %.4f mae %.4f rho %.3f auc@0.5 %.3f | val_train ce %.4f "
                    "mae %.4f | best %.4f @ %d",
                    step,
                    seconds,
                    h["ce"],
                    h["mae"],
                    h["spearman"],
                    h["auc@0.5"],
                    t["ce"],
                    t["mae"],
                    best_ce,
                    best_step,
                )
                extra = {**resolved, "checkpoint": {"step": step, "val_heldout_ce": held_ce}}
                if improved:
                    save_checkpoint(ckpt_root / "best", model, extra)
                save_checkpoint(ckpt_root / "last", model, extra, trainer_state())
                plot_curves(run_dir)
                if cfg.early_stop_patience > 0 and bad_evals >= cfg.early_stop_patience:
                    log.info("early stopping: no val_heldout CE improvement for %d evals", bad_evals)
                    early_stopped = True
                    break
    finally:
        if monitor is not None:
            monitor.stop()
        writer.close()
        del train_iter

    summary = {
        "run_name": cfg.run_name,
        "step": step,
        "best_step": best_step,
        "best_val_heldout_ce": best_ce,
        "early_stopped": early_stopped,
        "wall_h": (time.time() - t_start) / 3600,
        "peak_mem_gib": torch.cuda.max_memory_allocated(device) / 2**30 if cuda else 0.0,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    log.info("done: %s", json.dumps(summary))
    return summary
