"""Step 4 training loop on CPU: config, target join, sampler, and a tiny end-to-end run with resume."""

import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import yaml
from transformers import Gemma3TextConfig, SiglipVisionConfig

from recap_vf import paths
from recap_vf.constants import CAMERAS, IMAGE_SIZE, STATE_DIM
from recap_vf.data.dataset import FrameDataset
from recap_vf.data.frames import encode_jpeg, write_episode_h5
from recap_vf.model import ValueModel, ValueModelConfig
from recap_vf.returns import compute_returns_table
from recap_vf.train import (
    InfiniteSampler,
    TargetDataset,
    TargetTable,
    TrainConfig,
    apply_overrides,
    load_config,
    lr_lambda,
    parse_episodes,
    resolve_config,
    train,
)

LENGTHS = [12, 9, 14, 10, 11, 8]  # episodes 0..5
SUCCESS = [True, False, True, False, True, False]
HELDOUT = [0, 1]
TRAIN = [2, 3, 4, 5]


# ---------------------------------------------------------------------------------------------------------------------
def test_default_config_loads_and_overrides():
    cfg = load_config("configs/train_default.yaml")
    assert cfg.run_name == "vf_c2229_s0" and cfg.steps == 2100 and cfg.grad_accum == 8
    assert cfg.model["language_backbone"] == "unsloth/gemma-3-270m" and cfg.model["freeze_vision"] is False
    assert cfg.warmup_steps == 200 and cfg.eval_every == 250 and cfg.early_stop_patience == 3
    cfg = load_config("configs/train_default.yaml", ["steps=20", "model.freeze_vision=true", "train_episodes=1-9"])
    assert cfg.steps == 20 and cfg.model["freeze_vision"] is True and cfg.train_episodes == "1-9"
    with pytest.raises(ValueError, match="unknown train config"):
        load_config(None, ["nope=1"])
    with pytest.raises(ValueError, match="unknown model config"):
        load_config(None, ["model.nope=1"])
    with pytest.raises(ValueError, match="multiple"):
        load_config(None, ["micro_batch=48"])


def test_apply_overrides_and_parse_episodes():
    d = apply_overrides({"a": 1, "m": {"x": 1}}, ["a=2.5", "m.y=[1, 2]", "b=null"])
    assert d == {"a": 2.5, "m": {"x": 1, "y": [1, 2]}, "b": None}
    assert parse_episodes("1-3,7") == [1, 2, 3, 7]
    assert parse_episodes(0) == [0] and parse_episodes([3, 1, 3]) == [1, 3] and parse_episodes(None) is None


def test_lr_schedule_warmup_then_constant():
    f = lr_lambda(200)
    assert f(0) == pytest.approx(1 / 200) and f(99) == pytest.approx(0.5) and f(199) == 1.0
    assert f(200) == 1.0 and f(5000) == 1.0


def test_sampler_is_deterministic_and_resumable():
    n = 7
    full = InfiniteSampler(n, seed=3)
    it = iter(full)
    stream = [next(it) for _ in range(30)]
    for p in range(0, n * 3, n):  # every pass is a permutation
        assert sorted(stream[p : p + n]) == list(range(n))
    for start in (0, 5, 7, 13):  # resuming at `start` continues the exact same stream (across pass boundaries)
        it2 = iter(InfiniteSampler(n, seed=3, start=start))
        assert [next(it2) for _ in range(30 - start)] == stream[start:]
    assert stream[:n] != [next(iter(InfiniteSampler(n, seed=4))) for _ in range(1)] * n


def _returns_table() -> pd.DataFrame:
    lengths = np.array(LENGTHS)
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]]) + 1000  # global index offset on purpose
    eps = pd.DataFrame(
        {
            "episode_index": np.arange(len(lengths)),
            "length": lengths,
            "success": SUCCESS,
            "dataset_from_index": starts,
            "dataset_to_index": starts + lengths,
        }
    )
    table, meta = compute_returns_table(eps, c_fail=20)
    return table, meta


def test_target_table_joins_on_index_not_row_order():
    table, _ = _returns_table()
    shuffled = table.sample(frac=1.0, random_state=0).reset_index(drop=True)
    tt = TargetTable.from_returns(shuffled)
    idx = table["index"].to_numpy()
    np.testing.assert_array_equal(tt.target_bin[idx], table["bin_index"].to_numpy())
    np.testing.assert_allclose(tt.return_norm[idx], table["return_norm"].to_numpy())
    tt.check_covers(idx, table["episode_index"].to_numpy(), table["frame_index"].to_numpy())
    with pytest.raises(ValueError, match="disagree"):
        tt.check_covers(idx, table["episode_index"].to_numpy(), table["frame_index"].to_numpy() + 1)
    with pytest.raises(KeyError):
        tt.check_covers(np.array([5]), np.array([0]), np.array([0]))  # index without a returns row


# ---------------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def store(tmp_path_factory):
    """Synthetic frame store + tables + splits + returns for 6 tiny episodes."""
    root = tmp_path_factory.mktemp("store")
    table, meta = _returns_table()
    rng = np.random.default_rng(0)
    for e, n in enumerate(LENGTHS):
        frames = {}
        for cam in CAMERAS:
            base = rng.integers(0, 256, (IMAGE_SIZE // 16, IMAGE_SIZE // 16, 3)).astype(np.uint8)
            img = np.kron(base, np.ones((16, 16, 1), np.uint8))
            frames[cam] = [encode_jpeg(img) for _ in range(n)]
        write_episode_h5(root / "frames" / f"ep_{e:04d}.h5", e, SUCCESS[e], frames)
    vec = pa.list_(pa.float32(), STATE_DIM)
    state = rng.normal(size=(len(table), STATE_DIM)).astype(np.float32)
    pq.write_table(
        pa.table(
            {
                "index": table["index"].to_numpy(),
                "episode_index": table["episode_index"].to_numpy(),
                "frame_index": table["frame_index"].to_numpy(),
                "timestamp": (table["frame_index"].to_numpy() / 50).astype(np.float32),
                "state": pa.array(list(state), type=vec),
                "action": pa.array(list(state), type=vec),
            }
        ),
        root / "frames.parquet",
    )
    (root / "splits.json").write_text(json.dumps({"seed": 0, "heldout": HELDOUT, "train": TRAIN, "folds": {}}))
    table.to_parquet(root / "returns_ctest.parquet", index=False)
    (root / "returns_ctest.json").write_text(json.dumps(meta))
    return root


def tiny_model(cfg: TrainConfig) -> ValueModel:
    vcfg = SiglipVisionConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4, image_size=224, patch_size=14
    )
    tcfg = Gemma3TextConfig(
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
    return ValueModel.from_backbone_configs(ValueModelConfig.from_dict(cfg.model), vcfg, tcfg)


def _cfg(store, run_name: str, **kw) -> TrainConfig:
    base = {
        "run_name": run_name,
        "returns_tag": "ctest",
        "returns_dir": str(store),
        "splits_file": str(store / "splits.json"),
        "frames_dir": str(store / "frames"),
        "frame_table": str(store / "frames.parquet"),
        "val_train_num_episodes": 2,
        "val_frame_stride": 3,
        "num_workers": 0,
        "eval_num_workers": 0,
        "model": {"gradient_checkpointing": False},
        "micro_batch": 2,
        "global_batch": 4,
        "steps": 6,
        "eval_every": 2,
        "eval_batch": 8,
        "warmup_steps": 2,
        "lr_backbone": 3e-4,
        "lr_new": 3e-3,
        "log_every": 1,
        "precision": "fp32",
        "early_stop_patience": 0,
        "plot_heldout_per_class": 1,
        "plot_train_episodes": 2,
    }
    base.update(kw)
    cfg = TrainConfig.from_dict(base)
    cfg.validate()
    return cfg


def test_target_dataset_items(store):
    table, _ = _returns_table()
    base = FrameDataset(TRAIN, frames_dir=store / "frames", frame_table=store / "frames.parquet")
    ds = TargetDataset(base, TargetTable.from_returns(table))
    item = ds[3]
    row = table.set_index("index").loc[int(item["index"])]
    assert item["target_bin"].item() == row["bin_index"] and item["target_bin"].dtype == torch.int64
    assert item["return_norm"].item() == pytest.approx(row["return_norm"])


def test_heldout_never_in_train(store, tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RUNS_ROOT", tmp_path)
    with pytest.raises(ValueError, match="held-out"):
        train(_cfg(store, "leak", train_episodes=[0, 2, 3]), model_factory=tiny_model, device="cpu")


def test_end_to_end_resume_matches_uninterrupted(store, tmp_path, monkeypatch):
    from safetensors.torch import load_file

    monkeypatch.setattr(paths, "RUNS_ROOT", tmp_path)
    full = train(_cfg(store, "full"), model_factory=tiny_model, device="cpu")
    assert full["step"] == 6 and full["best_step"] in (2, 4, 6)
    run = tmp_path / "full"
    for f in ("config.yaml", "metrics.jsonl", "train_log.jsonl", "summary.json", "plots/curves.png"):
        assert (run / f).is_file(), f
    assert (run / "plots" / "vcurves_000002.png").is_file()
    assert (run / "eval" / "val_heldout_step000006.parquet").is_file()
    assert list((run / "tb").iterdir())
    rows = [json.loads(x) for x in (run / "metrics.jsonl").read_text().splitlines()]
    assert [r["step"] for r in rows] == [2, 4, 6]
    for name in ("val_heldout", "val_train"):
        assert {"ce", "mae", "spearman", "auc@0", "auc@0.75", "fail_detect_time", "entropy"} <= set(rows[0][name])
    train_ce = [json.loads(x)["train_ce"] for x in (run / "train_log.jsonl").read_text().splitlines()]
    assert len(train_ce) == 6 and all(np.isfinite(train_ce))
    assert train_ce[0] == pytest.approx(np.log(201), abs=1e-3)  # zero head -> uniform distribution at step 1
    best_ce = min(r["val_heldout"]["ce"] for r in rows)
    assert full["best_val_heldout_ce"] == pytest.approx(best_ce)

    # checkpoint contract: model.safetensors + config.yaml with the resolved training config
    doc = yaml.safe_load((run / "checkpoints" / "best" / "config.yaml").read_text())
    assert doc["train"]["returns_tag"] == "ctest" and doc["resolved"]["returns"]["Z"] > 0
    assert len(doc["resolved"]["splits_sha256"]) == 64 and "commit" in doc["resolved"]["git"]
    assert doc["checkpoint"]["step"] == full["best_step"]
    assert doc["resolved"]["val_episodes"]["val_heldout"] == HELDOUT
    assert set(doc["resolved"]["val_episodes"]["val_train"]) <= set(TRAIN)
    ValueModel.from_pretrained(run / "checkpoints" / "best")  # loads without backbone downloads
    assert (run / "checkpoints" / "last" / "trainer_state.pt").is_file()

    # interrupted run: 4 steps, then resume to 6 -> same weights as the uninterrupted run
    train(_cfg(store, "split", steps=4), model_factory=tiny_model, device="cpu")
    with pytest.raises(FileExistsError):
        train(_cfg(store, "split", steps=4), model_factory=tiny_model, device="cpu")
    monkeypatch.chdir(paths.REPO_ROOT)
    cfg = resolve_config(None, ["steps=6"], resume=True, run_name="split")
    assert cfg.returns_tag == "ctest" and cfg.steps == 6  # saved config + overrides
    resumed = train(cfg, resume=True, model_factory=tiny_model, device="cpu")
    assert resumed["step"] == 6
    a = load_file(str(run / "checkpoints" / "last" / "model.safetensors"))
    b = load_file(str(tmp_path / "split" / "checkpoints" / "last" / "model.safetensors"))
    for k in a:
        torch.testing.assert_close(a[k], b[k], rtol=1e-4, atol=1e-5)
    rows = [json.loads(x) for x in (tmp_path / "split" / "metrics.jsonl").read_text().splitlines()]
    assert [r["step"] for r in rows] == [2, 4, 6]


def test_early_stopping(store, tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RUNS_ROOT", tmp_path)
    # zero learning rate -> val CE never improves after the first eval -> stop after `patience` more evals
    cfg = _cfg(store, "es", lr_backbone=0.0, lr_new=0.0, early_stop_patience=1, steps=10)
    out = train(cfg, model_factory=tiny_model, device="cpu")
    assert out["early_stopped"] and out["step"] == 4 and out["best_step"] == 2
