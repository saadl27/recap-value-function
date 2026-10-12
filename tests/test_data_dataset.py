"""Step 1: FrameDataset on a tiny synthetic store (2 episodes x 5 frames)."""

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from torch.utils.data import DataLoader

from recap_vf.constants import CAMERAS, IMAGE_SIZE, STATE_DIM
from recap_vf.data.dataset import FrameDataset, decode_jpeg
from recap_vf.data.frames import encode_jpeg, write_episode_h5

LENGTHS = {3: 5, 7: 5}  # episode -> length; non-contiguous episode ids on purpose


def _smooth_image(rng):
    """Low-frequency random RGB image (JPEG-friendly, like real camera frames)."""
    small = rng.integers(0, 256, (8, 8, 3)).astype(np.float32)
    big = np.kron(small, np.ones((IMAGE_SIZE // 8, IMAGE_SIZE // 8, 1), np.float32))
    k = np.ones(9, np.float32) / 9  # blur to avoid hard block edges
    for ax in (0, 1):
        big = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), ax, big)
    return big.clip(0, 255).astype(np.uint8)


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    root = tmp_path_factory.mktemp("store")
    rng = np.random.default_rng(0)
    images, rows, start = {}, [], 100
    for e, n in LENGTHS.items():
        imgs = {cam: [_smooth_image(rng) for _ in range(n)] for cam in CAMERAS}
        images[e] = imgs
        write_episode_h5(
            root / "frames" / f"ep_{e:04d}.h5", e, e == 3, {c: [encode_jpeg(x) for x in v] for c, v in imgs.items()}
        )
        for i in range(n):
            rows.append((start, e, i))
            start += 1
    rows = rows[::-1]  # table written out of order: the dataset must sort by index
    idx, ep, fi = (np.array(c) for c in zip(*rows, strict=True))
    state = np.stack([np.full(STATE_DIM, float(k), np.float32) for k in idx])
    vec = pa.list_(pa.float32(), STATE_DIM)
    table = pa.table(
        {
            "index": idx,
            "episode_index": ep,
            "frame_index": fi,
            "timestamp": (fi / 50).astype(np.float32),
            "state": pa.array(list(state), type=vec),
            "action": pa.array(list(state), type=vec),
        }
    )
    pq.write_table(table, root / "frames.parquet")
    return root, images


def _ds(store, **kw):
    root, _ = store
    return FrameDataset([7, 3], frames_dir=root / "frames", frame_table=root / "frames.parquet", **kw)


def test_item_shapes_dtypes_and_order(store):
    ds = _ds(store)
    assert len(ds) == 10
    assert ds.index.tolist() == list(range(100, 110))
    item = ds[6]
    assert set(item["images"]) == set(CAMERAS)
    for img in item["images"].values():
        assert img.dtype == torch.uint8 and img.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert item["state"].dtype == torch.float32 and item["state"].shape == (STATE_DIM,)
    assert item["index"].item() == 106 and item["episode_index"].item() == 7 and item["frame_index"].item() == 1
    assert torch.all(item["state"] == 106.0)  # state row joined by index
    assert item["index"].dtype == torch.int64


def test_jpeg_roundtrip_error_is_small(store):
    _, images = store
    ds = _ds(store)
    for i in (0, 9):
        item = ds[i]
        e, fi = item["episode_index"].item(), item["frame_index"].item()
        for cam in CAMERAS:
            orig = torch.from_numpy(images[e][cam][fi]).permute(2, 0, 1).float()
            err = (item["images"][cam].float() - orig).abs().mean().item()
            assert err < 2.0, (cam, err)


def test_decoders_agree(store):
    a, b = _ds(store, decoder="torchvision")[4], _ds(store, decoder="pil")[4]
    for cam in CAMERAS:
        assert (a["images"][cam].int() - b["images"][cam].int()).abs().max() <= 2
    raw = np.frombuffer(encode_jpeg(np.zeros((16, 16, 3), np.uint8)), np.uint8)
    assert decode_jpeg(raw, "pil").shape == decode_jpeg(raw).shape == (3, 16, 16)


def test_frame_stride_and_cameras(store):
    ds = _ds(store, frame_stride=2, cameras=("top",))
    assert ds.frame_index.tolist() == [0, 2, 4, 0, 2, 4]
    assert list(ds[0]["images"]) == ["top"]


def test_get_episode(store):
    ds = _ds(store, frame_stride=3)
    ep = ds.get_episode(7)
    assert ep["images"]["top"].shape == (5, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert ep["index"].tolist() == [105, 106, 107, 108, 109] and ep["success"] is False
    sub = ds.get_episode(3, frames=[0, 4], cameras=["left_wrist"])
    assert list(sub["images"]) == ["left_wrist"] and sub["frame_index"].tolist() == [0, 4]
    with pytest.raises(KeyError):
        ds.get_episode(5)


@pytest.mark.parametrize("context,workers", [("fork", 4), ("spawn", 2)])
def test_dataloader_with_workers(store, context, workers):
    ds = _ds(store)
    _ = ds[0]  # open a handle in the parent: workers must neither reuse it (fork) nor pickle it (spawn)
    dl = DataLoader(ds, batch_size=4, num_workers=workers, shuffle=True, multiprocessing_context=context)
    seen = []
    for batch in dl:
        assert batch["images"]["top"].shape[1:] == (3, IMAGE_SIZE, IMAGE_SIZE)
        assert batch["state"].shape[1:] == (STATE_DIM,)
        assert torch.all(batch["state"][:, 0] == batch["index"].float())
        seen += batch["index"].tolist()
    assert sorted(seen) == list(range(100, 110))


def test_missing_store_file_is_reported(store, tmp_path):
    root, _ = store
    with pytest.raises(FileNotFoundError):
        FrameDataset([3], frames_dir=tmp_path, frame_table=root / "frames.parquet")
    with pytest.raises(KeyError):
        FrameDataset([4], frames_dir=root / "frames", frame_table=root / "frames.parquet")
