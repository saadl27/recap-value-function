"""Step 1: timestamp mapping, table validation, HDF5 store and an end-to-end extraction on a synthetic mp4."""

import json

import av
import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from recap_vf.constants import CAMERAS, FPS, IMAGE_SIZE, VIDEO_KEYS
from recap_vf.data import frames as fr

# ---------------------------------------------------------------------------------------------------------------------
# timestamp -> (episode, frame)


def test_map_time_to_frame_boundaries():
    from_ts = np.array([0.0, 12.48, 21.78])  # ep0 624 frames, ep1 465 frames (real values of the dataset)
    t = np.array([0.0, 0.02, 12.46, 12.48, 12.48 - 5e-5, 12.50, 21.76, 21.78, 30.0])
    ep, fi, res = fr.map_time_to_frame(t, from_ts)
    assert ep.tolist() == [0, 0, 0, 1, 1, 1, 1, 2, 2]
    assert fi.tolist() == [0, 1, 623, 0, 0, 1, 464, 0, 411]
    assert np.abs(res).max() < 1e-4


def test_map_time_to_frame_before_first_episode_and_float_noise():
    from_ts = np.array([1.0, 2.0])
    ep, fi, _ = fr.map_time_to_frame(np.array([0.5, 1.0 - 1e-9, 1.0 + 1e-9]), from_ts)
    assert ep.tolist() == [-1, 0, 0]
    assert fi.tolist() == [-1, 0, 0]
    # pts * time_base of frame k at 1/12800 is exact; from_ts as float64 may carry rounding noise
    ts = np.arange(0, 200) * 256 / 12800 + 1.0
    ep, fi, _ = fr.map_time_to_frame(ts, from_ts)
    assert (ep[:50] == 0).all() and (ep[50:] == 1).all()
    assert np.array_equal(fi, np.r_[np.arange(50), np.arange(150)])


def test_parse_episodes():
    assert fr.parse_episodes("0-3", 10) == [0, 1, 2, 3]
    assert fr.parse_episodes("5, 1,2-3,2", 10) == [1, 2, 3, 5]
    assert fr.parse_episodes("all", 4) == [0, 1, 2, 3]
    with pytest.raises(ValueError):
        fr.parse_episodes("8-12", 10)


# ---------------------------------------------------------------------------------------------------------------------
# tables


def _raw_tables(lengths, success, fps=FPS):
    lengths = np.asarray(lengths)
    to_idx = np.cumsum(lengths)
    from_idx = to_idx - lengths
    raw_eps = pd.DataFrame(
        {
            "episode_index": np.arange(len(lengths)),
            "length": lengths,
            "dataset_from_index": from_idx,
            "dataset_to_index": to_idx,
            "success": success,
        }
    )
    for cam in CAMERAS:
        key = VIDEO_KEYS[cam]
        raw_eps[f"videos/{key}/chunk_index"] = 0
        raw_eps[f"videos/{key}/file_index"] = 0
        raw_eps[f"videos/{key}/from_timestamp"] = from_idx / fps
        raw_eps[f"videos/{key}/to_timestamp"] = to_idx / fps
    n = int(lengths.sum())
    ep = np.repeat(np.arange(len(lengths)), lengths)
    fi = np.arange(n) - np.repeat(from_idx, lengths)
    vec = pa.list_(pa.float32(), 14)
    rng = np.random.default_rng(0)
    raw_frames = pa.table(
        {
            "index": np.arange(n),
            "episode_index": ep,
            "frame_index": fi,
            "timestamp": (fi / fps).astype(np.float32),
            "observation.state": pa.array(list(rng.normal(size=(n, 14)).astype(np.float32)), type=vec),
            "action": pa.array(list(rng.normal(size=(n, 14)).astype(np.float32)), type=vec),
        }
    )
    return raw_eps, raw_frames


def test_tables_build_and_validate():
    raw_eps, raw_frames = _raw_tables([5, 3, 4], [True, False, True])
    eps = fr.build_episode_table(raw_eps)
    frames = fr.build_frame_table(raw_frames)
    assert list(eps.columns) == fr.EPISODE_TABLE_COLUMNS
    assert eps["success"].dtype == bool
    assert frames.schema.field("state").type == pa.list_(pa.float32(), 14)
    assert frames.schema.field("timestamp").type == pa.float32()
    spans = [fr.video_spans(raw_eps, cam) for cam in CAMERAS]
    summary = fr.validate_tables(eps, frames, spans)
    assert summary["num_frames"] == 12 and summary["num_success"] == 2


def test_tables_validation_catches_errors():
    raw_eps, raw_frames = _raw_tables([5, 3, 4], [True, False, True])
    eps = fr.build_episode_table(raw_eps)
    frames = fr.build_frame_table(raw_frames)
    bad_ts = frames.set_column(3, "timestamp", pa.array(np.zeros(12, np.float32)))
    with pytest.raises(fr.DataValidationError, match="timestamp"):
        fr.validate_tables(eps, bad_ts)
    bad_eps = eps.copy()
    bad_eps.loc[1, "length"] = 4
    with pytest.raises(fr.DataValidationError):
        fr.validate_tables(bad_eps, frames)
    raw_eps.loc[1, f"videos/{VIDEO_KEYS['top']}/to_timestamp"] += 0.04  # span says 5 frames, length says 3
    with pytest.raises(fr.DataValidationError, match="to-from"):
        fr.validate_tables(eps, frames, [fr.video_spans(raw_eps, "top")])


# ---------------------------------------------------------------------------------------------------------------------
# HDF5 store


def test_write_episode_h5_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    jpegs = {
        cam: [fr.encode_jpeg(rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)) for _ in range(4)] for cam in CAMERAS
    }
    p = tmp_path / "ep_0007.h5"
    fr.write_episode_h5(p, 7, True, jpegs)
    assert not list(tmp_path.glob("*.tmp"))
    assert fr.is_episode_complete(p, 4)
    assert not fr.is_episode_complete(p, 5)
    assert not fr.is_episode_complete(tmp_path / "missing.h5", 4)
    with h5py.File(p) as f:
        assert dict(f.attrs)["episode_index"] == 7 and bool(f.attrs["success"]) and f.attrs["length"] == 4
        for cam in CAMERAS:
            assert f[cam].dtype == h5py.vlen_dtype(np.uint8)
            assert f[cam][2].tobytes() == jpegs[cam][2]
    with pytest.raises(fr.DataValidationError):
        fr.write_episode_h5(p, 7, True, {"top": jpegs["top"], "left_wrist": jpegs["left_wrist"][:3]})


# ---------------------------------------------------------------------------------------------------------------------
# end-to-end extraction on a synthetic concatenated mp4


def _write_video(path, n_frames, fps=FPS):
    """Solid grey frames whose level encodes the global frame number (level = 10 + 15 * k)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as c:
        s = c.add_stream("libx264", rate=fps, options={"crf": "10", "g": "2"})
        s.width = s.height = IMAGE_SIZE
        s.pix_fmt = "yuv420p"
        for k in range(n_frames):
            img = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 10 + 15 * k, np.uint8)
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            frame.pts = k
            for pkt in s.encode(frame):
                c.mux(pkt)
        for pkt in s.encode():
            c.mux(pkt)


def _make_raw_dataset(root, lengths, success, video_frames=None):
    raw_eps, raw_frames = _raw_tables(lengths, success)
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    raw_eps.to_parquet(root / "meta/episodes/chunk-000/file-000.parquet")
    (root / "data" / "chunk-000").mkdir(parents=True)
    import pyarrow.parquet as pq

    pq.write_table(raw_frames, root / "data/chunk-000/file-000.parquet")
    info = {"video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"}
    (root / "meta/info.json").write_text(json.dumps(info))
    n = video_frames if video_frames is not None else int(sum(lengths))
    _write_video(root / f"videos/{VIDEO_KEYS['top']}/chunk-000/file-000.mp4", n)
    for cam in CAMERAS[1:]:
        dst = root / f"videos/{VIDEO_KEYS[cam]}/chunk-000/file-000.mp4"
        dst.parent.mkdir(parents=True)
        dst.write_bytes((root / f"videos/{VIDEO_KEYS['top']}/chunk-000/file-000.mp4").read_bytes())
    return raw_eps


def _cache_tables(raw, cache):
    cache.mkdir(parents=True, exist_ok=True)
    fr.build_episode_table(fr.load_raw_episodes(raw)).to_parquet(cache / "episodes.parquet")


def test_extract_frames_end_to_end(tmp_path):
    raw, cache, store = tmp_path / "raw", tmp_path / "cache", tmp_path / "cache/frames"
    lengths = [5, 3, 4]
    _make_raw_dataset(raw, lengths, [True, False, True])
    _cache_tables(raw, cache)
    # episodes 0 and 2: exercises a seek to episode 2 and decoding up to the end of the stream
    res = fr.extract_frames([0, 2], source="local", raw_dir=raw, cache_dir=cache, frames_dir=store)
    assert res["written_episodes"] == 2 and res["written_frames"] == 9
    for cam in CAMERAS:
        assert res["cameras"][cam]["episodes_exact"] == 2
    assert sorted(p.name for p in store.iterdir()) == ["ep_0000.h5", "ep_0002.h5"]
    starts = np.cumsum([0, *lengths])
    from recap_vf.data.dataset import decode_jpeg

    for e in (0, 2):
        with h5py.File(store / f"ep_{e:04d}.h5") as f:
            assert f.attrs["length"] == lengths[e] and bool(f.attrs["success"]) == (e != 1)
            for cam in CAMERAS:
                levels = [float(decode_jpeg(f[cam][i]).float().mean()) for i in range(lengths[e])]
                expected = [10 + 15 * (starts[e] + i) for i in range(lengths[e])]
                assert np.allclose(levels, expected, atol=3), (cam, e, levels, expected)
    # resumable: nothing to do the second time
    res2 = fr.extract_frames([0, 1, 2], source="local", raw_dir=raw, cache_dir=cache, frames_dir=store)
    assert res2["skipped_complete"] == 2 and res2["written_episodes"] == 1


def test_extract_frames_refuses_missing_frames(tmp_path):
    raw, cache, store = tmp_path / "raw", tmp_path / "cache", tmp_path / "cache/frames"
    _make_raw_dataset(raw, [5, 3, 4], [True, False, True], video_frames=10)  # last episode is 2 frames short
    _cache_tables(raw, cache)
    with pytest.raises(fr.DataValidationError, match="missing=2"):
        fr.extract_frames([1, 2], source="local", raw_dir=raw, cache_dir=cache, frames_dir=store)
    assert (store / "ep_0001.h5").is_file()  # complete episodes before the failure are kept
    assert not (store / "ep_0002.h5").exists() and not list(store.glob("*.tmp"))
