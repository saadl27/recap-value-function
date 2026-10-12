"""Raw-dataset validation, frame/episode tables and the mp4 -> per-episode JPEG/HDF5 frame store.

The HF dataset stores each camera as one H.264 mp4 with all episodes concatenated. We decode each camera once,
sequentially, map every decoded frame to `(episode, frame_index)` through the episode spans in `meta/episodes`,
assert that every frame of every requested episode is hit exactly once, and write one HDF5 file per episode holding
the JPEG bytes of the three cameras (`docs/conventions.md` -> "Frame store").
"""

from __future__ import annotations

import io
import json
import logging
import multiprocessing as mp
import os
import queue
import shutil
import ssl
import time
import traceback
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from PIL import Image

from recap_vf import paths
from recap_vf.constants import (
    CAMERAS,
    DATASET_REPO_ID,
    FPS,
    IMAGE_SIZE,
    NUM_EPISODES,
    NUM_FRAMES,
    STATE_DIM,
    VIDEO_KEYS,
)

log = logging.getLogger(__name__)

#: Tolerance (seconds) when comparing timestamps; LeRobot uses the same value. Frames are 20 ms apart at 50 fps.
TIME_TOL = 1e-4
JPEG_QUALITY = 95
EPISODE_TABLE_COLUMNS = ["episode_index", "length", "success", "dataset_from_index", "dataset_to_index"]


class DataValidationError(RuntimeError):
    """Raised when the raw dataset or the decoded video does not match what the metadata promises."""


# --------------------------------------------------------------------------------------------------------------------
# Raw dataset: info.json and tables
# --------------------------------------------------------------------------------------------------------------------


def validate_info(raw_dir: Path = paths.RAW_DIR) -> dict:
    """Check `meta/info.json` against the constants we rely on; raise DataValidationError on any mismatch."""
    info = json.loads((Path(raw_dir) / "meta" / "info.json").read_text())
    problems = []

    def expect(name: str, got, want) -> None:
        if got != want:
            problems.append(f"{name}: expected {want!r}, got {got!r}")

    expect("fps", info.get("fps"), FPS)
    expect("total_episodes", info.get("total_episodes"), NUM_EPISODES)
    expect("total_frames", info.get("total_frames"), NUM_FRAMES)
    feats = info.get("features", {})
    expect("observation.state.shape", feats.get("observation.state", {}).get("shape"), [STATE_DIM])
    expect("action.shape", feats.get("action", {}).get("shape"), [STATE_DIM])
    for key in VIDEO_KEYS.values():
        f = feats.get(key)
        if f is None:
            problems.append(f"missing video feature {key}")
            continue
        expect(f"{key}.dtype", f.get("dtype"), "video")
        expect(f"{key}.shape", f.get("shape"), [IMAGE_SIZE, IMAGE_SIZE, 3])
        expect(f"{key}.video.fps", float(f.get("info", {}).get("video.fps", -1)), float(FPS))
    video_feats = sorted(k for k, v in feats.items() if v.get("dtype") == "video")
    expect("video keys", video_feats, sorted(VIDEO_KEYS.values()))
    if problems:
        raise DataValidationError("meta/info.json does not match recap_vf.constants:\n  " + "\n  ".join(problems))
    return info


def load_raw_episodes(raw_dir: Path = paths.RAW_DIR) -> pd.DataFrame:
    """All `meta/episodes/*/*.parquet` rows, sorted by episode_index."""
    files = sorted((Path(raw_dir) / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no episode metadata under {raw_dir}/meta/episodes")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df.sort_values("episode_index").reset_index(drop=True)


def load_raw_frames(raw_dir: Path = paths.RAW_DIR) -> pa.Table:
    """All `data/*/*.parquet` rows (one per frame) as an Arrow table."""
    files = sorted((Path(raw_dir) / "data").glob("chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no frame data under {raw_dir}/data")
    cols = ["index", "episode_index", "frame_index", "timestamp", "observation.state", "action"]
    return pa.concat_tables([pq.read_table(f, columns=cols) for f in files])


def build_episode_table(raw_episodes: pd.DataFrame) -> pd.DataFrame:
    """Contract `episodes.parquet`: episode_index, length, success (bool), dataset_from_index, dataset_to_index."""
    df = raw_episodes[EPISODE_TABLE_COLUMNS].copy()
    for c in ("episode_index", "length", "dataset_from_index", "dataset_to_index"):
        df[c] = df[c].astype(np.int64)
    df["success"] = df["success"].astype(bool)
    return df.sort_values("episode_index").reset_index(drop=True)


def build_frame_table(raw_frames: pa.Table) -> pa.Table:
    """Contract `frames.parquet`: index, episode_index, frame_index, timestamp, state[14], action[14] (by index)."""
    vec = pa.list_(pa.float32(), STATE_DIM)
    table = pa.table(
        {
            "index": pc.cast(raw_frames["index"], pa.int64()),
            "episode_index": pc.cast(raw_frames["episode_index"], pa.int64()),
            "frame_index": pc.cast(raw_frames["frame_index"], pa.int64()),
            "timestamp": pc.cast(raw_frames["timestamp"], pa.float32()),
            "state": pc.cast(raw_frames["observation.state"], vec),
            "action": pc.cast(raw_frames["action"], vec),
        }
    )
    return table.sort_by("index")


@dataclass
class VideoSpans:
    """Where each episode lives inside one camera's concatenated mp4 (positions == episode_index)."""

    camera: str
    video_path: str  # relative to the dataset root, e.g. videos/observation.images.top/chunk-000/file-000.mp4
    from_ts: np.ndarray  # float64 [E]
    to_ts: np.ndarray  # float64 [E]
    lengths: np.ndarray  # int64 [E]


def video_spans(raw_episodes: pd.DataFrame, camera: str, info: dict | None = None) -> VideoSpans:
    """Episode spans of one camera; raises if the episodes span several mp4 files (not the case for this dataset)."""
    key = VIDEO_KEYS[camera]
    chunk = raw_episodes[f"videos/{key}/chunk_index"].to_numpy()
    file = raw_episodes[f"videos/{key}/file_index"].to_numpy()
    if len(np.unique(chunk)) != 1 or len(np.unique(file)) != 1:
        raise NotImplementedError(f"{camera}: episodes spread over several mp4 files; only one file is supported")
    template = (info or {}).get("video_path", "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4")
    rel = template.format(video_key=key, chunk_index=int(chunk[0]), file_index=int(file[0]))
    if not np.array_equal(raw_episodes["episode_index"].to_numpy(), np.arange(len(raw_episodes))):
        raise DataValidationError("episode_index is not 0..E-1 in episode order")
    return VideoSpans(
        camera=camera,
        video_path=rel,
        from_ts=raw_episodes[f"videos/{key}/from_timestamp"].to_numpy(np.float64),
        to_ts=raw_episodes[f"videos/{key}/to_timestamp"].to_numpy(np.float64),
        lengths=raw_episodes["length"].to_numpy(np.int64),
    )


def validate_tables(episodes: pd.DataFrame, frames: pa.Table, spans: Sequence[VideoSpans] = (), fps: int = FPS) -> dict:
    """Consistency checks between the episode table, the frame table and the video spans. Returns summary numbers."""
    problems: list[str] = []
    ep = episodes["episode_index"].to_numpy()
    length = episodes["length"].to_numpy()
    d_from = episodes["dataset_from_index"].to_numpy()
    d_to = episodes["dataset_to_index"].to_numpy()
    n_ep = len(episodes)

    if not np.array_equal(ep, np.arange(n_ep)):
        problems.append("episode_index is not contiguous 0..E-1")
    if (length <= 0).any():
        problems.append("non-positive episode length")
    if not np.array_equal(d_to - d_from, length):
        problems.append("dataset_to_index - dataset_from_index != length")
    if n_ep and (d_from[0] != 0 or not np.array_equal(d_from[1:], d_to[:-1])):
        problems.append("episodes are not contiguous in the global index")

    index = frames["index"].to_numpy()
    f_ep = frames["episode_index"].to_numpy()
    f_fi = frames["frame_index"].to_numpy()
    ts = frames["timestamp"].to_numpy().astype(np.float64)
    if not np.array_equal(index, np.arange(len(index))):
        problems.append(f"frame `index` is not contiguous 0..{len(index) - 1}")
    if len(index) != int(length.sum()):
        problems.append(f"frame table has {len(index)} rows but episode lengths sum to {int(length.sum())}")
    else:
        exp_ep = np.repeat(ep, length)
        exp_fi = np.arange(len(index)) - np.repeat(d_from, length)
        if not np.array_equal(f_ep, exp_ep):
            problems.append("frame rows do not map to episodes through [dataset_from_index, dataset_to_index)")
        if not np.array_equal(f_fi, exp_fi):
            problems.append("frame_index is not 0..length-1 within every episode")
    ts_err = float(np.abs(ts - f_fi / fps).max()) if len(ts) else 0.0
    if ts_err > TIME_TOL:
        problems.append(f"timestamp != frame_index / {fps} (max error {ts_err:.2e} s)")
    state = frames["state"].combine_chunks().flatten().to_numpy()
    if not np.isfinite(state).all():
        problems.append("non-finite values in state")

    summary: dict = {
        "num_episodes": n_ep,
        "num_frames": len(index),
        "num_success": int(episodes["success"].sum()),
        "max_timestamp_error_s": ts_err,
        "video": {},
    }
    for sp in spans:
        dur_err = np.abs((sp.to_ts - sp.from_ts) - sp.lengths / fps)
        gaps = sp.from_ts[1:] - sp.to_ts[:-1]
        if (np.diff(sp.from_ts) <= 0).any():
            problems.append(f"{sp.camera}: from_timestamp is not strictly increasing with episode_index")
        if dur_err.max(initial=0) > 0.5 / fps:
            problems.append(f"{sp.camera}: to-from differs from length/fps by up to {dur_err.max():.4f} s")
        if len(gaps) and gaps.min() < -TIME_TOL:
            problems.append(f"{sp.camera}: overlapping episode spans (min gap {gaps.min():.4f} s)")
        summary["video"][sp.camera] = {
            "video_path": sp.video_path,
            "max_abs_duration_minus_length_over_fps_s": float(dur_err.max(initial=0)),
            "max_abs_gap_between_episodes_s": float(np.abs(gaps).max(initial=0)),
            "end_s": float(sp.to_ts[-1]) if len(sp.to_ts) else 0.0,
        }
    if problems:
        raise DataValidationError("table validation failed:\n  " + "\n  ".join(problems))
    return summary


def write_tables(raw_dir: Path = paths.RAW_DIR, cache_dir: Path = paths.CACHE_DIR, validate_meta: bool = True) -> dict:
    """Build, validate and write `episodes.parquet` + `frames.parquet` for **all** episodes (they are small)."""
    info = validate_info(raw_dir) if validate_meta else json.loads((Path(raw_dir) / "meta/info.json").read_text())
    raw_eps = load_raw_episodes(raw_dir)
    episodes = build_episode_table(raw_eps)
    frames = build_frame_table(load_raw_frames(raw_dir))
    spans = [video_spans(raw_eps, cam, info) for cam in CAMERAS]
    summary = validate_tables(episodes, frames, spans)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write(cache_dir / "episodes.parquet", lambda p: episodes.to_parquet(p, index=False))
    _atomic_write(cache_dir / "frames.parquet", lambda p: pq.write_table(frames, p))
    return summary


def _atomic_write(path: Path, writer) -> None:
    tmp = path.with_name(path.name + ".tmp")
    writer(tmp)
    os.replace(tmp, path)


# --------------------------------------------------------------------------------------------------------------------
# Timestamp -> (episode, frame_index)
# --------------------------------------------------------------------------------------------------------------------


def map_time_to_frame(
    t: float | np.ndarray, from_ts: np.ndarray, fps: int = FPS, tol: float = TIME_TOL
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map video time(s) to `(episode, frame_index, residual_s)`.

    episode = last e with from_ts[e] <= t + tol (-1 if t precedes every episode),
    frame_index = round((t - from_ts[e]) * fps), residual = t - (from_ts[e] + frame_index / fps).
    `frame_index` may be >= the episode length; the caller decides what to do with such "extra" frames.
    """
    t = np.asarray(t, dtype=np.float64)
    ep = np.searchsorted(from_ts, t + tol, side="right") - 1
    start = np.where(ep >= 0, from_ts[np.clip(ep, 0, None)], 0.0)
    fi = np.rint((t - start) * fps).astype(np.int64)
    residual = t - (start + fi / fps)
    fi = np.where(ep >= 0, fi, -1)
    return ep, fi, residual


def parse_episodes(spec: str | None, num_episodes: int = NUM_EPISODES) -> list[int]:
    """'all' | '0-9' | '0,3,5-7' -> sorted unique episode list (ranges are inclusive)."""
    if spec is None or spec.strip().lower() in ("", "all"):
        return list(range(num_episodes))
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = (int(x) for x in part.split("-", 1))
            if b < a:
                raise ValueError(f"bad episode range {part!r}")
            out.update(range(a, b + 1))
        else:
            out.add(int(part))
    bad = [e for e in out if not 0 <= e < num_episodes]
    if bad:
        raise ValueError(f"episodes out of range [0, {num_episodes}): {sorted(bad)[:10]}")
    return sorted(out)


# --------------------------------------------------------------------------------------------------------------------
# Video decoding
# --------------------------------------------------------------------------------------------------------------------


def video_url(video_path: str, repo_id: str = DATASET_REPO_ID, revision: str = "main") -> str:
    from huggingface_hub import hf_hub_url

    return hf_hub_url(repo_id, video_path, repo_type="dataset", revision=revision)


def _ca_file() -> str | None:
    """CA bundle for ffmpeg's TLS (the PyAV wheel's OpenSSL may not find the system store by itself)."""
    candidates = [
        os.environ.get("SSL_CERT_FILE"),
        ssl.get_default_verify_paths().cafile,
        "/etc/ssl/certs/ca-certificates.crt",
        "/etc/pki/tls/certs/ca-bundle.crt",
        "/etc/ssl/cert.pem",
    ]
    try:
        import certifi

        candidates.append(certifi.where())
    except ImportError:
        pass
    return next((c for c in candidates if c and Path(c).is_file()), None)


def open_video(source: str, timeout: float = 60.0):
    """Open a local mp4 or an http(s) URL with PyAV (URL: CA bundle + ffmpeg's reconnect options)."""
    import av

    options: dict[str, str] = {}
    if source.startswith(("http://", "https://")):
        options = {"reconnect": "1", "reconnect_streamed": "1", "reconnect_on_network_error": "1"}
        options["reconnect_delay_max"] = "30"
        ca = _ca_file()
        if ca:
            options["ca_file"] = ca
    return av.open(source, options=options, timeout=timeout)


def encode_jpeg(rgb: np.ndarray, quality: int = JPEG_QUALITY) -> bytes:
    """HxWx3 uint8 RGB -> JPEG bytes (PIL / libjpeg-turbo, 4:2:0 like the yuv420p source)."""
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


@dataclass
class EpisodeDecodeStats:
    episode_index: int
    length: int
    hit: int = 0
    missing: int = 0
    duplicated: int = 0
    extra: int = 0  # decoded frames mapped to this episode with frame_index >= length
    max_residual_s: float = 0.0


@dataclass
class CameraDecodeStats:
    camera: str
    decoded: int = 0  # every frame that came out of the decoder (incl. pre-roll after seeks)
    encoded: int = 0
    skipped: int = 0  # decoded but outside the requested episodes (pre-roll, gaps, other episodes)
    jpeg_bytes: int = 0
    reconnects: int = 0
    seeks: int = 0
    max_residual_s: float = 0.0
    episodes: list[dict] = field(default_factory=list)


def _contiguous_runs(eps: Sequence[int]) -> list[list[int]]:
    runs: list[list[int]] = []
    for e in eps:
        if runs and e == runs[-1][-1] + 1:
            runs[-1].append(e)
        else:
            runs.append([e])
    return runs


def iter_camera_episodes(
    source: str,
    spans: VideoSpans,
    episodes: Sequence[int],
    quality: int = JPEG_QUALITY,
    fps: int = FPS,
    tol: float = TIME_TOL,
    max_retries: int = 5,
    stats: CameraDecodeStats | None = None,
) -> Iterator[tuple[int, list[bytes], EpisodeDecodeStats]]:
    """Decode one camera sequentially and yield `(episode, jpegs[length], stats)` for each requested episode in order.

    Seeks to the first frame of each contiguous run of requested episodes (cheap: a keyframe every 2 frames) and
    stops after the last one. Raises DataValidationError if an episode is not hit exactly once per frame.
    On transient I/O errors (URL source) the container is reopened and decoding resumes after the last frame seen.
    """
    import av

    stats = stats if stats is not None else CameraDecodeStats(camera=spans.camera)
    from_ts, lengths = spans.from_ts, spans.lengths
    container = None
    stream = None
    retries = 0

    def finish(e: int, slots: list[bytes | None], est: EpisodeDecodeStats):
        est.missing = sum(s is None for s in slots)
        est.hit = len(slots) - est.missing
        stats.episodes.append(asdict(est))
        if est.missing or est.duplicated or est.extra:
            raise DataValidationError(
                f"{spans.camera} episode {e}: length={est.length} hit={est.hit} missing={est.missing} "
                f"duplicated={est.duplicated} extra={est.extra}; refusing to pad or drop frames"
            )
        return e, slots, est

    try:
        for run in _contiguous_runs(sorted(episodes)):
            last = run[-1]
            cur = run[0]
            slots: list[bytes | None] = [None] * int(lengths[cur])
            est = EpisodeDecodeStats(cur, int(lengths[cur]))
            last_t = -np.inf
            seek_t = float(from_ts[cur])
            done = False
            while not done:
                try:
                    if container is None:
                        container = open_video(source)
                        stream = container.streams.video[0]
                        stream.thread_type = "AUTO"
                    if seek_t > 0:
                        container.seek(int(seek_t / stream.time_base), stream=stream, backward=True, any_frame=False)
                        stats.seeks += 1
                    for frame in container.decode(stream):
                        stats.decoded += 1
                        if frame.pts is None:
                            raise DataValidationError(f"{spans.camera}: decoded frame without pts")
                        t = float(frame.pts * stream.time_base)
                        if t <= last_t + tol:  # already handled (re-decoded after a reconnect)
                            stats.skipped += 1
                            continue
                        last_t = t
                        ep_arr, fi_arr, _ = map_time_to_frame(t, from_ts, fps, tol)
                        e, fi = int(ep_arr), int(fi_arr)
                        if e < cur:  # pre-roll after a seek
                            stats.skipped += 1
                            continue
                        if e > last:
                            done = True
                            break
                        while e > cur:  # episode `cur` is over
                            yield finish(cur, slots, est)
                            cur += 1
                            slots = [None] * int(lengths[cur])
                            est = EpisodeDecodeStats(cur, int(lengths[cur]))
                        residual = abs(t - (from_ts[e] + fi / fps))
                        if fi >= len(slots):
                            est.extra += 1
                            continue
                        if slots[fi] is not None:
                            est.duplicated += 1
                            continue
                        if residual > 0.25 / fps:
                            raise DataValidationError(
                                f"{spans.camera}: frame at t={t:.5f}s is {residual:.4f}s off grid"
                            )
                        est.max_residual_s = max(est.max_residual_s, residual)
                        stats.max_residual_s = max(stats.max_residual_s, residual)
                        if frame.width != IMAGE_SIZE or frame.height != IMAGE_SIZE:
                            raise DataValidationError(f"{spans.camera}: frame is {frame.width}x{frame.height}")
                        jpg = encode_jpeg(frame.to_ndarray(format="rgb24"), quality)
                        slots[fi] = jpg
                        stats.encoded += 1
                        stats.jpeg_bytes += len(jpg)
                    else:
                        done = True  # end of stream
                except (av.error.OSError, av.error.EOFError, av.error.ExitError, av.error.HTTPError) as err:
                    retries += 1
                    stats.reconnects += 1
                    if retries > max_retries or not source.startswith(("http://", "https://")):
                        raise
                    log.warning("%s: I/O error (%s), reopening (%d/%d)", spans.camera, err, retries, max_retries)
                    if container is not None:
                        container.close()
                    container = None
                    time.sleep(min(2.0**retries, 30.0))
                    seek_t = float(max(last_t, from_ts[cur]))
            # end of run: flush the current episode and any requested episode that never showed up
            while cur <= last:
                yield finish(cur, slots, est)
                cur += 1
                if cur <= last:
                    slots = [None] * int(lengths[cur])
                    est = EpisodeDecodeStats(cur, int(lengths[cur]))
    finally:
        if container is not None:
            container.close()


# --------------------------------------------------------------------------------------------------------------------
# HDF5 frame store
# --------------------------------------------------------------------------------------------------------------------


def write_episode_h5(
    path: Path,
    episode_index: int,
    success: bool,
    jpegs: dict[str, Sequence[bytes]],
    extra_attrs: dict | None = None,
) -> int:
    """Write one episode atomically (`*.tmp` then rename). Returns the file size in bytes."""
    lengths = {cam: len(v) for cam, v in jpegs.items()}
    if len(set(lengths.values())) != 1:
        raise DataValidationError(f"episode {episode_index}: cameras have different frame counts {lengths}")
    length = next(iter(lengths.values()))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    dt = h5py.vlen_dtype(np.uint8)
    with h5py.File(tmp, "w") as f:
        for cam, frames in jpegs.items():
            arr = np.empty(length, dtype=object)
            for i, b in enumerate(frames):
                arr[i] = np.frombuffer(b, dtype=np.uint8)
            f.create_dataset(cam, shape=(length,), dtype=dt, data=arr)
        f.attrs["episode_index"] = int(episode_index)
        f.attrs["length"] = int(length)
        f.attrs["success"] = bool(success)
        f.attrs["cameras"] = list(jpegs)
        for k, v in (extra_attrs or {}).items():
            f.attrs[k] = v
    os.replace(tmp, path)
    return path.stat().st_size


def is_episode_complete(path: Path, length: int, cameras: Sequence[str] = CAMERAS) -> bool:
    """True if `path` is a finished episode file with `length` frames for each camera."""
    path = Path(path)
    if not path.is_file():
        return False
    try:
        with h5py.File(path, "r") as f:
            return int(f.attrs.get("length", -1)) == length and all(
                cam in f and len(f[cam]) == length for cam in cameras
            )
    except OSError:
        return False


# --------------------------------------------------------------------------------------------------------------------
# Extraction orchestrator: one decoder process per camera, the parent merges episodes and writes HDF5
# --------------------------------------------------------------------------------------------------------------------


def _camera_worker(
    camera: str, source: str, spans: VideoSpans, episodes: list[int], quality: int, max_retries: int, out_q
) -> None:
    stats = CameraDecodeStats(camera=camera)
    try:
        for e, jpegs, _ in iter_camera_episodes(source, spans, episodes, quality, max_retries=max_retries, stats=stats):
            out_q.put(("episode", e, jpegs))
        out_q.put(("done", None, asdict(stats)))
    except BaseException:  # noqa: BLE001 - forward everything to the parent
        out_q.put(("error", None, traceback.format_exc()))


def extract_frames(
    episodes: Sequence[int] | None = None,
    source: str = "local",
    raw_dir: Path = paths.RAW_DIR,
    cache_dir: Path = paths.CACHE_DIR,
    frames_dir: Path = paths.FRAMES_DIR,
    cameras: Sequence[str] = CAMERAS,
    quality: int = JPEG_QUALITY,
    overwrite: bool = False,
    queue_size: int = 4,
    max_retries: int = 5,
    revision: str = "main",
    worker_timeout_s: float = 900.0,
) -> dict:
    """Decode the requested episodes of every camera into `frames_dir/ep_XXXX.h5`. Returns run statistics.

    `source`: "local" (mp4s under raw_dir), "url" (stream from the HF hub) or a directory containing `videos/...`.
    Writing strategy: each camera process yields complete episodes (JPEG lists) through a bounded queue, the parent
    takes episode e from all three queues, then writes `ep_e.h5.tmp` and renames it. Memory stays bounded to
    `queue_size` episodes per camera and no file is ever left half-written.
    """
    from tqdm import tqdm

    t0 = time.time()
    episodes_df = pd.read_parquet(Path(cache_dir) / "episodes.parquet")
    info_path = Path(raw_dir) / "meta" / "info.json"
    info = json.loads(info_path.read_text()) if info_path.is_file() else None
    raw_eps = load_raw_episodes(raw_dir)
    lengths = episodes_df["length"].to_numpy()
    success = episodes_df["success"].to_numpy()
    requested = list(range(len(episodes_df))) if episodes is None else sorted(set(int(e) for e in episodes))
    frames_dir = Path(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    for stale in frames_dir.glob("*.h5.tmp"):
        stale.unlink()

    done_before = set()
    if not overwrite:
        done_before = {
            e for e in requested if is_episode_complete(episode_file(frames_dir, e), int(lengths[e]), cameras)
        }
    todo = [e for e in requested if e not in done_before]
    result: dict = {
        "source": source,
        "requested_episodes": len(requested),
        "skipped_complete": len(done_before),
        "written_episodes": 0,
        "written_frames": 0,
        "cameras": {},
    }
    if not todo:
        log.info("all %d requested episodes already complete", len(requested))
        result.update(_store_stats(frames_dir, requested, lengths, cameras))
        result["wall_time_s"] = time.time() - t0
        return result

    sources = {}
    for cam in cameras:
        sp = video_spans(raw_eps, cam, info)
        if source == "url":
            sources[cam] = (video_url(sp.video_path, revision=revision), sp)
        else:
            base = Path(raw_dir) if source == "local" else Path(source)
            p = base / sp.video_path
            if not p.is_file():
                raise FileNotFoundError(f"{p} missing; run scripts/01_download.py (full) or use --source url")
            sources[cam] = (str(p), sp)

    ctx = mp.get_context("spawn")
    queues = {cam: ctx.Queue(maxsize=queue_size) for cam in cameras}
    procs = {
        cam: ctx.Process(
            target=_camera_worker,
            args=(cam, src, sp, todo, quality, max_retries, queues[cam]),
            name=f"decode-{cam}",
            daemon=True,
        )
        for cam, (src, sp) in sources.items()
    }
    for p in procs.values():
        p.start()

    def get(cam: str):
        waited = 0.0
        while True:
            try:
                return queues[cam].get(timeout=5.0)
            except queue.Empty:
                waited += 5.0
                if not procs[cam].is_alive():
                    raise RuntimeError(f"decoder process for {cam} died (exit code {procs[cam].exitcode})") from None
                if waited > worker_timeout_s:
                    raise TimeoutError(f"no output from {cam} decoder for {waited:.0f}s") from None

    try:
        written_bytes = 0
        for e in tqdm(todo, desc="episodes", unit="ep"):
            jpegs = {}
            for cam in cameras:
                kind, got, payload = get(cam)
                if kind == "error":
                    raise DataValidationError(f"{cam} decoder failed:\n{payload}")
                if kind != "episode" or got != e:
                    raise RuntimeError(f"{cam}: expected episode {e}, got {kind} {got}")
                jpegs[cam] = payload
            written_bytes += write_episode_h5(
                episode_file(frames_dir, e), e, bool(success[e]), jpegs, {"jpeg_quality": quality}
            )
            result["written_episodes"] += 1
            result["written_frames"] += int(lengths[e])
        for cam in cameras:
            kind, _, payload = get(cam)
            if kind == "error":
                raise DataValidationError(f"{cam} decoder failed:\n{payload}")
            eps = payload.pop("episodes")
            payload["episodes_checked"] = len(eps)
            payload["episodes_exact"] = sum(
                d["hit"] == d["length"] and not (d["missing"] or d["duplicated"] or d["extra"]) for d in eps
            )
            payload["frames_hit"] = sum(d["hit"] for d in eps)
            payload["per_episode_length_hit"] = [[d["episode_index"], d["length"], d["hit"]] for d in eps]
            result["cameras"][cam] = payload
    finally:
        for p in procs.values():
            if p.is_alive():
                p.terminate()
            p.join(timeout=10)
    result["wall_time_s"] = time.time() - t0
    result["written_bytes"] = written_bytes
    result["frames_per_s_per_camera"] = result["written_frames"] / max(result["wall_time_s"], 1e-9)
    result.update(_store_stats(frames_dir, requested, lengths, cameras))
    return result


def episode_file(frames_dir: Path, episode_index: int) -> Path:
    """`frames_dir/ep_XXXX.h5` (same naming as `recap_vf.paths.episode_frames_path`, but for any directory)."""
    return Path(frames_dir) / paths.episode_frames_path(episode_index).name


def _store_stats(frames_dir: Path, episodes: Sequence[int], lengths: np.ndarray, cameras: Sequence[str]) -> dict:
    """Size of the store for `episodes`: total bytes on disk, JPEG bytes per camera and per frame."""
    total_file = 0
    jpeg = dict.fromkeys(cameras, 0)
    n_frames = 0
    for e in episodes:
        p = episode_file(frames_dir, e)
        if not p.is_file():
            continue
        total_file += p.stat().st_size
        n_frames += int(lengths[e])
        with h5py.File(p, "r") as f:
            for cam in cameras:
                jpeg[cam] += int(sum(len(x) for x in f[cam][...]))
    n_img = n_frames * len(cameras)
    return {
        "store_episodes": len([e for e in episodes if episode_file(frames_dir, e).is_file()]),
        "store_frames": n_frames,
        "store_file_bytes": total_file,
        "store_jpeg_bytes_per_camera": jpeg,
        "mean_jpeg_bytes_per_image": (sum(jpeg.values()) / n_img) if n_img else 0.0,
        "mean_file_bytes_per_frame_3cams": (total_file / n_frames) if n_frames else 0.0,
        "free_disk_bytes_after": shutil.disk_usage(frames_dir).free,
    }
