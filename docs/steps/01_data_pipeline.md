# Step 1 — Data pipeline

**Depends on:** step 0. **Can run in parallel with:** steps 2 and 3.
**Owns:** `src/recap_vf/data/` (`__init__.py`, `frames.py`, `dataset.py`, `splits.py`), `scripts/01_*.py`,
`tests/test_data_*.py`, `reports/step_01_data.md`, `reports/figures/step_01/`.

## Goal
Turn the HF dataset into something a training loop can read **fast and randomly**: every frame of every episode,
three camera images + robot state, addressable by `(episode_index, frame_index)` or global `index`; plus
reproducible train / held-out / fold splits.

## Why this design
- The dataset stores each camera as **one mp4 with all 800 episodes concatenated** (H.264, 50 fps, 224×224,
  634 838 frames per stream). Measured: a **keyframe every 2 frames** (so seeking is cheap) and the `moov` index at
  the **end** of the file (PyAV over HTTPS still works: it fetches the index, then reads sequentially).
- We still decode **once, sequentially**, into a JPEG store: training needs ~3 random frames per sample from many
  dataloader workers, and a JPEG lookup gives predictable, codec-independent throughput with no seek edge cases.
  **Fallback** if the workstation disk is short: a `FrameDataset` backend that seeks directly in the mp4s
  (LeRobot-style); keep the `FrameDataset` interface identical so either backend can be used.
- Fully decoded raw uint8 would be 634 838 × 3 × 224 × 224 × 3 B ≈ **287 GB** → too big. JPEG at quality 95 is
  ~10–15 KB per frame → **~20–30 GB**, and decodes in ~1 ms per image in dataloader workers.
- One HDF5 file per episode keeps the file count small (800 files instead of 1.9 M JPEGs) and allows O(1) access.
- We use **all frames** (no temporal subsampling). Training samples frames uniformly at random and is measured in
  optimisation steps, so redundancy between neighbouring frames costs nothing.

## Two scales (same code)
| | Laptop (dev, ~15 GB free) | Workstation (full) |
|---|---|---|
| Download | `meta/*` + `data/*` only (~60 MB) | full snapshot (5.3 GB) |
| Video source | stream from the HF URL (`--source url`) | local mp4s (`--source local`, default) |
| Episodes | `--episodes 0-9` (decode stops after the last requested episode) | all 800 |
| Output | `data/cache/` with 10 episodes | full store |

## Inputs
- HF dataset `DreamMachines/20h_fullft_eval_success` (`constants.DATASET_REPO_ID`). Layout (LeRobot v3.0):
  ```
  meta/info.json                         fps, features, total_episodes=800, total_frames=634838
  meta/episodes/chunk-000/file-000.parquet   one row / episode: episode_index, length, success,
                                         dataset_from_index, dataset_to_index,
                                         videos/<video_key>/{chunk_index,file_index,from_timestamp,to_timestamp}
  meta/tasks.parquet                     one task
  data/chunk-000/file-000.parquet        one row / frame: timestamp, frame_index, episode_index, index,
                                         task_index, observation.state[14], action[14]
  videos/observation.images.{top,left_wrist,right_wrist}/chunk-000/file-000.mp4
  ```
- `from_timestamp`/`to_timestamp` give each episode's span **inside the concatenated mp4** (e.g. ep 0: 0.0–12.48 s,
  ep 1: 12.48–21.78 s). Frame `i` of episode `e` is at video time `from_timestamp[e] + i / 50`.

## Outputs (contracts — see [conventions](../conventions.md#file-contracts-between-steps))
| Path | Content |
|---|---|
| `data/raw/20h_fullft_eval_success/` | HF snapshot (unmodified) |
| `data/cache/frames/ep_{e:04d}.h5` | datasets `top`, `left_wrist`, `right_wrist`: `h5py.vlen_dtype(np.uint8)`, length `T`, JPEG bytes; attrs `episode_index`, `length`, `success` |
| `data/cache/frames.parquet` | `index, episode_index, frame_index, timestamp, state[14], action[14]` (from the data parquet, sorted by `index`) |
| `data/cache/episodes.parquet` | `episode_index, length, success, dataset_from_index, dataset_to_index` |
| `data/cache/splits.json` | `seed`, `heldout`, `train`, `folds` (see below) |
| `src/recap_vf/data/dataset.py` | `FrameDataset` (below) |

## Tasks

### 1.1 Download — `scripts/01_download.py`
- `huggingface_hub.snapshot_download(repo_id=DATASET_REPO_ID, repo_type="dataset", local_dir=paths.RAW_DIR)`.
  `--meta-only` fetches only `meta/*` + `data/*` (laptop).
- Validate `meta/info.json`: `fps == 50`, `total_episodes == 800`, `total_frames == 634838`, the three video keys,
  `observation.state` shape `[14]`. Fail loudly on mismatch.

### 1.2 Tables — inside `scripts/01_extract_frames.py` (or a separate `01_tables.py`)
- Write `episodes.parquet` from `meta/episodes` (keep the columns listed above, `success` as bool).
- Write `frames.parquet` from the data parquet. Check: `index` is `0…634837` contiguous; for each episode,
  `frame_index` is `0…length−1` and `index ∈ [dataset_from_index, dataset_to_index)`.
- Check that `timestamp == frame_index / 50` (within 1e-4 s).

### 1.3 Frame extraction — `scripts/01_extract_frames.py`
- Decode each camera's mp4 **sequentially** with PyAV (`av.open`, `stream.thread_type = "AUTO"`), one process per
  camera (3 in parallel). For every decoded frame compute its time `pts * time_base`, map it to
  `(episode, frame_index)` using the episode spans: `episode = last e with from_timestamp[e] ≤ t + 1e-4`,
  `frame_index = round((t − from_timestamp[e]) * 50)`.
- **Assert** for each camera: every `(episode, frame_index)` with `frame_index < length` is hit exactly once; report
  any missing / extra / duplicated frames (LeRobot uses a 1e-4 s tolerance; at 50 fps frames are 20 ms apart, so
  rounding is unambiguous). If the decoder yields a frame count per episode that differs from `length`, stop and
  report — do not pad silently.
- Encode each frame as JPEG (quality 95, `PIL.Image.save(..., quality=95)`, or `torchvision.io.encode_jpeg`).
  Do not resize — frames are already 224×224.
- Writing: the three cameras are decoded by different processes, so either (a) each camera process writes its own
  temporary per-episode files that a final pass merges into `ep_XXXX.h5`, or (b) collect a whole episode in memory
  then write — choose what is simplest and robust; document the choice. Make the script **resumable**
  (skip episodes whose `.h5` is complete) and support `--episodes 0-9` for laptop testing.
- Measure and report: wall time, total store size, mean bytes / frame.
- **Do not leave partial files** on crash (write to `*.tmp`, then rename).

### 1.4 Splits — `src/recap_vf/data/splits.py` + `scripts/01_make_splits.py`
- Stratified by `success`, `seed=0`:
  - `heldout`: 15 % of episodes (→ 120: ~63 success / ~57 failure). Never used for training in the default run.
  - `train`: the remaining 680.
  - `folds`: all 800 episodes partitioned into 5 stratified folds `"0"…"4"` (currently unused: cross-fitting is out of scope). Independent of `heldout`.
- Also stratify on episode length (e.g. bin lengths into quartiles within each success class) so held-out is
  representative. Print the success rate and length quartiles of each split.
- `load_splits()` helper returning the dict.

### 1.5 `FrameDataset` — `src/recap_vf/data/dataset.py`
```python
class FrameDataset(torch.utils.data.Dataset):
    def __init__(self, episodes: Sequence[int], cameras=CAMERAS, frame_stride: int = 1): ...
    def __len__(self) -> int: ...                 # number of (episode, frame) pairs selected
    def __getitem__(self, i: int) -> dict: ...    # batch contract in docs/conventions.md
    @property
    def index(self) -> np.ndarray: ...            # global `index` of every item, in item order
```
- Items are **all frames** of the given episodes (`frame_stride` only for evaluation subsampling).
- Returns `images: {cam: uint8 Tensor[3,224,224]}`, `state: float32[14]`, `index`, `episode_index`, `frame_index`.
- **HDF5 + multiprocessing:** open files lazily *inside the worker* (e.g. a per-process dict of open handles created on
  first access; never pass an open `h5py.File` across `fork`). Test with `num_workers=4`.
- JPEG decode with `torchvision.io.decode_jpeg` (CPU) or PIL; benchmark both and pick the faster.
- Also provide `get_episode(e) -> dict` returning all frames of one episode (used by inference / plots).
- Benchmark and report: items / s with `batch_size=32, num_workers ∈ {4, 8, 12}`. Target: > 1 500 items/s on the
  office machine (training will consume ~100–300/s; we want I/O to never be the bottleneck).

### 1.6 Visual sanity check — `scripts/01_sanity_figures.py`
- For 6 episodes (3 success, 3 failure): a grid of the three cameras at frame 0, 25 %, 50 %, 75 %, last frame →
  `reports/figures/step_01/ep_XXXX_grid.png`. Look at them (you can open PNGs) and describe what you see in the
  report: the last frame of a success should show the actuator in the tray.
- Alignment check video ↔ state: plot left/right gripper state over time for one episode and mark 3 timestamps where
  the gripper value changes sharply; save the wrist-camera frames at those times. The gripper visibly opening/closing
  at the same time confirms alignment.

## Tests (`tests/test_data_*.py`, CPU, synthetic)
- Build a tiny synthetic episode store (2 episodes, 5 frames, random images) in `tmp_path`; check `FrameDataset`
  returns correct shapes/dtypes, `index` ordering, and JPEG round-trip error is small.
- Splits: disjointness, coverage (folds partition all episodes), stratification within ±1 episode, determinism.
- Timestamp→(episode, frame) mapping on hand-made spans, including boundaries.

## Pitfalls
- `success` lives in `meta/episodes`, not in the frame table.
- Global `index` is the join key for everything downstream — keep it.
- Don't trust `from_timestamp` floats blindly: compare `to_timestamp − from_timestamp` with `length / 50`.
- The laptop has ~15 GB free: never run the full extraction or download the full videos there.
- Streaming from a URL: open the container once per camera and decode sequentially; stop as soon as the requested
  episodes are done. Retry on transient HTTP errors.

## Definition of done
- [ ] Laptop smoke scale works end-to-end (`--meta-only`, `--source url --episodes 0-9`), commands in the report
- [ ] Full-scale commands documented for the workstation (run them there if available; otherwise mark as pending)
- [ ] Raw dataset downloaded and validated against `info.json`
- [ ] (full scale) 800 `.h5` files, every camera has exactly `length` frames for every episode (assert passed, numbers in report)
- [ ] `frames.parquet`, `episodes.parquet`, `splits.json` written and validated
- [ ] `FrameDataset` benchmarked with ≥ 4 workers; throughput reported
- [ ] Sanity grids + alignment figure in `reports/figures/step_01/`, described in the report
- [ ] Store size, extraction time reported
- [ ] Tests pass, ruff clean
- [ ] `reports/step_01_data.md` written
