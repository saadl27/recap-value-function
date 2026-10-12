# Step 01 — Data pipeline
**Status:** partial (laptop smoke scale done and validated; full-scale extraction pending on the workstation)
**Agent/date:** Claude (step-1 agent), 2026-10-12

## Summary
- Built `recap_vf.data`: raw-dataset validation, `episodes.parquet` / `frames.parquet` (all 800 episodes / 634 838
  frames, validated), a resumable mp4 → per-episode JPEG/HDF5 frame store (`ep_XXXX.h5`, contract layout),
  stratified `splits.json` and `FrameDataset` (lazy per-worker HDF5 handles, fork- and spawn-safe).
- Laptop smoke run (`--source url --episodes 0-9`, streamed from the HF hub, no video download): 10 episodes /
  6 786 frames / 3 cameras extracted in **6.5 s**. Every camera hit **exactly `length` frames for each of the 10
  episodes**: 0 missing, 0 duplicated, 0 extra, max timestamp residual 3e-14 s. Seek-based and sequential decoding
  give byte-identical JPEGs.
- Store size at JPEG q95: **20.6 KB per image, 62.3 KB per frame (3 cams, incl. ~0.6 KB HDF5 overhead)**.
  Projected full store is **≈ 38 GB (37.7–39.5 GB)**, above the spec's 20–30 GB estimate. `--quality 90` gives
  ≈ 27 GB (see open issue 1).
- Decode + encode runs at **≈ 2 400 frames/s per camera with the 3 cameras in parallel** (steady state, laptop,
  16 threads). Projected full extraction: **≈ 5–15 min** (decode/encode alone ≈ 4.5 min; HDF5 write speed is the
  unknown).
- `FrameDataset` (batch 32, shuffle, torchvision decode): **2 133 / 3 100 / 3 548 items/s with 4 / 8 / 12
  workers**. Each item has 3 images, and every setting beats the 1 500 items/s target. Measured on a warm page cache
  with a 0.4 GB store, so it must be re-measured on the 38 GB store.
- Visual checks pass: grids, gripper ↔ wrist-camera alignment, and success labels match what the tray shows. The
  tray also shows something unexpected about how **failed episodes end** (see Results §4 and open issue 2).

## How to reproduce

### Laptop (smoke scale; this is what was run)
```bash
cd ~/Documents/DreamMachines
env -u PYTHONPATH uv run python scripts/01_download.py --meta-only            # meta/* + data/*, skips existing files, validates info.json
env -u PYTHONPATH uv run python scripts/01_extract_frames.py --source url --episodes 0-9   # tables (all eps) + store for eps 0-9
env -u PYTHONPATH uv run python scripts/01_make_splits.py --seed 0
env -u PYTHONPATH uv run python scripts/01_benchmark_dataset.py --episodes 0-9 --workers 4 8 12
env -u PYTHONPATH uv run python scripts/01_sanity_figures.py --episodes 0-9
env -u PYTHONPATH uv run pytest -q tests/test_data_*.py
```
Optional: `scripts/01_extract_frames.py --tables-only` rebuilds only the two parquet tables (0.5 s).
Each run writes a JSON record to `data/cache/logs/` (`01_download.json`, `01_extract_<time>.json`,
`01_benchmark_<time>.json`).

### Workstation (full scale) — pending, not run
```bash
cd ~/DreamMachines && git pull
df -h ~        # need ≈ 45 GB: 5.3 GB raw snapshot + ≈ 38 GB store at q95 (≈ 27 GB with --quality 90)
uv run python scripts/01_download.py                                          # full snapshot (5.3 GB), validates info.json + 634838 frames per mp4
uv run python scripts/01_extract_frames.py --source local --episodes all      # resumable: re-run the same command after a crash
uv run python scripts/01_make_splits.py --seed 0                              # identical to the laptop splits.json (deterministic)
uv run python scripts/01_benchmark_dataset.py --episodes all --workers 4 8 12 16
uv run python scripts/01_sanity_figures.py --episodes all
```
From the laptop: `scripts/remote.sh update`, then
`scripts/remote.sh bg extract01 "uv run python scripts/01_extract_frames.py --source local --episodes all"` and
`scripts/remote.sh log extract01`. Extraction is CPU and disk only (no GPU). The `extract` record in
`data/cache/logs/01_extract_*.json` holds the per-camera `episodes_exact` (should be 800) and `frames_hit` (should be
634838) numbers needed for the DoD.

## Results

### 1. Raw dataset and tables
| Check | Result |
|---|---|
| `meta/info.json` | fps 50, 800 episodes, 634 838 frames, 3 video keys 224×224×3, state/action `[14]` — OK |
| HF snapshot sha | `38aa17046bff033b9f7812288e48f7a6adb23655` (laptop: meta + data only, 58 MB) |
| `episodes.parquet` | 800 rows, 423 success; lengths match `dataset_to − dataset_from`; episodes contiguous in the global index |
| `frames.parquet` | 634 838 rows, `index` = 0…634837, `frame_index` = 0…length−1 for each episode, 59 MB; `state`/`action` are `fixed_size_list<float32>[14]` (same type as raw) |
| `timestamp` vs `frame_index/50` | max error 1.8e-6 s (float32 rounding) |
| Video spans (all 3 cameras) | identical across cameras; `to − from` = `length/50` within 2e-12 s; **no gaps** between consecutive episodes; last episode ends at 12 696.76 s = 634 838 / 50 |

### 2. Frame store (smoke: episodes 0–9)
| | top | left_wrist | right_wrist | total |
|---|---|---|---|---|
| frames hit / expected | 6 786 / 6 786 | 6 786 / 6 786 | 6 786 / 6 786 | |
| episodes exact (hit == length, 0 missing/dup/extra) | 10 / 10 | 10 / 10 | 10 / 10 | |
| decoded frames | 6 787 | 6 787 | 6 787 | (+1 = first frame of ep 10, which stops the decoder) |
| JPEG bytes / image (q95) | 16.9 KB | 23.4 KB | 21.4 KB | **20.6 KB** |

- Per-episode hits (identical for the 3 cameras): ep0 624/624, ep1 465/465, ep2 230/230, ep3 280/280, ep4 688/688,
  ep5 434/434, ep6 223/223, ep7 512/512, ep8 1119/1119, ep9 2211/2211.
- Store for eps 0–9: 422 MB on disk (62 256 B/frame); wall time 6.5 s, of which ≈ 4–5 s is opening the three HTTPS
  streams (the moov atom is at the end of each 1.5 GB file). Peak RSS of the parent process is 0.94 GB.
- Extra checks: episodes 400, 401, 403 and 799 extracted via URL with seeks (3 seeks, 0–2 pre-roll frames skipped,
  ep 799 runs to end-of-stream), all exact. Episode 5 extracted alone via seek is **byte-identical** to the
  sequential extraction. Re-running `--episodes 0-9` skips all 10 in 0.2 s (resume works).
- Stream facts confirmed: time_base 1/12800 (256 ticks per frame), a keyframe every 2 frames, yuv420p 224×224.

**Projection to the full store (634 838 frames).** Mean JPEG size over eps 0–39 (eps 10–39 measured without
writing) is 19.8 KB/image at q95, i.e. ≈ 60.1 KB/frame with HDF5 overhead:

| JPEG quality | KB / image (eps 10–39) | projected store |
|---|---|---|
| 95 (default, spec) | 19.6 | **≈ 38 GB** (37.7–39.5 GB depending on episode mix) |
| 90 | 14.2 | ≈ 27 GB |
| 85 | 11.6 | ≈ 23 GB |

Decode + encode rate is the same at every quality: **≈ 2 400 frames/s per camera**, 3 cameras in parallel, over
18 606 frames (eps 10–39). Full extraction = 634 838 / 2 400 ≈ 4.5 min, plus HDF5 writes of ≈ 140 MB/s, so
**≈ 5–15 min** expected on a machine like this laptop. The workstation's CPU and disk may change this.

### 3. FrameDataset benchmark (laptop, 16 threads, store = eps 0–9, warm page cache)
| Setting | items/s (1 item = 3 images + state) |
|---|---|
| main process, 0 workers | 712 |
| batch 32, 4 workers | **2 133** |
| batch 32, 8 workers | **3 100** |
| batch 32, 12 workers | **3 548** |
| 12 workers, handle cache forced to 1 file (emulates 800 episodes) | 3 558 (file opens cost nothing measurable) |

JPEG decode: torchvision 0.17 ms/image vs PIL 0.26–0.36 ms/image, so **torchvision is the default**
(`decoder="pil"` remains available). Caveat: the 38 GB full store will not fit in the page cache of most machines,
so random reads will hit the disk. Re-run the benchmark on the workstation.

### 4. Visual sanity checks (`reports/figures/step_01/`)
Episodes 0–9 contain 4 successes (1, 4, 5, 7) and 6 failures (0, 2, 3, 6, 8, 9). Grids: `ep_0001/0004/0005_grid.png`
(success), `ep_0000/0002/0003_grid.png` (failure). What I saw:
- **Scene.** The top camera shows the box of silver actuators (left, in plastic wrap) and a black tray with a grid of
  slots (right). The task is a handover. The left arm picks an actuator out of the box (left wrist, 25 % column),
  carries it over the tray, and hands it to the right gripper. The right arm then inserts it into the tray's front
  row (right wrist, 50–75 % columns: a row of white actuator tops in the slots).
- **Success episodes (1, 4, 5).** At the last frame both arms are back in their home pose and both wrist cameras show
  their resting view (the box for the left, the empty gripper above the tray for the right). The top camera's last
  frame shows **one more actuator in the tray's front row than at frame 0**. `tray_first_last.png` zooms on that
  row: ep1 1→2, ep4 4→5, ep5 5→6, ep7 7→8.
- **Failure episodes.** ep0: the left arm goes into the box at ~3 s and stays there. Frames from 3.1 s to 12.5 s are
  almost identical (the gripper is pressed into the actuators; the right arm hovers over the box). It looks like a
  stall or failed grasp. ep2 (4.6 s) and ep3 (5.6 s) are short and **end mid-motion**: both arms are out, around the
  handover (ep3's last frame shows the right wrist above the tray). The tray count does not change in any failure
  (ep2 2→2, ep3 3→3, ep6 6→6, ep8 8→8). In ep9 (2 211 frames, close to the maximum length) the last frame shows the
  front row disturbed: 4 actuators, a gap, then a few more next to the arm.
- **The tray count at the start of episodes 0, 1, …, 9 is exactly 0, 1, …, 9.** So an actuator also lands in the tray
  after each failed episode, but outside the recorded frames (open issue 2).
- **Alignment** (`alignment_ep_0001.png`). Gripper state is ≈ 0 when open and ≈ 0.87 when closed on an actuator.
  The three sharpest changes are: E1 at frame 72 (left closes), E2 at frame 194 (left opens), E3 at frame 334 (right
  opens). The right gripper closes at ≈ 3.0–3.8 s, before the left opens at 3.9 s, which is the handover. In the
  wrist frames the fingers move with the state. In E1 they close around the actuator as g goes 0.40 → 0.87. In E3
  they stay tight on the actuator until f329 (g 0.86), start spreading at f334 (g 0.66), and are wide open by f349
  (g 0.16), with the actuator left in its slot. I also checked a finer 5-frame strip, not saved. Video and state
  agree to within ≈ 5 frames (0.1 s), which was the sampling resolution of the check. No offset is visible.

### 5. Splits (`data/cache/splits.json`, seed 0)
Strata: success × length quartile (within each success class). Held-out uses largest-remainder allocation. The 5
folds are dealt round-robin within strata, independently of held-out.

| split | episodes | success | rate | len q25 | median | q75 | frames |
|---|---|---|---|---|---|---|---|
| all | 800 | 423 | 0.529 | 490 | 623 | 918 | 634 838 |
| train | 680 | 360 | 0.529 | 490 | 624.5 | 936 | 543 030 |
| heldout | 120 | 63 | 0.525 | 484 | 612 | 847 | 91 808 |
| fold0–4 | 160 each | 84, 84, 85, 85, 85 | 0.525–0.531 | 484–498 | 607–641.5 | 878–963 | 124 307–128 896 |

Note for smoke tests: episode 0 is in `heldout`.

### 6. Tests
`tests/test_data_frames.py` (8), `tests/test_data_dataset.py` (8), `tests/test_data_splits.py` (4): 20 tests in about
5 s on CPU. Besides the spec's cases, they cover an **end-to-end extraction on a synthetic libx264 mp4** (3 concatenated
episodes; grey level encodes the frame number; covers seek, end of stream and resume) and **refusing to write an
episode with missing frames**. Full suite: `env -u PYTHONPATH uv run pytest -q` gives 43 passed.
`env -u PYTHONPATH uv run ruff check .` reports all checks passed; my files are `ruff format`-clean.

## Deviations from the spec
1. **Seek per contiguous run, not decode from t = 0.** Each camera seeks to the first requested episode of each
   contiguous run, then decodes sequentially. This is needed for resume and subsets (e.g. `--episodes 400-409`).
   With `--episodes all` it is one sequential pass. It was verified byte-identical to sequential decoding.
2. **Writing strategy.** Each camera process sends whole episodes (lists of JPEG bytes) through a bounded queue (4
   episodes). The parent takes episode *e* from all three queues and writes `ep_e.h5.tmp`, then renames it. This is
   the spec's option (b), with memory bounded to a few episodes per camera. Stale `*.tmp` files are deleted at start.
3. **"Extra" frames are fatal**, like missing and duplicated ones (strict; none were found).
4. **HTTPS streaming needs a CA bundle.** In this environment the PyAV wheel's ffmpeg failed TLS verification
   ("Peer certificate failed verification") unless given `ca_file`. `open_video()` finds the system bundle
   automatically (SSL_CERT_FILE, then ssl defaults, then common paths) and sets ffmpeg's reconnect options. On top of
   that, Python-level retries reopen the stream and resume after the last frame seen.
5. **`FrameDataset.__init__`** keeps the spec's signature and adds keyword-only `frames_dir`, `frame_table`,
   `decoder`, `max_open_files`. Items also expose `episode_index` / `frame_index` arrays.
   `get_episode(e, frames=None, cameras=None)` can fetch a subset of frames or cameras, because a whole episode
   decoded is up to ≈ 0.3 GB per camera.
6. **Tables live in `01_extract_frames.py`** (rebuilt on every run, 0.5 s; `--tables-only` / `--skip-tables`). They
   always cover all 800 episodes, even when the store holds only a subset.
7. **Extra keys** in `splits.json` (`heldout_frac`, `n_length_bins`, `stratified_on`) and extra HDF5 attrs
   (`cameras`, `jpeg_quality`). The contract keys are unchanged.
8. **Extra script and figure:** `scripts/01_benchmark_dataset.py` and `reports/figures/step_01/tray_first_last.png`.
9. **Only 3 + 3 grids, from episodes 0–9**, so failures 6, 8 and 9 have no grid. They appear in the tray figure.

## Open issues / questions for the orchestrator
1. **Disk: the full store is ≈ 38 GB at q95, not 20–30 GB.** With the 5.3 GB raw snapshot that is ≈ 44 GB, right at
   the "≥ 45 GB free" assumption. Options if the workstation is short:
   - `--quality 90` (≈ 27 GB; visually fine at 224 px, but it changes the training input slightly);
   - delete `data/raw/.../videos` after extraction;
   - implement the mp4-seek `FrameDataset` backend (not done).

   Please check `df -h ~` on the workstation and pick one. The quality is recorded in each file's attrs.
2. **How failed episodes end (open questions 1–2).** In eps 0–9 every episode, failures included, is followed by
   exactly one more actuator in the tray. Failed episodes never show that placement and often end mid-motion with the
   arms out, while successes end at the home pose. This fits failures being cut where a human (or a reset) took over,
   with that segment not exported (`interventions=false`). It is seen on 10 episodes only; please confirm with
   DreamMachines. Consequences:
   - (a) the end-of-episode pose is a strong visual cue for success vs failure, which matters for the
     memorisation check (step 5) and for what V learns near episode ends (step 7);
   - (b) the "−C_fail at the last frame" reward is applied at the cut point.
3. **DataLoader start method.** The dataset is safe under `fork` (handles reset per PID) and `spawn` (handles never
   pickled). Both are tested. Python 3.12 warns about `fork()` in a multi-threaded process (torch), so step 4 may
   prefer `multiprocessing_context="forkserver"` if workers ever hang.
4. Once, a spawned process of a scratch benchmark crashed while **importing torch**
   (`c10 ApproximateClock ... getCount is non-monotonic`), probably CPU contention from the parallel agents. Since
   then `recap_vf.data` imports lazily, so the decoder processes never import torch. The extractor also checks
   decoder liveness and stops with an error instead of hanging.
5. The 1 500 items/s target is met on a warm cache. Re-measure on the workstation against the full store
   (cold-cache random reads).
6. URL streaming uses `--revision main`. For strict reproducibility, pin
   `--revision 38aa17046bff033b9f7812288e48f7a6adb23655`.
7. No new dependencies. `pyproject.toml` and `uv.lock` are untouched.

## Definition-of-done checklist
- [x] Laptop smoke scale works end-to-end (`--meta-only`, `--source url --episodes 0-9`), commands in the report
- [x] Full-scale commands documented for the workstation — **run: pending (workstation)**
- [~] Raw dataset downloaded and validated against `info.json`: meta + data downloaded and validated on the laptop;
      full snapshot including videos **pending (workstation)**. `01_download.py` also checks each mp4 has 634 838
      frames.
- [ ] (full scale) 800 `.h5` files, every camera has exactly `length` frames for every episode — **pending
      (workstation)**. The smoke scale passes: 10/10 episodes exact on all 3 cameras, plus episodes 400, 401, 403
      and 799.
- [x] `frames.parquet`, `episodes.parquet`, `splits.json` written and validated (all 800 episodes)
- [x] `FrameDataset` benchmarked with ≥ 4 workers; throughput reported (laptop; re-run on the workstation)
- [x] Sanity grids + alignment figure in `reports/figures/step_01/`, described in the report
- [x] Store size, extraction time reported (measured on 10 episodes, projected to 800)
- [x] Tests pass, ruff clean
- [x] `reports/step_01_data.md` written

### Files
`src/recap_vf/data/{__init__,frames,dataset,splits}.py`;
`scripts/01_{download,extract_frames,make_splits,benchmark_dataset,sanity_figures}.py`;
`tests/test_data_{frames,dataset,splits}.py`;
`reports/figures/step_01/{ep_0000..0005_grid,alignment_ep_0001,tray_first_last}.png` (2.2 MB, palette PNGs).
Laptop disk used by step 1: ≈ 0.5 GB (`data/cache/frames` 403 MB + tables 63 MB). Scratch files are deleted.
