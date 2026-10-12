# Conventions & contracts

Rules every step follows so that independently-built pieces fit together. If a step needs to break a contract, it
must say so in its report and the orchestrator updates this file — never silently diverge.

## Environment
- One environment, managed by **uv** from [`pyproject.toml`](../pyproject.toml). Run everything with `uv run …`.
- Python ≥ 3.11 (the venv uses 3.12). PyTorch wheels from the CUDA 12.8 index (required for RTX 50xx / Blackwell).
- Laptop only: ROS Humble puts Python 3.10 packages on `PYTHONPATH`. Run everything as `env -u PYTHONPATH uv run …`.
- Verified versions (laptop, 2026-10-12): torch 2.11.0+cu128, transformers 5.19.0, av 19.0.1.
- Language backbone: `ValueModelConfig` defaults to `constants.LANGUAGE_BACKBONE_MIRROR` (`unsloth/gemma-3-270m`,
  identical weights to the gated `google/gemma-3-270m`). The laptop HF login now has access to the official repo; the
  workstation has no HF login, so keep the mirror there (or log in and pass `--language-backbone google/gemma-3-270m`).
- Need a dependency that is not in `pyproject.toml`? Add it with `uv add <pkg>` **and** list it in your report.
  Do not run `uv sync`/`uv add` while another agent is running a job in the same `.venv`.

## Machines
| Machine | GPU | Disk | Use for |
|---|---|---|---|
| Laptop (dev) | RTX 4060 Laptop, 8 GB | ~15 GB free | **all development**: code, unit tests, smoke tests on ≤ 10 episodes |
| Workstation `saad@curiosity.local` | RTX 5090, 32 GB, **shared** | check `df -h ~` | **full runs**: frame extraction, training, inference on all 800 episodes |

**Workflow.** Everything is developed and tested on the laptop. Code reaches the workstation through GitHub:
commit on the laptop → push → `git pull` in `~/DreamMachines` on the workstation → run there.
Every script must therefore work at **two scales** with the same code: a laptop smoke scale (a few episodes,
frozen backbone / tiny batch where needed) and the full scale, selected by CLI flags or config — never by editing code.

**Laptop disk budget (~15 GB free):** `.venv` ~7 GB + backbones ~5 GB (HF cache) + a data subset < 1 GB. Never download the
full videos (3 × 1.5 GB) or extract all frames on the laptop. The videos can be **streamed over HTTPS** with PyAV
(`av.open("https://huggingface.co/datasets/…/file-000.mp4")` — the decoder fetches the index at the end of the file,
then reads sequentially), so laptop smoke tests decode just the first episodes. Clean up anything you no longer need.

Full runs can be launched/monitored from the laptop with [`scripts/remote.sh`](../scripts/remote.sh)
(`update`, `run`, `bg`, `log`, `jobs`, `gpu`, `fetch-reports`) once key-based SSH is set up.

**Workstation rules (from its admin — mandatory):**
- **The GPU is shared.** Check `nvidia-smi` (`scripts/remote.sh gpu`) before every GPU job. If another user's process
  holds significant memory or utilisation, **do not start** — report and wait.
- **No sudo.** Everything installs in user space (uv at `~/.local/bin`). If a system package is missing, stop and
  ask the orchestrator (the admin installs it).
- **Data and checkpoints stay in the home directory** (`~/DreamMachines/data`, `~/DreamMachines/runs`).
- Never touch other users' files or processes. Only kill processes you started (their PIDs are in `runs/logs/*.pid`).
- Never read, print or store passwords or tokens.

Everything must stay portable: no hard-coded absolute paths, GPU memory–hungry settings live in configs.

## Paths (see [`src/recap_vf/paths.py`](../src/recap_vf/paths.py))
| Env var | Default | Content |
|---|---|---|
| `RECAP_DATA_ROOT` | `<repo>/data` | `raw/` HF download, `cache/` derived artefacts |
| `RECAP_RUNS_ROOT` | `<repo>/runs` | one sub-directory per training / labelling run |

Both directories are git-ignored. Never write outside them (except `reports/`).

## Code style
- Package code in `src/recap_vf/`, entry points in `scripts/NN_name.py` (NN = step number), configs in `configs/*.yaml`.
- Scripts are thin: parse args / load YAML → call package functions. Logic lives in the package so it is testable.
- Type hints on public functions; short docstrings that say *what* and *why*, not *how*.
- Formatting/linting: `uv run ruff format` and `uv run ruff check` must pass.
- Tests: `tests/test_<module>.py`, run with `uv run pytest`. Tests must run on CPU in < 1 min and must not need the
  full dataset (use tiny synthetic episodes or a fixture written by the test).
- Shared constants (`constants.py`), paths (`paths.py`) and the value support (`bins.py`) are **owned by the
  orchestrator**. Import them; do not redefine `NUM_BINS`, camera names, fps, etc. elsewhere.
- Logging via `logging`/`tqdm`; metrics via TensorBoard (`runs/<run>/tb/`).
- Seeds: every script takes `--seed` (default 0) and records it.

## File contracts between steps

All tabular artefacts are **parquet**, keyed by the dataset's global frame `index` (0 … 634 837) and also carrying
`episode_index` and `frame_index` for readability.

| Artefact | Produced by | Path | Schema |
|---|---|---|---|
| Raw dataset | 1 | `data/raw/20h_fullft_eval_success/` | untouched HF snapshot |
| Frame store | 1 | `data/cache/frames/ep_{episode:04d}.h5` | per episode: datasets `top`, `left_wrist`, `right_wrist` (variable-length `uint8` JPEG bytes, length T); attrs `episode_index`, `length`, `success` |
| Frame table | 1 | `data/cache/frames.parquet` | `index:int64, episode_index:int64, frame_index:int64, timestamp:float32, state:list<float32>[14], action:list<float32>[14]` |
| Episode table | 1 | `data/cache/episodes.parquet` | `episode_index, length, success:bool, dataset_from_index, dataset_to_index` |
| Splits | 1 | `data/cache/splits.json` | `{"seed", "heldout": [ep…], "train": [ep…], "folds": {"0": [ep…], …, "4": [ep…]}}` |
| Returns | 2 | `data/cache/returns_{tag}.parquet` + `returns_{tag}.json` | parquet: `index, episode_index, frame_index, success, episode_length, reward, return_raw, return_norm, bin_index`; json: `{"tag", "c_fail", "Z", "num_bins", "v_min", "v_max", "t_max"}` |
| Checkpoint | 4 | `runs/<run>/checkpoints/{best,last}/` | `model.safetensors` + `config.yaml` (full resolved config incl. returns tag & split) |
| Values | 6 | `runs/<run>/values.parquet` | `index, episode_index, frame_index, value, value_std, model_id` (model_id = checkpoint/fold that scored this episode) |
| Advantages | 6 | `runs/<run>/advantages_{N}_{p}.parquet` + `.json` | parquet: `index, episode_index, frame_index, value, value_next, reward_sum, advantage, indicator:bool`; json: `{"N", "p_pos", "epsilon", "returns_tag", "Z", …}` |

The default returns tag is `c2229` (C_fail = 2229 = longest episode, see [step 2](steps/02_returns.md)).

## Batch contract (dataset → model)

`FrameDataset.__getitem__` (step 1) returns a dict; the training collate (step 4) stacks it:

| Key | Type / shape (batched) | Notes |
|---|---|---|
| `images` | `dict[str, uint8 Tensor[B, 3, 224, 224]]` keyed by `constants.CAMERAS` | raw RGB, **not** normalised — the model normalises |
| `state` | `float32 Tensor[B, 14]` | raw joint positions; the model normalises with stats it is given |
| `index`, `episode_index`, `frame_index` | `int64 Tensor[B]` | bookkeeping |
| `target_bin` | `int64 Tensor[B]` | added by step 4 from the returns table |
| `return_norm` | `float32 Tensor[B]` | added by step 4 |

The model (step 3) consumes `images` + `state` and returns `logits: float32 Tensor[B, 201]`.

## Reports

Each step ends with `reports/step_XX_<name>.md`. Template:

```markdown
# Step XX — <name>
**Status:** done | blocked | partial        **Agent/date:** …
## Summary            (3–5 bullets: what was built, key numbers)
## How to reproduce   (exact commands, in order)
## Results            (tables / plots saved under reports/figures/step_XX/…)
## Deviations from the spec   (and why)
## Open issues / questions for the orchestrator
## Definition-of-done checklist  (copy from the step doc, tick each box)
```
Figures go to `reports/figures/step_XX/` (PNG, small). Large artefacts stay in `data/` / `runs/`.

## Git
- The orchestrator commits and pushes to GitHub. Agents do not commit or push.
- Never commit anything under `data/`, `runs/`, `.venv/`, or checkpoints.
