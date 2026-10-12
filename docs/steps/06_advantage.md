# Step 6 — Value inference & advantage labelling

**Depends on:** step 4 (and the checkpoint decision of step 5). The advantage code itself can be written and
unit-tested earlier on synthetic values.
**Owns:** `src/recap_vf/advantage.py`, `scripts/06_infer_values.py`, `scripts/06_compute_advantages.py`,
`configs/advantage.yaml`, `tests/test_advantage.py`, `reports/step_06_advantage.md`, `reports/figures/step_06/`.

## Goal
`V(o_t)` for **every frame of all 800 episodes**, then the N-step advantage `A_t` and the binary indicator
`I_t = 1[A_t ≥ ε]` for every frame — the deliverable of this project.

## 6.1 Value inference — `scripts/06_infer_values.py`
```bash
uv run python scripts/06_infer_values.py --run vf_c2229_s0 --ckpt best                 # single model
uv run python scripts/06_infer_values.py --run vf_c2229_cf --crossfit 5 --ckpt last   # fold k scores fold k
```
- `model.eval()`, `torch.no_grad()`, bf16 autocast, no augmentation, large batch (e.g. 256), all frames
  (`frame_stride=1`), iterate episode by episode (`FrameDataset.get_episode` or an episode-ordered dataset).
- Output `runs/<run>/values.parquet`: `index, episode_index, frame_index, value, value_std, model_id`.
  `model_id` = the checkpoint (and fold) that scored the episode.
- Assert: 634 838 rows, every `index` exactly once; with cross-fitting, every episode scored by the model of its own
  fold.
- Report wall time and frames/s.

## 6.2 Advantage — `src/recap_vf/advantage.py`
Per episode, with `g_t = return_norm` and `V_t = value` (sorted by `frame_index`), lookahead `N` (frames):
```
if t + N < T:  reward_sum_t = g_t − g_{t+N}         (= −N/Z exactly, for successes and failures)
               value_next_t = V_{t+N}
else:          reward_sum_t = g_t                    (window reaches the end: includes terminal reward)
               value_next_t = 0                      (nothing after the episode)
A_t = reward_sum_t + value_next_t − V_t
```
This equals the paper's `Σ_{t'=t}^{t+N−1} r_{t'} + V(o_{t+N}) − V(o_t)` in normalised units (γ = 1), and matches
RLinf's handling of the episode end.

**Threshold.** `ε = quantile(A over all frames, 1 − p_pos)`, `I_t = A_t ≥ ε`. One threshold for the whole dataset
(single task). Defaults: **N = 50 (1 s), p_pos = 0.4** (paper, rollout fine-tuning). Human-correction frames would be
forced to `True` — there are none flagged in this dataset (see [open questions](../open_questions.md)); keep a
`forced_positive` hook (a set of global indices) so they can be added later.

API:
```python
def compute_advantages(values: pd.DataFrame, returns: pd.DataFrame, lookahead: int) -> pd.DataFrame
def threshold(advantages: np.ndarray, p_pos: float) -> float
def label(adv_df: pd.DataFrame, p_pos: float, forced_positive: set[int] | None = None) -> tuple[pd.DataFrame, dict]
```
Vectorise per episode with NumPy (no Python loop over frames).

## 6.3 Script — `scripts/06_compute_advantages.py`
```bash
uv run python scripts/06_compute_advantages.py --run vf_c2229_s0 --returns-tag c2229 --lookahead 50 --p-pos 0.4
```
Writes `runs/<run>/advantages_{N}_{p}.parquet` + `.json` (schema in [conventions](../conventions.md)), with
`epsilon`, `Z`, the positive fraction, and the source values file in the JSON.

Also produce the **sensitivity grid** N ∈ {25, 50, 100} × p_pos ∈ {0.3, 0.4} and report the label agreement of each
setting with the default (fraction of frames with the same `I_t`).

Optional second view for the policy team: **segment labels** with `--segment-len 50`: non-overlapping 50-frame
chunks, label = `I` at the chunk's first frame, plus mean `A` over the chunk → `segments_{N}_{p}.parquet`.

## 6.4 Diagnostics (in the report, figures in `reports/figures/step_06/`)
- ε; histogram of `A` for success-episode frames vs failure-episode frames.
- Fraction of positive frames in success vs failure episodes, and the histogram of per-episode positive fraction by
  outcome. Expect: successes clearly higher than failures, **but neither 100 % nor 0 %** — that within-episode
  variation is the whole point compared with simply labelling by outcome.
- **Value jitter:** std of `V_{t+1} − V_t` vs std of `A`. If frame-to-frame jitter is a large fraction of the spread
  of `A`, labels are noisy; report it. A small temporal smoothing of V (e.g. centred median over 5 frames) is allowed
  as a **flagged option** (`--smooth-values 5`), not the default — report both if used.
- **Label flicker:** mean run length of consecutive identical `I_t` (in frames). Very short runs (≲ 5 frames) mean
  noise.
- Example strips for 4 held-out episodes (2 success, 2 failure): V(t), A(t), I(t) shaded.

## Tests (`tests/test_advantage.py`)
- Perfect value function on a success episode (`V = g`): `A_t = 0` for every t (incl. the tail).
- Constant V on a failure episode: `A` for t + N < T equals `−N/Z`; on the tail equals `g_t − V`.
- Tail boundary exactly at `t + N = T` uses the terminal branch.
- Threshold gives the requested positive fraction (± one frame), ties handled with `≥`.
- `forced_positive` frames are positive regardless of A.

## Definition of done
- [ ] `values.parquet` for all frames with the step-5-approved checkpoint(s)
- [ ] `advantages_50_0.4.parquet` + JSON (the deliverable) and the sensitivity grid
- [ ] All diagnostics in the report with figures
- [ ] Tests pass, ruff clean
- [ ] `reports/step_06_advantage.md` written
