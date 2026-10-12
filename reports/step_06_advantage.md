# Step 06 — Value inference & advantage labelling
**Status:** partial (code complete and validated on fake values; real values pending steps 4/5)        **Agent/date:** step-6 agent, 2026-10-12

## Summary
- `src/recap_vf/advantage.py`: `compute_advantages` (fully vectorised over the index-sorted table, joined on global
  `index`, validates complete/contiguous episodes), `threshold` (order-statistic quantile: exact positive fraction,
  ties with `>=`), `label` (one dataset-wide ε, `forced_positive` hook, **graceful degenerate-threshold detection**),
  plus `smooth_values` (flagged median filter), `segment_labels`, `run_lengths`, `label_agreement`, `diagnostics`.
  31 unit tests in `tests/test_advantage.py` cover every test listed in the spec plus edge cases.
- `scripts/06_compute_advantages.py`: full CLI with the N ∈ {25, 50, 100} × p ∈ {0.3, 0.4} grid and label agreement,
  `--segment-len 50`, `--smooth-values K`, all diagnostics figures. **~4 s for all 634 838 frames × 6 settings** (CPU).
- `scripts/06_fake_values.py`: contract-schema `values.parquet` from the returns table, modes `oracle`, `noisy`,
  `lagged`, `smoothed-outcome` (+ optional iid / AR(1)-correlated noise on any mode), `model_id = fake:<mode>`.
  Eight fake runs (`runs/dev_fake_*`) were labelled end to end on all 634 838 frames.
- `scripts/06_infer_values.py`: implemented against the real `ValueModel` (`from_pretrained`, `predict`) and
  `FrameDataset`; `--crossfit K`, `--episodes`, `--batch-size`, bf16 autocast, row/index/fold assertions.
  **Smoke-tested** with a tiny random model on the local 10-episode store (CPU, plus one episode on GPU in bf16),
  including the cross-fit path and a `save_pretrained` → `from_pretrained` checkpoint.
- Key fake-value finding: **the oracle gives |A| ≤ 6e-8 everywhere** (= float32 round-off), which is correctly reported
  as a degenerate threshold (100 % positive + warning) rather than labels drawn from round-off noise.

## How to reproduce
```bash
# unit tests + lint
env -u PYTHONPATH uv run pytest -q tests/test_advantage.py
env -u PYTHONPATH uv run ruff check .

# fake values (dev only; ~2 s each, all 634 838 frames)
F="env -u PYTHONPATH uv run python scripts/06_fake_values.py"
$F --mode oracle                                   # -> runs/dev_fake_oracle
$F --mode noisy --sigma 0.002                      # -> runs/dev_fake_noisy_s0.002
$F --mode noisy --sigma 0.01                       # -> runs/dev_fake_noisy_s0.01
$F --mode noisy --sigma 0.01 --noise-corr 25 --run dev_fake_noisy_s0.01_corr25
$F --mode lagged                                   # -> runs/dev_fake_lagged (lag 100, ramp 25 frames)
$F --mode lagged --sigma 0.005 --noise-corr 10 --run dev_fake_lagged_s0.005_corr10
$F --mode smoothed-outcome                         # -> runs/dev_fake_smoothed-outcome (gamma 1)
$F --mode smoothed-outcome --sigma 0.005 --noise-corr 10 --run dev_fake_smoothed_s0.005_corr10

# labelling + sensitivity grid + segments + smoothed variant + figures (~4–7 s per run)
for r in dev_fake_oracle dev_fake_noisy_s0.002 dev_fake_noisy_s0.01 dev_fake_noisy_s0.01_corr25 dev_fake_lagged \
         dev_fake_lagged_s0.005_corr10 dev_fake_smoothed-outcome dev_fake_smoothed_s0.005_corr10; do
  env -u PYTHONPATH uv run python scripts/06_compute_advantages.py --run $r --segment-len 50 --smooth-values 5
done

# inference smoke tests (tiny random model; local frame store has episodes 0–9 only)
I="env -u PYTHONPATH uv run python scripts/06_infer_values.py"
$I --run dev_infer_tiny --tiny-random-model --episodes 0-9 --device cpu --batch-size 32 --num-workers 2
$I --run dev_infer_tiny_cf --crossfit 5 --tiny-random-model --episodes 1,2,6,8 --device cpu --batch-size 32 --num-workers 2
$I --run dev_infer_tiny_gpu --tiny-random-model --episodes 2 --device cuda --batch-size 32 --num-workers 2
# checkpoint path: a tiny model saved with ValueModel.save_pretrained to runs/dev_infer_ckpt/checkpoints/best, then
$I --run dev_infer_ckpt --ckpt best --episodes 2,6 --device cpu --batch-size 64 --num-workers 2
env -u PYTHONPATH uv run python scripts/06_compute_advantages.py --run dev_infer_tiny --segment-len 50  # subset run

# REAL RUN (workstation, after steps 4/5) — use the arguments step 5 prescribes, e.g.
env -u PYTHONPATH uv run python scripts/06_infer_values.py --run vf_c2229_s0 --ckpt best          # or:
env -u PYTHONPATH uv run python scripts/06_infer_values.py --run vf_c2229_cf --crossfit 5 --ckpt last
env -u PYTHONPATH uv run python scripts/06_compute_advantages.py --run vf_c2229_s0 --segment-len 50 [--smooth-values 5]
```

## Outputs (contract)
| File | Content |
|---|---|
| `runs/<run>/values.parquet` (+ `values.json`) | `index, episode_index, frame_index, value, value_std, model_id`; JSON = timings, device, ckpts (sidecar, not in contract) |
| `runs/<run>/advantages_{N}_{p}.parquet` + `.json` | `index, episode_index, frame_index, value (f32), value_next (f32), reward_sum (f64), advantage (f64), indicator (bool)`; JSON: `N, p_pos, epsilon, returns_tag, Z, c_fail, values_file` (repo-relative), `model_ids, num_frames, num_episodes, positive_fraction(_success/_failure_frames), smooth_values, threshold{…, degenerate, warnings}, diagnostics{…}` |
| `runs/<run>/advantages_{N}_{p}_smooth{K}.*` | same, computed from median-smoothed V (only with `--smooth-values K`) |
| `runs/<run>/segments_{N}_{p}.parquet` | `index (first frame), episode_index, segment_index, frame_start, frame_end, num_frames, indicator (= I at first frame), advantage_mean, frame_positive_fraction` |
| `runs/<run>/sensitivity.json` | grid rows: N, p_pos, ε, positive fractions, agreement vs default, mean run length, degenerate |
| `reports/figures/step_06/<run>/` | `adv_hist.png`, `episode_pos_frac.png`, `examples.png` (+ `_smooth5`), `summary.json` |

`advantage` is stored in float64 so that `advantage >= epsilon − tie_atol` reproduces `indicator` exactly from the file.

## Results — validated on FAKE values only
All numbers below are from `runs/dev_fake_*` (all 800 episodes, 634 838 frames, returns `c2229`, Z = 4457, so one
frame of progress = 1/Z = 2.2e-4 and the N = 50 reward sum is −N/Z = −0.0112). They validate the **code and the
diagnostics**, not the value function.

### Default labels (N = 50, p_pos = 0.4)
| fake run | V variant | ε | pos. frac. success frames | pos. frac. failure frames | episodes all-negative S / F | std ΔV / std A | robust ratio | mean / median run (frames) | frames in runs ≤ 5 | degenerate |
|---|---|---|---|---|---|---|---|---|---|---|
| oracle | raw | 1.1e-9 | 1.000 | 1.000 | 0 / 0 | — | — | 794 / 623 | 0.00 | **yes** (zero spread) |
| noisy σ = 0.002 | raw | 6.9e-4 | 0.399 | 0.401 | 0 / 0 | 1.02 | 1.03 | 2.1 / 1 | 0.85 | no |
| noisy σ = 0.002 | median-5 | 3.7e-4 | 0.400 | 0.401 | 0 / 0 | 0.56 | 0.20 | 4.6 / 3 | 0.36 | no |
| noisy σ = 0.01 | raw | 3.4e-3 | 0.397 | 0.403 | 0 / 0 | 1.02 | 1.03 | 2.1 / 1 | 0.84 | no |
| noisy σ = 0.01 | median-5 | 1.8e-3 | 0.398 | 0.402 | 0 / 0 | 0.56 | 0.17 | 4.7 / 3 | 0.34 | no |
| noisy σ = 0.01, τ = 25 | raw | 3.2e-3 | 0.398 | 0.402 | 0 / 0 | 0.22 | 0.22 | 10.6 / 3 | 0.12 | no |
| noisy σ = 0.01, τ = 25 | median-5 | 3.1e-3 | 0.398 | 0.401 | 0 / 0 | 0.13 | 0.06 | 20.1 / 11 | 0.05 | no |
| lagged | raw | 1.1e-9 | 1.000 | 0.921 | 0 / 0 | 0.03 | — | 409 / 148 | 0.00 | **yes** (606 940 ties) |
| lagged + σ 0.005, τ 10 | raw | 1.4e-3 | 0.417 | 0.386 | 0 / 0 | 0.04 | 0.31 | 7.5 / 3 | 0.18 | no |
| lagged + σ 0.005, τ 10 | median-5 | 1.3e-3 | 0.417 | 0.386 | 0 / 0 | 0.03 | 0.08 | 14.5 / 8 | 0.07 | no |
| smoothed-outcome | raw | 8.4e-3 | 0.900 | 0.000 | 11 / **377** | 0.02 | 0.02 | 524 / 471 | 0.00 | no |
| smoothed-outcome + σ 0.005, τ 10 | raw | 6.0e-3 | 0.876 | 0.018 | 0 / **185** | 0.12 | 0.10 | 31 / 3 | 0.04 | no |
| smoothed-outcome + σ 0.005, τ 10 | median-5 | 6.1e-3 | 0.882 | 0.014 | 0 / 213 | 0.07 | 0.03 | 64 / 9 | 0.02 | no |

"Robust ratio" = 1.4826·MAD(V_{t+1} − V_t) / (IQR(A)/1.349); σ is in normalised value units; τ is the AR(1)
correlation time in frames. Agreement of default labels raw vs median-5: 0.66 (iid noise), 0.95 (τ = 25), 0.93
(lagged + noise), 0.98 (smoothed-outcome + noise).

### Sensitivity grid (label agreement with N = 50, p = 0.4)
| fake run | N=25 p=.3 | N=25 p=.4 | N=50 p=.3 | N=100 p=.3 | N=100 p=.4 |
|---|---|---|---|---|---|
| noisy σ = 0.01 (iid) | 0.694 | 0.693 | 0.900 | 0.701 | 0.706 |
| noisy σ = 0.01, τ = 25 | 0.720 | 0.720 | 0.900 | 0.710 | 0.715 |
| lagged + noise | 0.702 | 0.703 | 0.900 | 0.706 | 0.710 |
| smoothed-outcome | 0.896 | 0.988 | 0.900 | 0.899 | 0.977 |
| smoothed-outcome + noise | 0.844 | 0.861 | 0.900 | 0.884 | 0.941 |

Full grids (ε, positive fractions, run lengths) are in `runs/dev_fake_*/sensitivity.json` and the figures' `summary.json`.

### What the fake runs tell us (to keep in mind when reading the real numbers)
1. **Oracle → A ≡ 0 → degenerate.** The formula is exact (max |A| = 5.8e-8, float32 round-off of V); the script says
   "advantage has (near-)zero spread … 100 % positive instead of 40 %" and sets `degenerate: true` in the JSON.
   Ties within `tie_atol = 1e-6` are positive, so round-off never picks the labels. A **memorising** value function
   (V ≈ g on train episodes) would land here: A ≈ 0 plus residual noise → noise labels. Step 5 matters.
2. **Pure noise on a perfect V gives the same labels whatever σ is.** The threshold rescales with the noise, so
   σ = 0.002 and σ = 0.01 give identical diagnostics (outcome-blind 40 %/40 %, mean run ≈ 2.1 frames — the value
   expected for independent coin flips, 85 % of frames in runs ≤ 5 frames, std ΔV / std A ≈ 1). What matters is how
   big the noise is **relative to the signal**, not its absolute size. These jitter and flicker numbers are the alarm
   to look for in real values.
3. **Median-5 smoothing helps against iid jitter but not against slow drift.** With iid noise it doubles run lengths
   (2.1 → 4.7) and flips 34 % of labels, but the labels are still noise. With correlated noise (τ = 25) it changes
   only 5 % of labels. Smoothing hides flicker; it does not create signal. Keep it as a flagged option only.
4. **A late-reacting VF (lagged) gives a strong negative A in a ~1.5 s window per failure and noise elsewhere.**
   Success vs failure positive fraction is only 0.417 vs 0.386, because the rest of each failure looks like a
   success. The plain `std ΔV / std A` ratio (0.04) looks reassuring there, because the drops dominate std(A).
   The robust ratio (0.31) shows that the labels outside the drops are noise. **Report both.**
5. **An "honest uncertainty" VF (smoothed-outcome) collapses labels onto the outcome:** 90 % of success frames vs
   0 % of failure frames positive, and **all 377 failure episodes are entirely negative**. This is the "labelling by
   outcome" failure mode the spec warns about. The per-episode histogram and the `num_all_negative` /
   `num_all_positive` counters catch it immediately. The labels also barely depend on N (agreement 0.98–0.99).
6. **Mechanical bounds in the grid:** at fixed N the p = 0.3 positives are a subset of the p = 0.4 positives, so their
   agreement is always exactly 1 − 0.1 = 0.900. Only the comparison **across N** says anything. For noise-dominated
   values it sits around 0.69–0.72, so real-value agreements near 0.7 mean "N-dependent noise".

Example strips (V, true g, A, shaded I_t) for the held-out episodes 531 and 674 (success) and 181 and 244 (failure):
`reports/figures/step_06/dev_fake_*/examples.png`. They show, for example, the lagged drop windows and the
outcome-collapsed labels.

### Inference smoke test (tiny random model; numbers are NOT representative of the real model)
| test | frames | result |
|---|---|---|
| single model, episodes 0–9, CPU, batch 32, 2 workers | 6 786 | all checks pass; 191 frames/s (dominated by JPEG decode + CPU) |
| `--crossfit 5`, episodes 1, 2, 6, 8 (folds 0, 1, 4, 2), CPU | 2 037 | each episode scored only by `tiny-random:fold{k}` of its own fold |
| single model, episode 2, **CUDA + bf16 autocast** | 230 | passes (GPU was shared with step 3, so only a 1-episode test) |
| `from_pretrained` of a `save_pretrained` tiny checkpoint (`runs/dev_infer_ckpt/checkpoints/best`) | 453 | passes; `model_id = dev_infer_ckpt/best` |
| `06_compute_advantages.py` on the 10-episode values (subset path) | 6 786 | works, warns "values cover 6786 frames of 10 episodes" |

**Pending (needs steps 4/5):** real `values.parquet` for all 634 838 frames, wall time and frames/s on the 5090, and
every diagnostic above computed on real values.

## Deviations from the spec
- **Threshold = order statistic.** ε is the k-th largest A with k = round(p_pos·n), rather than an interpolated
  `np.quantile`. It gives exactly the requested fraction when there are no ties, as the "± one frame" test requires.
- **Tie tolerance.** `I_t = A_t ≥ ε − tie_atol` with `tie_atol = 1e-6`, which is about 0.005 frames of progress and
  above float32 resolution. Without it, float32 round-off would decide labels in the degenerate cases.
  The JSON reports `ties_at_epsilon`, and `degenerate` is set when ties push the positive fraction away from p_pos
  by more than max(2/n, 1 %), or when A has (near-)zero spread.
- **Exact reward sum.** `compute_advantages(..., z=Z)` takes an extra keyword: with Z from the returns JSON, g = return_raw / Z
  in float64, so `reward_sum = −N/Z` exactly. Without it, the float32 `return_norm` column is used.
  `reward_sum`/`advantage` are float64 in the parquet. The contract does not fix dtypes; the column names are exact.
- **`forced_positive`.** ε is computed over the non-forced frames, then the forced frames are set True.
- **Grid outputs.** By default only the default `(N, p)` parquet is written (about 25 MB). The grid goes to
  `sensitivity.json`; `--write-grid` (or `write_grid: true`) writes every grid parquet.
- **Figure location.** Figures go to `reports/figures/step_06/<run>/` (one subdirectory per run), and smoothed
  outputs carry a `_smooth{K}` suffix.
- **Sidecar `values.json`.** `values.parquet` gets a sidecar `values.json` (timings / fake parameters), which is not
  part of the contract.
- **Extra robust jitter ratio** in `diagnostics`, motivated by finding 4.
- **Fake modes are my interpretation** of the brief. `lagged`: before the reaction, failures follow the success-shaped
  return −(T−1−t)/Z. `smoothed-outcome`: blend toward the outcome-averaged return, weight (t/(T−1))^γ. Both use T, so
  they are not causal — fine for development data.

## Open issues / questions for the orchestrator
1. **Cross-fitting is out of scope** (orchestrator decision, 2026-10-12). `--crossfit K` was already implemented and
   smoke-tested before that decision, so it is left in place but unused; the supported path is a single checkpoint via
   `--run/--ckpt`. No step-4 contract change needed.
2. `model_id` format: `<run>/<ckpt>` for a single model, `<fold_run>/<ckpt>:fold<k>` for cross-fit, `fake:<mode>` for
   fake runs. Step 7 should treat it as an opaque string.
3. The `runs/dev_fake_*` (~210 MB) and `runs/dev_infer_*` (< 1 MB) directories are dev artefacts and can be deleted
   at any time. `runs/dev_fake_noisy_s0.01` is a reasonable stand-in for step 7 to develop against.
4. Real-value expectations to check first: `degenerate` must be false; success > failure positive fraction but few
   episodes at 0 % / 100 %; robust jitter ratio well below 1; mean run length well above 5 frames; and across-N
   agreement clearly above the ~0.7 noise baseline.
5. README status row for step 6 not updated (orchestrator-owned file).

## Definition-of-done checklist
- [ ] `values.parquet` for all frames with the step-5-approved checkpoint(s) — **pending steps 4/5** (script done and smoke-tested)
- [ ] `advantages_50_0.4.parquet` + JSON (the deliverable) and the sensitivity grid — **code done, validated on fake values; real run pending**
- [ ] All diagnostics in the report with figures — **done for fake values; real-value diagnostics pending**
- [x] Tests pass, ruff clean (`pytest -q`: 82 passed, 31 in `tests/test_advantage.py`; `ruff check .` clean)
- [x] `reports/step_06_advantage.md` written (to be updated with real values)
