# Step 10 — Bootstrapped critic distilled from the OOF teacher, labels v2

> **Not implemented** (decided 2026-10-12, no time). This spec documents the design of [method v2](../method_v2.md).

**Depends on:** step 9 outputs (`runs/vf_c2229_cf/{values.parquet,value_probs.npy}`). **Runs on:** the RTX 5090,
inside `scripts/run_v2_pipeline.sh` (see [step 9](09_crossfit.md#95-the-pipeline-script--scriptsrun_v2_pipelinesh-s---smoke)).
Background: [method v2](../method_v2.md).
**Owns:**
- `src/recap_vf/bootstrap.py`, `scripts/10_bootstrap_targets.py`, `configs/train_td.yaml`, `tests/test_bootstrap.py`
- `reports/step_10_bootstrapped_critic.md`, `reports/figures/step_10/`
- **additive** edits to `src/recap_vf/train.py`, `src/recap_vf/metrics.py`, `tests/test_train.py`. With the new
  option unset, the baseline config must behave exactly as before (same loss, same metrics). Write a test for that.

## 10.1 Targets — `src/recap_vf/bootstrap.py`, `scripts/10_bootstrap_targets.py`
**Notation**
- Teacher `p_t` = row of `value_probs.npy` (OOF distribution of frame t); support `v_b = −1 + b/200`
  (`recap_vf.bins`).
- `N = 50`; `Z` from `returns_c2229.json`; `g_t = return_raw / Z`, in float64.

For frame t of an episode of length T:
```
t + N <  T :  q_t = Proj( p_{t+N} shifted by r = −N/Z )
t + N >= T :  q_t = two-hot(g_t)            # true return incl. terminal reward; the episode ends in the window
```
- **Proj** (C51 projection): each atom `v_b` moves to `z = clip(v_b + r, −1, 0)`. Its mass `p_{t+N,b}` is split
  linearly between the two bins that bracket `z`.
- **two-hot(x)**: mass on the two bins bracketing x, linear in the distance.
- Both keep the mean exact: `Σ_b q_t,b v_b = −N/Z + V^oof_{t+N}`, except for the mass clipped at −1. Report how
  many frames clip.
- **Never bootstrap across episodes.** Index t + N within the episode using `frame_index` / `episode_length`, not
  row offsets alone.
- **Output:**
  - `runs/vf_c2229_cf/targets_td50.npy`: float16 `[634 838, 201]`, row = global index; renormalise rows after the
    float16 cast.
  - `targets_td50.json`: N, Z, teacher run, sha256 of `value_probs.npy`, number of clipped frames, max |mean − (−N/Z + V_next)|.

## 10.2 Training with soft targets (`train.py`, additive)
- `TrainConfig.soft_targets: str | None = None` — path to such an `.npy`. Open it with `np.load(mmap_mode="r")` so
  workers share it.
- When set:
  - each training item also carries `target_probs` [201];
  - loss = `−Σ_b q_b · log_softmax(ℓ)_b`, averaged over the batch;
  - every val set also reports `soft_ce`;
  - `best` and early stopping use `val_heldout/soft_ce`.
- All existing metrics (MC CE vs `target_bin`, MAE vs the MC return, AUCs, `fail_detect`, …) stay, so the student
  can be compared with the baseline.
- `soft_targets: null` → exactly today's code path.

## 10.3 Student run `vf_c2229_td_s0` — `configs/train_td.yaml`
- Baseline recipe, plus `soft_targets: runs/vf_c2229_cf/targets_td50.npy`.
- Original `splits.json`: 680 train / 120 held-out. The held-out episodes stay out of training for good: they are
  the clean check.
- Initialise from the pretrained backbones (same as the baseline and the paper).
- `steps` cap 2100, `eval_every` 250, `early_stop_patience` 3 on held-out `soft_ce`.
- Held-out targets also come from the OOF teacher, which never saw those episodes, so held-out `soft_ce` is a clean
  selection signal.
- Then:
```bash
uv run python scripts/06_infer_values.py --run vf_c2229_td_s0 --ckpt best          # all frames
uv run python scripts/06_compute_advantages.py --run vf_c2229_td_s0 --segment-len 50   # labels v2
```

## Out of scope (needs Saad's decision; do not build)
- Expectile / optimistic value (ROVE's OVE): it changes what V estimates and needs a design for the 201-bin head.
- Iterating stage 2 (student → new targets → …).
- TD(λ) mixtures, other N.
- Penalty at the mistake ([step 12](12_failure_time_penalty.md)).

## Tests (`tests/test_bootstrap.py`, CPU, < 10 s)
- Projection:
  - the mean is preserved;
  - rows sum to 1;
  - `r = 0` gives the identity;
  - mass below −1 lands in bin 0.
- two-hot: exact mean, at most two adjacent non-zero bins, and `x` on a bin centre → one-hot.
- Episode boundaries: frames with t + N ≥ T use `two-hot(g_t)`; no read crosses into the next episode.
- A vectorised implementation matches a slow per-frame reference on random teachers.
- `train.py`:
  - `soft_targets` unset → the same loss as before on a fixed batch;
  - soft targets equal to `one_hot(target_bin)` → `soft_ce` equals `F.cross_entropy`.

## Report (`reports/step_10_bootstrapped_critic.md`)
- Curves: train soft CE, held-out soft CE, held-out MC CE next to the baseline's. Best step; wall time.
- Memorisation signals, `val_train` vs held-out:
  - early-episode V gap (success − failure);
  - auc@0 and auc@0.25;
  - MAE on success frames.
- Step-6 diagnostics for labels v2, and the agreement of labels v1 vs v2.
- Target sanity: number of clipped frames, max mean error, an example plot of `E[q_t]` vs the MC return for 2
  success and 2 failure episodes.

## Definition of done
- [ ] Bootstrap module + targets script + soft-target training; tests pass; ruff clean; smoke run via the pipeline
- [ ] `targets_td50.npy` for all frames
- [ ] `vf_c2229_td_s0` trained; best checkpoint by held-out `soft_ce`
- [ ] Values for all frames + labels v2 (`runs/vf_c2229_td_s0/advantages_50_0.4.*`)
- [ ] `reports/step_10_bootstrapped_critic.md` written
