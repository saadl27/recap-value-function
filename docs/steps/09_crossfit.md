# Step 9 — Cross-fitted MC critics (the teacher) and labels v1

> **Not implemented** (decided 2026-10-12, no time). This spec documents the design of [method v2](../method_v2.md).

**Depends on:** steps 1–4 and 6 (code), and `S` from [step 8](08_failure_annotations.md). **Runs on:** the RTX 5090
(smoke tests on the laptop). Background: [method v2](../method_v2.md).
**Owns:**
- `scripts/09_make_fold_splits.py`, `scripts/run_v2_pipeline.sh` (shared with step 10), `configs/train_crossfit.yaml`
- `reports/step_09_crossfit.md`, `reports/figures/step_09/`
- **additive** edits to `scripts/06_infer_values.py` (`--save-probs`) and `tests/test_advantage.py` (or a new
  `tests/test_infer.py`)

## Goal
For every frame of all 800 episodes:
- a value from a model that **never trained on that episode** (out-of-fold, OOF);
- its 201-bin distribution (needed by step 10);
- the step-6 labels from these values = **labels v1**.

## 9.1 Folds — `scripts/09_make_fold_splits.py`
- `splits.json["folds"]`: 5 × 160 episodes, stratified (53 % success each), covering all 800. The 120 held-out
  episodes are spread 27 / 24 / 17 / 29 / 23 across the folds.
- Write `data/cache/splits_cf{k}.json` with the same schema as `splits.json`, plus `"crossfit_fold": k`:
  - `heldout = folds[k]`
  - `train` = the other 640 episodes
  - `folds` copied unchanged
- `train.py` then works **unchanged**: `val_heldout` = fold k (monitoring only), `val_train` = 120 random training
  episodes of that fold model. Training on the original held-out episodes of *other* folds is intended.

## 9.2 Fold training
- Baseline recipe (`configs/train_default.yaml`), with:
  - `splits_file: data/cache/splits_cf{k}.json`, `run_name: vf_c2229_cf{k}`
  - `steps: S`, `eval_every: 250` (S is a multiple of 250), `early_stop_patience: 0`, seed 0
- Keep these in `configs/train_crossfit.yaml`; the fold-specific keys go through `--set`.
- Use the **`last`** checkpoint (= step S). Never `best`: it is selected on the very fold it would label.
- About 1.35 h per fold for S = 500 (including evals on 160 + 120 episodes), so about 7 h for the 5 folds.
- Disk: about 11 GB per fold run (with optimiser state); the workstation has about 450 GB free.

## 9.3 OOF inference — `scripts/06_infer_values.py --save-probs`
```bash
uv run python scripts/06_infer_values.py --run vf_c2229_cf --crossfit 5 --ckpt last --save-probs
```
- Add `--save-probs`: also write `runs/<run>/value_probs.npy`.
  - float16 `[n_rows, 201]` = softmax of the logits; row i ↔ row i of `values.parquet`.
  - Rows are sorted by global `index`, so for the full run row i is global index i. About 255 MB.
  - Record the dtype, shape and sha256 in `values.json`.
- The existing `--crossfit` assertions check that every episode is scored by its own fold model.
- Speed: step 6 measured about 55 frames/s at batch 64 / 8 workers, i.e. about 3.2 h for all frames. Time 2 000
  frames at batch 256 with 8 and 16 workers before the full run, and keep the fastest setting.
- Tests:
  - probability rows sum to 1 (±1e-3 in float16);
  - Σ p·v equals `value` to within 1e-3;
  - the default behaviour (no `--save-probs`) is unchanged.

## 9.4 Labels v1
```bash
uv run python scripts/06_compute_advantages.py --run vf_c2229_cf --segment-len 50      # N = 50, p = 0.4 defaults
```

## 9.5 The pipeline script — `scripts/run_v2_pipeline.sh S [--smoke]`
- One resumable chain for steps 9 **and** 10, run unattended on the workstation:
  `scripts/remote.sh bg v2_pipeline "bash scripts/run_v2_pipeline.sh S"`.
- Each stage is skipped if its final output exists. `set -euo pipefail`; log one line per stage start/end with
  timestamps.
- Stages:
  1. fold splits
  2. folds 0–4
  3. OOF inference (+ probabilities)
  4. labels v1
  5. bootstrapped targets (step 10)
  6. student training (step 10)
  7. student inference
  8. labels v2
- **Smoke mode** (laptop): episodes 0–9 only (the local frame store), K = 2 folds via a temporary splits file,
  `steps 4`, `eval_every 2`, micro-batch 2, `freeze_vision: true`, `val_frame_stride 10`.
  - It must run end-to-end to labels v2 in a few minutes.
  - Smoke outputs go to `runs/smoke_v2_*` and `data/cache/splits_smoke_*`; delete them afterwards.
- `--crossfit` in `06_infer_values.py` reads the folds from `--splits`, which is how the smoke run gets its 2 folds.

## 9.6 Workstation rules (mandatory)
- `scripts/remote.sh gpu` before launching. **Saad's own run `vf_c1115_s0` may still be training. Never stop it.**
  Wait until the GPU is free, re-checking every ~15 min.
- Code reaches the workstation through git only: commit the files you own (explicit paths), push,
  `scripts/remote.sh update`.
- After launching, watch the log until fold 0 has trained about 50 steps:
  - the loss is falling;
  - throughput is ≈ 31 samples/s;
  - data wait is about 0 %.

  Then report and finish. Do not block for hours; the orchestrator monitors the chain.

## Report (`reports/step_09_crossfit.md`)
- **Fold table:** fold, number of train episodes, steps, wall time, CE on its own fold at 250 and at S, `val_train` CE.
- **No-memorisation check on the OOF values of all 800 episodes:** mean V over the first 10 % of the episode,
  success − failure, overall and per fold. It should be ≈ the held-out baseline (≈ 0.03 or less).
- **Step-6 diagnostics for labels v1:** positive fractions on success / failure frames, all-negative episodes, run
  lengths, robust jitter ratio.
- Inference throughput.
- Smoke-test commands and full commands.

## Definition of done
- [ ] Fold splits + `--save-probs` + pipeline script; smoke run end-to-end on the laptop; tests pass; ruff clean
- [ ] 5 fold models trained to S on the workstation
- [ ] `runs/vf_c2229_cf/{values.parquet,value_probs.npy}` for all 634 838 frames, each scored by its own fold
- [ ] Labels v1 (`runs/vf_c2229_cf/advantages_50_0.4.*`) and their diagnostics
- [ ] `reports/step_09_crossfit.md` written
