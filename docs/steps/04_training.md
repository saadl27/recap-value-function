# Step 4 — Value-function training

**Depends on:** steps 1, 2, 3. **Runs on:** the RTX 5090 (smoke tests on the laptop).
**Owns:** `src/recap_vf/train.py`, `src/recap_vf/metrics.py`, `src/recap_vf/augment.py`,
`scripts/04_train_value.py`, `configs/train_default.yaml`, `tests/test_train.py`, `tests/test_metrics.py`,
`reports/step_04_training.md`, `reports/figures/step_04/`.

## Goal
Train the value model with Monte-Carlo targets on the **train** split, select the checkpoint by held-out loss, and log
everything step 5 needs to diagnose memorisation.

## Data
- Train set: `FrameDataset(splits["train"])` — all frames of 680 episodes (~540 k frames), joined with
  `returns_{tag}.parquet` on `index` to add `target_bin` and `return_norm`.
- Val sets (both evaluated every `eval_every` steps, **no augmentation**):
  - `val_heldout`: all 120 held-out episodes, `frame_stride=10` (~9.5 k frames);
  - `val_train`: 120 randomly chosen **train** episodes (fixed seed), `frame_stride=10`. Same metrics as held-out →
    the train/held-out gap is the memorisation signal used in step 5.
- Sampling: uniform over train frames, shuffled, steps-based (an "epoch" has no special meaning). Optional
  `balance: outcome` re-weights so success and failure frames are drawn 50/50 (failure frames are ~56 % by default) —
  off by default.

## Loss & targets
- Default (paper): `F.cross_entropy(logits, target_bin)`.
- Option `target: hlgauss` → soft targets `bins.hl_gauss_targets(return_norm, sigma_bins=0.75)` with
  `-(targets * log_softmax(logits)).sum(-1).mean()`.

## Optimisation (defaults ≈ RLinf's recipe, adapted to one GPU)
| Knob | Default | Note |
|---|---|---|
| optimiser | AdamW, betas (0.9, 0.95), eps 1e-8 | |
| lr pretrained / new | 5e-5 / 1e-4 | `model.param_groups` |
| weight decay | 0.0 | knob for step 5 |
| schedule | linear warm-up 200 steps → **constant** | RLinf's choice; lets a run be extended by resuming |
| grad clip | 1.0 | |
| global batch | 256 = micro-batch × grad-accum (micro 32 × 8 to start; tune to memory) | |
| steps | **cap 2 100 (≈ 540 k samples = one pass over the train frames, ~4–5 h on the 5090)** + early stopping: stop when `val_heldout/ce` has not improved for 3 evals | 8 000 (RLinf) ≈ 18 h here — unnecessary for 50 fps frames that are near-duplicates, and more passes raise the memorisation risk |
| precision | bf16 autocast, fp32 master weights, TF32 on | |
| grad checkpointing | as needed for micro-batch 32 | |
| eval / ckpt | every 250 steps | |
| augmentation | **off** (`augment: false`) | step 5 may turn it on |
| seed | 0 | |

`torch.compile` is optional — only if it works first try; report.

## Augmentation (`src/recap_vf/augment.py`, used only if `augment: true`)
Mirror RLinf's openpi-style recipe, applied on GPU per sample, train only:
- non-wrist cameras (`top`): random crop 95 % of H and W → resize back to 224; random rotation ±5°;
- all cameras: brightness ×U(0.7, 1.3), contrast ×U(0.6, 1.4) around the image mean, saturation ×U(0.5, 1.5);
  clamp to [0, 1].
Unit-test that it keeps shape/dtype/range and is a no-op when disabled.

## Metrics (`src/recap_vf/metrics.py`, computed on both val sets)
| Metric | Definition |
|---|---|
| `ce` | mean cross-entropy vs `target_bin` |
| `mae` | mean \|E[V] − return_norm\| |
| `spearman` | Spearman ρ(E[V], return_norm) over all val frames |
| `auc@f` for f ∈ {0, 0.1, 0.25, 0.5, 0.75} | ROC-AUC of predicting episode success from V at the frame nearest to `f·T` (one sample per episode) |
| `fail_detect_time` | for failure episodes: first `t/T` where V < `−C_fail/Z + δ` (δ = 0.02), i.e. the model is confident of failure; report the median for each val set |
| `entropy` | mean entropy of the predicted distribution |

Note: val CE **cannot** go to ~0 — the outcome is genuinely uncertain early in an episode. Expect train CE to keep
decreasing while val CE bottoms out and rises: that's the over-fitting point.

## Logging & artefacts (`runs/<run_name>/`)
- `config.yaml` (fully resolved, incl. returns tag, Z, split file hash, git commit), `tb/` TensorBoard,
  `metrics.jsonl` (one line per eval), `checkpoints/{best,last}/` (`best` = min `val_heldout/ce`), `plots/`.
- Every eval: plot E[V] (line) and the true `return_norm` (dashed) vs time for **8 fixed held-out episodes**
  (4 success, 4 failure) and **4 fixed train episodes**; save PNG + log to TensorBoard.
- Resumable from `last` (`--resume`).
- Default run name: `vf_c2229_s0`.

## Lessons from step 3 (measured)
- Batches of 4 at lr 1e-4 on the pretrained LM could not even memorise 16 samples in 200 steps; full-batch 16 did.
  Keep the large effective batch (256 via accumulation) and lr ≤ 5e-5 on pretrained weights.
- Laptop (≈5 GiB usable on the RTX 4060): smoke tests need `freeze_vision: true`, `gradient_checkpointing: true`,
  micro-batch ≤ 8. Full fine-tune only on the 5090.
- Backbone default is the ungated mirror `unsloth/gemma-3-270m` (identical weights); the workstation has no HF login.

## Running on the workstation
1. Develop + smoke-test on the laptop.
2. Commit **only the files this step owns** (`git add <paths>` — never `git add -A`) on `main`, push.
3. `scripts/remote.sh update` (git pull there), `scripts/remote.sh gpu` (must be free of other users' jobs),
   then `scripts/remote.sh bg <name> "uv run python scripts/04_train_value.py …"`; follow with `log` / `jobs`.
4. The frame store, splits and returns tables already exist on the workstation (`~/DreamMachines/data/cache/`).

## Tasks
1. `train.py`: config dataclass (loaded from YAML, overridable from CLI like `--set steps=200`), dataloaders,
   loop with grad accumulation, eval, checkpointing, resume.
2. `metrics.py` + tests (synthetic predictions where the answer is known, e.g. perfect predictions → AUC 1, MAE 0).
3. **Smoke test** (laptop): 10 train episodes, 2 val episodes, `freeze_vision: true`, micro-batch 2, 50 steps → loss
   decreases, a checkpoint is written, resume works.
4. **Throughput benchmark** (5090, before the real run): 30 steps for micro-batch ∈ {16, 32, 48} × grad-ckpt
   on/off → samples/s and peak memory table. Make sure the dataloader keeps up (GPU util > 90 %; if not, add workers).
   Estimate wall time for the step cap. Measured on the 5090 (step 3): full FT + gradient checkpointing, micro-batch
   32 ≈ 1.0 s/step ≈ 31 samples/s (~50 % of the card's bf16 peak), so speed-ups beyond ~20–30 % need less compute
   per sample, not tuning. Freezing the vision tower is 1.7× faster but changes the recipe — not the default.
5. **Main run** `vf_c2229_s0` with defaults. Monitor: loss curves, val metrics, the V-curve plots.
6. Report: curves (train CE, val_heldout CE, val_train CE), best step, final metrics table for both val sets,
   example V-curve plots, wall time, peak memory, anything surprising.

## Pitfalls
- Join targets by **global `index`**, never by row position.
- Evaluate in `model.eval()` + `torch.no_grad()` + autocast; restore `train()` afterwards.
- Make sure the held-out episodes never enter the train sampler (assert disjointness at start-up).
- Run long jobs under `nohup`/`tmux` and log to a file so they survive the agent session; record the PID.

## Definition of done
- [ ] Smoke test passes on the laptop; tests pass; ruff clean
- [ ] Throughput/memory table in report
- [ ] `vf_c2229_s0` trained; best checkpoint selected by held-out CE
- [ ] Metrics table for `val_heldout` and `val_train` at the best step and at the last step
- [ ] V-curve plots for held-out and train episodes in `reports/figures/step_04/`
- [ ] `reports/step_04_training.md` written
