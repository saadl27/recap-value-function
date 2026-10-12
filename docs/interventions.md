# Rewards with human interventions (DAgger rollouts)

**Status:** implemented on 2026-10-12 for
[`DreamMachines/actuator_unboxing_1h_HQ_dagger_rollouts`](https://huggingface.co/datasets/DreamMachines/actuator_unboxing_1h_HQ_dagger_rollouts),
end to end (data, splits, returns, value training, labels). Code: `recap_vf.returns.compute_episode_returns_interventions`,
`scripts/02_compute_intervention_returns.py`, the cut handling in `recap_vf.advantage.compute_advantages`, and
`scripts/run_dagger_pipeline.sh`. Results are in [RESULTS.md](../RESULTS.md).

## The dataset
- 240 episodes, 256 537 frames at 50 fps; same robot, three cameras, 14-D state and task as the main dataset.
- A per-frame `intervention` flag: `control_phase` is 0 for the robot and 2 for the human; no other values occur.
  31 % of frames are under human control.
- Interventions per episode: 0 → 28 episodes, 1 → 141, 2 → 69, 3 → 2. Human runs last 90–749 frames (median 262,
  ≈ 5 s).
- **No `success` column**, only a quality rating of 3–5.
  - Every episode ends under robot control, and in the episodes checked (3 and 111) the actuator ends in the tray
    with the arms back home.
  - All episodes are therefore treated as **successes**: the human makes sure the task is finished. This is an
    explicit flag (`--assume-success`), never a silent default.

## The reward
Every frame costs −1. The **robot's last frame before each takeover** gets −C_interv instead. The last frame of the
episode gets 0 on success (−C_fail on failure, which does not occur here).

The **return-to-go is cut at the next takeover**:
- An episode splits into *stretches*. Each stretch starts at frame 0 or at a takeover, i.e. a human correction
  followed by the robot's control.
- A stretch ends at the next takeover or at the end of the episode.
- A frame's return sums its own stretch only:

```
G_t = −(end_s − 1 − t) − C_interv                 stretch ends at a takeover
G_t = −(T − 1 − t)   (− C_fail on failure)        last stretch
```

Why the cut rather than letting penalties pile up to the end of the episode:
1. **Blame stays local.** A second takeover is caused by robot actions after the human handed back, in a state the
   human just repaired. Without the cut, every frame before the first takeover would also pay for the second.
2. **Less noise.** Whether a later takeover happens is mostly unpredictable from the current frame. Counting it adds
   outcome noise to the targets, which the baseline showed it memorises.
3. **The value scale stays usable.** `Z = |min G|`. With 2–3 penalties piled up, `Z` grows to about 3·C and squeezes
   every other return into a fraction of the 201 bins.
4. **It matches the main dataset**, whose failed episodes already end at the (cut) human takeover with −C_fail.

The cost: V no longer predicts how many takeovers remain in the episode. It predicts how long until the robot either
finishes or needs help, which is the quantity that matters for labelling the robot's actions.

**C_interv** defaults to the longest stretch that ends in success, the analogue of `C_fail = T_max` in
[step 2](steps/02_returns.md). With it, every frame of a stretch that ends in a takeover scores below every frame of
a stretch that ends in success.

| Quantity | Value |
|---|---|
| C_interv | 1060 frames |
| Z | 1898 = 839 − 1 + 1060 (longest stretch that ends in a takeover) |
| `return_norm`, stretches ending in a takeover | [−1, −0.558] |
| `return_norm`, stretches ending in success | [−0.558, 0] |
| bins used | all 201 |

Figure: `reports/figures/interventions/actuator_unboxing_1h_HQ_dagger_rollouts_i1060_examples.png`.

## Advantages and labels
- **The N-step window stops at the cut.** A window that reaches the next takeover sums the rewards up to it,
  including −C_interv, and adds no value after it, exactly as at an episode end.
- If the window crossed the cut, it would add `V(o_{t+N})` from the next stretch. That value reflects a different
  future (after the human's repair) and would cancel the penalty.
- A unit test checks that a perfect V gives A = 0 everywhere, including across cuts.
- **Human frames are labelled positive** (RECAP's rule for corrections). The threshold ε is set on the robot's frames
  only, so 40 % of robot frames are positive, plus all human frames (58.5 % of all frames with fake values).
- Use `--no-force-interventions` to turn this off.

## How to run
Select the dataset with `RECAP_DATASET`. Everything (raw data, frame store, tables, splits, returns) then lives under
`data/raw/<dataset>/` and `data/cache/<dataset>/`; the main dataset's layout is unchanged.

```bash
bash scripts/run_dagger_pipeline.sh            # workstation: everything, resumable (or: scripts/remote.sh bg dagger "...")
# the same, step by step:
export RECAP_DATASET=actuator_unboxing_1h_HQ_dagger_rollouts
uv run python scripts/01_download.py                                   # ~2 GB (3 cameras x 240 mp4s)
uv run python scripts/01_extract_frames.py --source local --episodes all
uv run python scripts/01_make_splits.py --stratify interventions      # 204 train / 36 held-out, 5 folds
uv run python scripts/02_compute_intervention_returns.py               # -> returns_i1060.*
uv run python scripts/04_train_value.py --config configs/train_dagger.yaml
uv run python scripts/06_infer_values.py --run vf_dagger_i1060_s0 --ckpt best
uv run python scripts/06_compute_advantages.py --run vf_dagger_i1060_s0 --returns-tag i1060 --segment-len 50
```
- **Splits** are stratified on the number of interventions (0 / 1 / 2+) × length quartile, instead of on success.
- **Training** uses the baseline recipe unchanged. Only the data differ, plus a step cap of 850 ≈ one pass over the
  218 k training frames, the same rule as the baseline.
- `configs/train_dagger.yaml` refuses to run unless `RECAP_DATASET` matches.

## Open choices (not implemented)
- **The first moments of human control.** The operator hesitates and re-aligns before correcting. These frames are
  forced positive today. ROVE instead assigns its penalty at the end of this "adaptation" phase. Its ablation
  (80 % vs 50 % policy success) also changes the critic horizon, so it is not proof.
- **Pauses or frozen frames around the handover.** None are flagged in this export (`control_phase` has no
  in-between value). Worth a visual check before training.
