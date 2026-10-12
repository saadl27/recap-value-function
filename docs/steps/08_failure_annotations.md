# Step 8 — Failure-time annotations & localisation yardstick

> **Not implemented** (decided 2026-10-12, no time). This spec documents the design of [method v2](../method_v2.md).

**Depends on:** step 1 (frame store, on the workstation) and step 4 outputs (`runs/vf_c2229_s0/eval/*.parquet`, on
the workstation). **No GPU.** **Parallel with:** the implementation of steps 9–10.
**Owns:** `annotations/failure_times.csv` (git-tracked), `src/recap_vf/yardstick.py`, `scripts/08_contact_sheets.py`,
`scripts/08_yardstick.py`, `tests/test_yardstick.py`, `reports/step_08_failure_annotations.md`,
`reports/figures/step_08/`. Background: [method v2](../method_v2.md).

## Goal
1. A record of **when the decisive mistake happens** in failures, and **where the robot fumbles but recovers** in
   successes. It must be made **blind** to every value curve.
2. A **yardstick**: metrics that score any value function against these annotations. It is used to choose the
   cross-fit step budget `S` (step 9) and to compare the methods (step 11).

## Episodes (240, in two phases)
- **Phase 1 (needed for S, do first):** the 120 held-out episodes (57 failures, 63 successes). This is the
  `val_heldout` set of step 4.
- **Phase 2 (needed for step 11):** the 120 `val_train` episodes of step 4 (51 failures, 69 successes). Take the
  episode list from `runs/vf_c2229_s0/eval/val_train_step000250.parquet`.

## 8.1 Blind contact sheets — `scripts/08_contact_sheets.py`
- Reads the frame store `data/cache/frames/ep_XXXX.h5` (JPEG bytes per camera; schema in
  [conventions](../conventions.md)). It exists only on the workstation; the laptop has episodes 0–9.
- Writes JPGs with **cameras only**: never V, g, A or labels.
  - **Overview:** one tile per second (top | left wrist | right wrist), each tile labelled `ep, t = …s, outcome`.
    Use 4 tiles per row at about 0.75 scale.
  - **Zoom:** `--window T0 T1 --every 0.2` for a close look around an event.
- Run it on the workstation CPU (`scripts/remote.sh run "…"`; it must not touch the GPU, which is busy with step 9).
  Then rsync the JPGs to the laptop under `runs/annotation_sheets/` (git-ignored, ~100 MB).
- Code reaches the workstation through git: commit `scripts/08_contact_sheets.py` alone (explicit path, see
  [conventions → Git](../conventions.md#git)), push, then `scripts/remote.sh update`.

## 8.2 Annotation protocol
Do **not** open any V plot, any `values.parquet`, the `runs/viz_*` videos or the step-4 plots until every episode's
rows are written. Look at the overview first, then zoom around candidate events.

| event | in | meaning | time |
|---|---|---|---|
| `failure` | failures | first moment the outcome is clearly compromised: actuator slips or drops, gripper closes on nothing, actuator held sideways or tilted after the handover, collision, actuator lying in the tray instead of inserted | first frame where it is visible |
| `fumble` | any | a struggle that is later recovered (repeated insertion attempts, regrasp, slow correction) | `t_start`–`t_end` |
| `stall` | any | robot motionless for > 3 s | `t_start`–`t_end` |
| `human` | any | a human hand visible | `t_start` |
| `none` | successes | no visible problem | — |
| `unclear` | failures | failure cause not visible (e.g. ep 723) | — |

- If a failure has several candidate events, mark the earliest **decisive** one `primary = true`; others are `false`.
- `confidence`: `high` = visible in the zoom to ±0.2 s; `medium` = ±1 s; `low` = guess.
- Write a 5–15-word `description` of what you saw.

**CSV** `annotations/failure_times.csv`:
```
episode_index,split,success,event,primary,t_start_s,t_end_s,confidence,annotator,description
```
- One row per event; every annotated episode has at least one row.
- `split` ∈ {heldout, train}; `annotator = claude-step8`. Saad will spot-check and may change rows to `saad`.
- **Append rows as soon as an episode is done**, so the file survives interruptions and context compaction.

## 8.3 Yardstick — `src/recap_vf/yardstick.py`, `scripts/08_yardstick.py`
**Inputs**
- One or more values tables (`index, episode_index, frame_index, value`, at stride 1 or 10; the stride must divide 50).
- The annotations CSV and the returns JSON (for `Z`, `C_fail`).

**Conventions**
- Times are in seconds (50 fps).
- Before detecting drops, smooth V with a centred rolling median over 0.5 s (in samples).
- Advantage: `A_t = −50/Z + V(t+50) − V(t)` for `t + 50 < T`.

**Per failure episode with a `primary` failure of confidence ≥ medium** (time `t_f`):

| metric | definition | ideal |
|---|---|---|
| `drop_hit` | Take the largest drawdown of smoothed V (peak → later trough). `t_drop` = first time after the peak where V < peak − ½·drawdown. Hit if `|t_drop − t_f| ≤ 1 s`. | 1 |
| `drop_err_s` | `|t_drop − t_f|` | 0 |
| `drop_size` | mean V over [t_f − 2 s, t_f − 0.5 s] − mean V over [t_f + 1 s, t_f + 3 s] (clipped to the episode; NaN if a window is empty) | large |
| `adv_concentration` | share of `Σ min(A_t, 0)` falling in t ∈ [t_f − 1.5 s, t_f + 0.5 s] | high |
| `pre_fail_gap` | mean V over [0, t_f − 0.5 s] − mean V of the split's successes over the same time span from their start | ≈ 0 |

**Per success episode:**

| metric | definition | ideal |
|---|---|---|
| `fumble_dip` | for each `fumble` window [t0, t1]: smoothed V falls ≥ 0.15 below its running peak within [t0 − 1 s, t1] | 1 |
| `false_alarm` | for episodes with only `none` rows: smoothed V ever < −C_fail/Z + 0.02 (same level as step 4's `fail_detect`) | 0 |

**Global:** `flicker` = share of consecutive 10-frame steps with |ΔV| > 0.2 (subsample to stride 10 first).

**Output:** per split × values source:
- episode counts, `drop_hit` rate, median `drop_err_s`, median `drop_size`, median `adv_concentration`,
  mean `pre_fail_gap`, `fumble_dip` rate, `false_alarm` rate, `flicker`;
- plus a per-episode CSV under `reports/figures/step_08/`.

## 8.4 Decision: the fold step budget `S`
- Score the baseline at steps 250 / 500 / 750 / 1000 on the held-out episodes, from
  `runs/vf_c2229_s0/eval/val_heldout_step*.parquet` (rsync it from the workstation into the same path on the laptop).
- `S` = the step with the highest `drop_hit` rate. If several are within 0.05 of the best, prefer the lowest
  `false_alarm` rate, then the smaller step.
- Memorisation is irrelevant here: fold models never see the fold they label.
- Put **`S = …`** in the report's first lines, and **end phase 1 by reporting `S` to the orchestrator** (your final
  message). Step 9 is waiting for it. The orchestrator will tell you to continue with phase 2.

## 8.5 Phase 2
Annotate the 120 `val_train` episodes. Score the baseline at the 4 steps on them as well. The question to answer:
how far below held-out quality are the in-sample train curves (`drop_hit`, `pre_fail_gap`, `fumble_dip`)?

## Tests (`tests/test_yardstick.py`, synthetic only, CPU, < 10 s)
- A V step at `t_f` → hit, error ≈ 0, `drop_size` = the step size. A step at `t_f + 3 s` → miss. Flat V → miss,
  `drop_size` 0.
- `adv_concentration` = 1 for a single step at `t_f`. Stride 1 and stride 10 give the same metrics on a
  piecewise-constant V.
- `false_alarm` and `fumble_dip` on synthetic successes.
- The CSV loader rejects unknown events, `t_end < t_start`, and a failure episode without exactly one `primary` row
  (except `unclear`).

## Definition of done
- [ ] Contact-sheet script; sheets for the 240 episodes
- [ ] `annotations/failure_times.csv` complete for phase 1 (then phase 2), written blind
- [ ] Yardstick module + tests pass, ruff clean
- [ ] Baseline scored at 4 steps on held-out (phase 1) and val_train (phase 2), tables in the report
- [ ] `S` chosen by the rule and stated at the top of the report
- [ ] `reports/step_08_failure_annotations.md` written (template in [conventions](../conventions.md#reports))
