# Step 5 — Memorisation check

**Depends on:** step 4. **Owns:** `scripts/05_memorization_report.py`,
`reports/step_05_memorization.md`, `reports/figures/step_05/`.

## Why this step exists
We will label the **same episodes we train on**. A 670M network trained for several passes over 800 episodes can learn
to recognise individual episodes (actuator position in the box, lighting, the arm's exact start pose) and recall how
they ended. Then, on a *training* episode that eventually fails, V is already low at frame 0 instead of dropping at
the actual mistake → the advantage inside that episode is flat noise → useless labels.

What we want: `V(o_t)` = expected outcome **given only what is observable at t**. Held-out episodes tell us what an
honest model looks like; the gap between train and held-out tells us how much it memorised.

The paper avoids this with much more data and web-data co-training. **Cross-fitting is out of scope for now**
(decision 2026-10-12); if memorisation shows up we fix it with checkpoint choice and regularisation only.

## Diagnostics (`scripts/05_memorization_report.py --run vf_c2229_s0 --ckpt best`)
Run the selected checkpoint on `val_train` and `val_heldout` (same 120-episode, stride-10 sets as step 4) and on
**all frames of 40 train + 40 held-out episodes** (balanced success/failure) for curve-level analysis.

| # | Check | Memorisation looks like |
|---|---|---|
| 1 | CE / MAE gap train vs held-out | train ≪ held-out |
| 2 | `auc@0` (success predicted from the **first frame**) train vs held-out | held-out ≈ 0.5–0.7 but train ≥ 0.9 |
| 3 | `fail_detect_time` distribution (first t/T where V is confidently in the failure range) for failure episodes | train failures detected at t ≈ 0; held-out failures detected later, near the mistake |
| 4 | Mean V(t/T) curves, split by outcome × split (4 curves with bands) | train curves separated from t = 0; held-out curves start together and diverge |
| 5 | Same checks at an earlier checkpoint (e.g. step 2 000) and at `last` | gap grows with training |

Save figures under `reports/figures/step_05/` and a metrics table in the report.

## Decision rule
- **No significant memorisation** — `auc@0` gap < 0.1, `fail_detect_time` medians within 0.1, curves look alike:
  → keep the single model (`best` checkpoint). Go to step 6 with `--run vf_c2229_s0`.
- **Memorisation present** — try, in this order, stopping as soon as the gap closes:
  1. **Earlier checkpoint** (the gap often appears after the val-CE minimum). Cheap.
  2. **Regularised re-run**: `head_dropout: 0.1`, `weight_decay: 0.01`, and/or fewer steps.
     One run (~same cost as step 4).

## Definition of done
- [ ] All five diagnostics computed, figures + table in the report
- [ ] A clear decision (single model / earlier checkpoint / regularised) with the numbers that justify it
- [ ] If re-training was needed: the new runs exist, and the diagnostics re-run on them show the gap closed
- [ ] The report states exactly which checkpoint(s) step 6 must use (`--run` / `--ckpt` arguments)
- [ ] `reports/step_05_memorization.md` written
