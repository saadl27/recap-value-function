# Step 11 — Compare labels: baseline vs v1 (cross-fit) vs v2 (bootstrapped)

> **Not implemented** (decided 2026-10-12, no time). This spec documents the design of [method v2](../method_v2.md).

**Depends on:** step 8 (both annotation phases), steps 9 and 10. **No GPU.** The values are small: rsync them from
the workstation. Renders need the frame store, so make them on the workstation CPU.
**Owns:** `scripts/11_compare.py`, `scripts/11_render_compare.py`, `reports/step_11_label_comparison.md`,
`reports/figures/step_11/`. Background: [method v2](../method_v2.md).

## Methods compared
| name | values | note |
|---|---|---|
| baseline | `runs/vf_c2229_s0/eval/val_{heldout,train}_step000{250,500}.parquet` | stride 10; in-sample on `val_train` |
| v1 | `runs/vf_c2229_cf/values.parquet` | out-of-fold for every episode |
| v2 | `runs/vf_c2229_td_s0/values.parquet` | student; in-sample on train, but trained on bootstrapped targets |

For a fair comparison, subsample v1 and v2 to the same stride-10 frames as the baseline eval files.

## Comparisons
1. **Yardstick** ([step 8](08_failure_annotations.md#83-yardstick--srcrecap_vfyardstickpy-scripts08_yardstickpy)),
   per method × split (heldout, val_train). The key question: do v1 and v2 bring `val_train` up to held-out quality
   (`drop_hit`, `drop_size`, `pre_fail_gap`, `fumble_dip`) while keeping held-out at least as good as the baseline?
2. **Memorisation signals** per method, `val_train` vs held-out:
   - early-episode V gap (success − failure);
   - auc@0 and auc@0.25;
   - share of successes with a dip > 0.2.
3. **Label diagnostics** (step 6) for v1 and v2 on all 800 episodes:
   - positive fraction on success / failure frames;
   - all-negative / all-positive episodes;
   - mean run length;
   - robust jitter ratio;
   - agreement v1 vs v2.

   Full-frame baseline labels were never computed: say so, don't make them.
4. **Videos** of the 20 episodes in `runs/viz_compare/` (10 train, 10 held-out), with stacked V panels for
   baseline-best, v1 and v2. Use the same layout as `runs/viz_compare`; start from the one-off
   `runs/diag/compare_video.py` on the workstation and copy it into `scripts/11_render_compare.py`.
5. **Blind sheets for Saad**:
   - 30 annotated episodes: 15 held-out and 15 val_train, half failures.
   - One PNG each: camera keyframes every 2 s, with the three V curves below in **random order**, labelled
     A / B / C, method names hidden.
   - Key: `reports/figures/step_11/blind_key.json`.
   - Template for his picks: `reports/step_11_blind_choices.csv`.

## Verdict
Recommend which labels to deliver (v1, v2 or the baseline), with the numbers that support it, and list each
method's typical failure modes. Recommend; do not change any method.

## Definition of done
- [ ] Yardstick + memorisation tables for the 3 methods × 2 splits
- [ ] Label diagnostics for v1 and v2
- [ ] 20 comparison videos (in `runs/`, not git) + 3–5 stills in `reports/figures/step_11/`
- [ ] Blind sheets + key + empty choices CSV
- [ ] `reports/step_11_label_comparison.md` with a verdict
