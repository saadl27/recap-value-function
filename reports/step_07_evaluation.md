# Step 07 — Empirical evaluation
**Status:** partial — **tooling done, evaluation pending** (no trained value model yet)        **Agent/date:** Claude (Step-7 tooling agent), 2026-10-12

## Summary
- `src/recap_vf/viz.py`: reusable helpers. It loads one episode's V/g/A/I from the step-6 artefacts and the step-2
  returns table, composes the camera strip (in `constants.CAMERAS` order), builds a V/g/A/I panel that matplotlib
  draws **once** (each frame then only paints a cursor into cached pixels), and has an H.264 writer (PyAV/libx264),
  the mp4 and keyframe-strip renderers, and the selection logic (keyframes, V-dip depth, the 20-episode evaluation
  set).
- `scripts/07_render_episode.py`: one mp4 per episode. Layout: header with time and V/g/A/I readout, then
  top | left wrist | right wrist, then a panel with V(t) solid, g(t) dashed, A(t) on its own axis below with ε and
  0 lines, I_t=True shaded green and a moving cursor. Every 2nd frame at 25 fps (real time), CRF 28.
- `scripts/07_keyframes.py`: the k most negative and k most positive A_t (≥ 2 s apart, sign enforced), saved as PNG
  strips at t−1 s, t, t+1 s, t+2 s with V and A printed above each column, plus `keyframes.json`.
- `scripts/07_select_episodes.py`: the fixed-seed held-out set (10 random failures, 5 successes with the deepest
  V dip, 5 random successes), written to `runs/<run>/eval_episodes.json`. It runs in 0.6 s on all 800 episodes / 120
  held-out and gives the same output on reruns (identical md5).
- **Render speed (laptop CPU):** 3–4 s for a typical 12 s episode, about 9 ms per written frame, so about 10 s for
  the longest (44 s) episode. Expect about 1.5–2 min for all 20 episodes. **Size:** 0.3–0.9 MB per typical episode,
  2.4 MB for the 44 s one. Keyframe strip about 250 KB.

## How to reproduce
Development used fake values (no model exists yet). `runs/dev_viz/` was made by a throw-away scratchpad generator,
not committed. It is V = g + noise, with sigmoid "change of mind" drops in failures and Gaussian dips in 30 % of
successes, and A computed per docs/background.md §3. Step 6's `runs/dev_fake_*` runs were also used to check
compatibility.
```bash
# tests + lint for the step-7 files
env -u PYTHONPATH uv run pytest -q tests/test_viz.py
env -u PYTHONPATH uv run ruff check src/recap_vf/viz.py scripts/07_*.py tests/test_viz.py

# laptop smoke (frame store only has episodes 0-9; only ep 0 is held-out -> --pool all --require-frames)
env -u PYTHONPATH uv run python scripts/07_select_episodes.py --run dev_viz --pool all --require-frames \
    --out runs/dev_viz/eval_episodes_local.json
env -u PYTHONPATH uv run python scripts/07_render_episode.py --run dev_viz --episodes-file runs/dev_viz/eval_episodes_local.json
env -u PYTHONPATH uv run python scripts/07_keyframes.py --run dev_viz --episodes-file runs/dev_viz/eval_episodes_local.json \
    --out-dir runs/dev_viz/keyframes
```
**Real evaluation, once step 6 has run** (on a machine whose frame store has the held-out episodes, i.e. the
workstation). `<run>` is the step-5-approved run:
```bash
env -u PYTHONPATH uv run python scripts/07_select_episodes.py --run <run>          # --seed 0, held-out pool
#   cross-fitted run: add --pool all (every episode is then scored by a model that never saw it)
env -u PYTHONPATH uv run python scripts/07_render_episode.py --run <run> --episodes-file runs/<run>/eval_episodes.json \
    --advantages advantages_50_0.4 --returns-tag c2229                             # -> runs/<run>/videos/ep_XXXX.mp4
env -u PYTHONPATH uv run python scripts/07_keyframes.py --run <run> --episodes-file runs/<run>/eval_episodes.json --k 3
#   -> reports/figures/step_07/ep_XXXX_{drop,rise}_{i}.png + keyframes.json (see open issue on PNG volume)
```
Useful flags: `--episode 0,3,5-7` (either script), `--advantages none` (values only, no A/I), `--max-frames`
(quick look), `--stride/--fps/--crf/--preset`, `--min-sep-s`, `--out-dir`. `render_log.json` in the video dir records
time and size per episode.

## Results (tooling validation only, fake values)
Example figures, **fake values, not results**:
[`dev_fake_video_frame_ep_0009.png`](figures/step_07/dev_fake_video_frame_ep_0009.png) (one mp4 frame) and
[`dev_fake_ep_0009_drop_0.png`](figures/step_07/dev_fake_ep_0009_drop_0.png) (one keyframe strip).

| check (by looking at decoded mp4 frames and strips) | result |
|---|---|
| camera order | top, left wrist, right wrist, labelled in each tile, matching `constants.CAMERAS` |
| timing | ep 9: 2211 frames → 1106 written frames at 25 fps = 44.2 s = real time; header clock matches |
| cursor | moves: seen at 6 s, 24 s and 44 s in ep 9, and at 4.8 s and 8.0 s in ep 5; a dot follows V and A |
| legibility | 672 × 548 px video; header 13 px, axes 8.5 pt, all labels visible after fixes (below) |
| colours | V blue, g grey dashed, A orange, I_t green wash, ε dashed line labelled; strip colours preserved after quantisation |
| degenerate inputs | oracle V = g (A ≡ 0): flat A axis ±1e-4, fully shaded; values-only (`--advantages none`): A panel dropped |
| edge cases | short episode (ep 6, 4.5 s): t+1 s / t+2 s tiles show "after end"; picks near t = 0 show "before start" |
| step-6 compatibility | rendered and selected from `dev_fake_noisy_s0.01_corr25`, `dev_fake_oracle`, `dev_fake_smoothed-outcome` without changes |

Fixed after looking at the frames:
- "time [s]" was clipped at the bottom, and the y-labels were pushed off the left edge by wide tick labels. Margins
  are now in pixels and the y-labels sit at a fixed pixel position.
- The legend covered the g(t) line. It now sits in a row above the plot.
- Median-cut quantisation turned the orange A line brown. Switched to octree quantisation, which also makes files
  30 % smaller.
- A "rise" pick could land on a negative A in short episodes. Picks must now have the right sign.

Dev artefacts: `runs/dev_viz/` (45 MB, can be deleted).

## Episode selection (fill in once real values exist)
`runs/<run>/eval_episodes.json`, seed 0. "Deepest V dip" = the largest drop from a peak to a later trough of V,
after a centred 25-frame (0.5 s) rolling median.

| group | episodes |
|---|---|
| failure_random (10) | _pending_ |
| success_deepest_dip (5) | _pending_ |
| success_random (5) | _pending_ |

## Per-episode judgement (to be filled from actually looking at the frames)
| ep | outcome | what visibly happens (time) | biggest V drop at | matches a visible problem? (±1 s) | lead time (V drop before visible failure) | recovery visible? | V rises at recovery? | notes |
|---|---|---|---|---|---|---|---|---|
| | failure | | | | | | | |
| | failure | | | | | | | |
| | failure | | | | | | | |
| | failure | | | | | | | |
| | failure | | | | | | | |
| | failure | | | | | | | |
| | failure | | | | | | | |
| | failure | | | | | | | |
| | failure | | | | | | | |
| | failure | | | | | | | |
| | success (dip) | | | | — | | | |
| | success (dip) | | | | — | | | |
| | success (dip) | | | | — | | | |
| | success (dip) | | | | — | | | |
| | success (dip) | | | | — | | | |
| | success | | | | — | | | |
| | success | | | | — | | | |
| | success | | | | — | | | |
| | success | | | | — | | | |
| | success | | | | — | | | |

Reminders from the spec: describe problems concretely (missed grasp, slip/drop, collision, stall, placed outside the
tray). For failures, note whether V ever looked confident and when it changed its mind. For dips in successes,
decide whether each was a real fumble that was recovered or a false alarm. Flag `I_t = True` during clearly bad
behaviour and `False` during clearly good, fast progress.

## Summary (pending)
- Fraction of failures whose largest V drop lines up (±1 s) with a visible problem: _pending_; median lead time: _pending_
- Fraction of observed recoveries where V rises: _pending_
- False alarms (V dips in successes with no visible problem): _pending_
- Typical failure modes of the value function: _pending_
- 3–5 best illustrative stills: _pending_
- mp4 paths for Saad: `runs/<run>/videos/ep_XXXX.mp4` for the 20 episodes above, _pending_

## Deviations from the spec
- **A(t) "on a second axis"** is drawn as a second stacked axis with a shared time axis, not a twin-y overlay. Two
  value scales on one plot are easy to misread, and A's scale (~±0.05) is very different from V's.
- **Timestamp overlay** is a dark header bar above the cameras. It holds the episode, outcome, t, V, g, A and I. It
  is not burned into a camera image, so it never hides the scene. Each camera tile has a small name label.
- **"Deepest dip in V"** is not defined in the spec. I used the peak-to-later-trough drawdown of a 0.5 s rolling
  median of V, so single-frame noise cannot choose the episodes (`--smooth 1` gives raw V).
- **Keyframe spacing** follows the spec's "≥ 2 s apart" (`--min-sep-s 2`). The 3 s windows (t−1…t+2 s) can
  therefore overlap by up to 1 s. Use `--min-sep-s 3` for strictly disjoint windows. Picks must have the matching
  sign, so short episodes may give fewer than k strips.
- Additions not in the spec: ε line in the A panel, `keyframes.json` and `render_log.json` indexes, `--pool` /
  `--require-frames` on the selection script (laptop smoke and the cross-fitted case), and `--advantages none`.

## Open issues / questions for the orchestrator
- **PNG volume in git.** 20 episodes × 6 strips × ~250 KB ≈ 30 MB under `reports/figures/step_07/`. Suggestion:
  write all strips to `runs/<run>/keyframes/` (`--out-dir`) and copy only the 3–5 illustrative stills into
  `reports/figures/step_07/`. The default currently follows the spec path.
- `ruff check .` reports 3 E501 errors in Step 6's files (`scripts/06_compute_advantages.py:4,295`,
  `tests/test_advantage.py:57`), which I don't own. My files are clean, and `pytest -q` passes (82 tests).
- The real renders need the held-out episodes' frame stores. The laptop has episodes 0–9 only, and only ep 0 is
  held-out, so run step 7 on the workstation or copy the 20 `ep_XXXX.h5` files (~0.5 GB) to the laptop.
- With noisy V, I_t flickers frame to frame and the green shading becomes a barcode (visible on fake data). That is
  faithful to the labels, but it may be worth checking alongside step 6's flicker diagnostic.

## Definition-of-done checklist
- [ ] 20 held-out mp4s rendered; keyframe strips saved — *tooling ready and validated on episodes 0–9 with fake values; real run pending*
- [ ] Per-episode table filled from actually looking at the frames — *template above; pending real values*
- [ ] Summary numbers + failure modes written up — *pending*
- [x] `reports/step_07_evaluation.md` written — *tooling version; to be completed after evaluation*
