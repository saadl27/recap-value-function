# Step 7 — Empirical evaluation

**Depends on:** step 6 (rendering code can be written earlier against fake values).
**Owns:** `src/recap_vf/viz.py`, `scripts/07_render_episode.py`, `scripts/07_keyframes.py`,
`reports/step_07_evaluation.md`, `reports/figures/step_07/`.

## Goal
Check, by looking at the robot, that the value function behaves sensibly: **V drops when the robot is about to fail
(or fails) and rises when it recovers**, and the positive/negative labels line up with good/bad behaviour. This is
the evaluation the paper itself uses (qualitative V plots); there is no ground truth for per-frame advantage.

Evaluate **only held-out episodes** (or, after cross-fitting, any episode — each is scored by a model that never saw
it). On training episodes a memorising model would look misleadingly good.

## Tools
### 7.1 `scripts/07_render_episode.py --run <run> --episode <e> [--advantages advantages_50_0.4]`
An mp4 per episode in `runs/<run>/videos/ep_XXXX.mp4`:
- top row: the three cameras side by side (top | left wrist | right wrist), timestamp overlay;
- bottom: V(t) (solid) and the true normalised return g(t) (dashed, for reference), A(t) on a second axis, frames
  with `I_t = True` shaded green, a vertical cursor at the current time;
- play every 2nd frame at 25 fps (real time); keep files small (H.264, CRF ~28).

### 7.2 `scripts/07_keyframes.py --run <run> --episode <e> --k 3`
Finds the **k most negative** and **k most positive** `A_t` (non-overlapping, at least 2 s apart) and saves, for
each, a PNG strip of the top + both wrist cameras at `t − 1 s, t, t + 1 s, t + 2 s` with V and A printed on each
frame → `runs/<run>/keyframes/ep_XXXX_{drop,rise}_{i}.png` (only the 3–5 best stills are copied to
`reports/figures/step_07/`, to keep the repo small). Agents can open these PNGs to judge what is
happening; humans can use the mp4s.

## Episode selection (fixed seed, list them in the report)
From held-out: 10 failures at random; 5 successes with the **deepest dip in V** (most likely fumbles/recoveries);
5 successes at random. Render all 20 mp4s; keyframes for all 20.

## What to judge, per episode
Fill a table in the report:

| ep | outcome | what visibly happens (time) | biggest V drop at | matches a visible problem? (±1 s) | lead time (V drop before visible failure) | recovery visible? | V rises at recovery? | notes |
|---|---|---|---|---|---|---|---|---|

Guidelines:
- A "visible problem" = missed grasp, actuator slips/drops, collision, arms stalling, actuator placed outside the
  tray, etc. Describe it concretely.
- For failures, also check whether V **ever** looked confident of success and when it changed its mind.
- For successful episodes with a dip: is the dip a real fumble that was then recovered (good), or a false alarm?
- Flag frames where `I_t = True` during clearly bad behaviour, or `False` during clearly good fast progress.

## Summary to report
- Fraction of failures whose largest V drop lines up (±1 s) with a visible problem; median lead time.
- Fraction of observed recoveries where V rises.
- False alarms: V dips in successful episodes with no visible problem.
- Typical failure modes of the value function (e.g. "reacts only after the drop, not before", "confused by occlusion
  in the top camera", "noisy while both arms are idle").
- 3–5 best illustrative stills for the team.
- The list of mp4 paths for Saad to review.

## Definition of done
- [ ] 20 held-out mp4s rendered; keyframe strips saved
- [ ] Per-episode table filled from actually looking at the frames
- [ ] Summary numbers + failure modes written up
- [ ] `reports/step_07_evaluation.md` written
