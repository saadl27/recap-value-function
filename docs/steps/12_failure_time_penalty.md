# Step 12 — Failure penalty at the mistake (GATED: do not start without Saad's go)

**Depends on:** steps 8–11. Background: [method v2](../method_v2.md).

## Idea
Failed episodes get `C_fail` at the human takeover, which is the episode end. The frames between the mistake and the
takeover then get advantages near the threshold, so noise decides their labels. Examples:
- ep 155: about 18 s of fumbling after the actuator ends up lying in the tray at about 9 s;
- ep 364: the robot retracts, a human hand reaches into the tray, then about 20 s of a still scene.

The idea is to move the terminal penalty to the annotated or detected failure time `t_fail`.

## Decisions needed before any work
1. **Source of `t_fail` for all 377 failures:**
   - (a) the OOF critic's largest drop, used only when it is large and step 8/11 showed a high `drop_hit` rate,
     otherwise keep the episode end;
   - (b) a vision-language-model pass;
   - (c) human annotation.
2. **Frames after `t_fail`:**
   - (a) exclude them from value training and force `I_t = False`;
   - (b) keep them with a flat target of −C_fail/Z.
3. **Returns:** `r = −1` up to `t_fail`, `−C_fail` at `t_fail`. `Z` is recomputed → a new returns tag (e.g. `c2229_tf`).

## Evidence so far
- ROVE assigns its penalty at a chosen boundary (the end of the human "adaptation" phase) rather than at the
  takeover onset. It reports 80 % vs 50 % policy success. However, that ablation row also changes the critic horizon
  (16 → 50 steps), so it does not isolate the penalty timing.
- ROVE's boundary is not "the mistake".
- There is no direct evidence for our variant yet. Steps 8 and 11 must first show that `t_fail` can be found
  reliably.
