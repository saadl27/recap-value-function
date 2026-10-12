# Background: RECAP value function & advantage estimation

This page is the shared mental model for everyone (human or agent) working in this repo. Read it before any step doc.

## 1. What RECAP does, in one paragraph

RECAP (*RL with Experience and Corrections via Advantage-conditioned Policies*) improves a VLA policy from its own
rollouts without online RL. It (1) trains a **value function** `V(o_t)` that predicts how good the current situation is
— concretely, the (negative, normalised) number of steps left until success, with failures heavily penalised;
(2) uses it to compute an **advantage** for each timestep — did the next ~1 s of behaviour move us closer to success
than expected?; (3) converts that into a binary **improvement indicator** `I_t = 1[A_t > ε]`; and (4) trains the policy
conditioned on `I_t` (text token "Advantage: positive/negative"), then at test time always asks for "positive".
**This repo implements (1)–(3).**

## 2. Verified facts from the paper

Checked against the arXiv HTML text (sections IV–V and Appendix A). "—" means the paper does not say.

| Item | What the paper says |
|---|---|
| Backbone | "smaller **670M** parameter VLM backbone that is also initialized from **Gemma 3**" |
| Inputs | same as the VLA: camera images, robot configuration (state), language prompt |
| Reward | −1 every step; at the final step **0 if success**, **−C_fail if failure** ("a large constant chosen so as to ensure that failed episodes have low values"). C_fail value: — |
| Normalisation | values "normalized per task based on the maximum episode length of the task" into **(−1, 0)** |
| Value head | distributional: return discretised into **B = 201 bins**, trained with **cross-entropy**; scalar value = expectation over bins |
| Estimator | **Monte-Carlo**: "a Monte Carlo estimator for the value function of the policy represented by the dataset" |
| Advantage | n-step: `A_t = Σ_{t'=t}^{t+N−1} r_{t'} + V(o_{t+N}) − V(o_t)`; **N = 50** in post-training; N = T (full MC) in pre-training |
| Threshold ε | chosen per task so that **~30 %** of (demo) pre-training data is positive and **~40 %** of evaluation rollouts are positive in fine-tuning; ~10 % for some laundry tasks |
| Human corrections | always forced positive (`I_t = True`) |
| Regularisation | value function co-trained on "a small mixture of multi-modal web data" to prevent overfitting |
| Value function evaluation | **qualitative only** (plots of V over episodes, Fig. 4 & App. A-B): "the VF correctly identifies mistakes in the episode, as well as the speed of progress" |
| Augmentation, sampling, LR, steps | — |

Our dataset is **evaluation rollouts of a fine-tuned policy**, i.e. the paper's *post-training* regime → N = 50, ~40 % positive.

## 3. The maths we implement

Let an episode have `T` frames, `t = 0 … T−1`, success flag `s ∈ {0,1}`.

**Reward**
```
r_t = −1                       for t < T−1
r_{T−1} = 0        if s = 1
r_{T−1} = −C_fail  if s = 0
```

**Return-to-go (undiscounted, γ = 1)**
```
G_t = Σ_{t'≥t} r_{t'} = −(T−1−t) − C_fail · 1[s = 0]
```

**Normalisation** — one constant `Z` for the whole dataset (single task):
```
Z = |min_{episodes, t} G_t| = (T_max_fail − 1) + C_fail     (if the longest episode is a failure)
g_t = G_t / Z ∈ [−1, 0]
```
`Z` is saved once (step 2) and reused everywhere (training targets, advantage reward sums).

**Distributional value.** Support `v_b = −1 + b/200`, `b = 0 … 200`. The model outputs logits `ℓ ∈ R^201`,
`p = softmax(ℓ)`, `V(o_t) = Σ_b p_b v_b`. Loss = cross-entropy against the bin nearest to `g_t`
(optionally HL-Gauss soft targets). Code: [`src/recap_vf/bins.py`](../src/recap_vf/bins.py).

**Advantage (N-step, normalised units)**
```
if t + N < T:  A_t = (G_t − G_{t+N}) / Z + V(o_{t+N}) − V(o_t)      # = −N/Z + V(o_{t+N}) − V(o_t)
else:          A_t = G_t / Z − V(o_t)                               # episode ends inside the window; V(after end) = 0
```

**Indicator.** `ε = quantile(A, 1 − p_pos)` over all frames, `I_t = 1[A_t ≥ ε]`, default `p_pos = 0.4`.

### Intuition worth internalising
- In a **successful** episode an accurate V rises by exactly `1/Z` per frame, so `A_t ≈ 0` when the robot progresses
  "as fast as usual", `> 0` when faster / when it resolves uncertainty in its favour (e.g. secures a grasp), `< 0` when
  it stalls or fumbles.
- In a **failed** episode V should *drop* around the moment the failure becomes likely (drop, bad grasp, collision) —
  the large negative `A_t` there is exactly the signal we want.
- Because the estimator is Monte-Carlo, `V` is the value of **the policy that collected the data**. The start state of
  a failed episode looks like the start of a successful one, so `V(o_0)` should be roughly the average outcome, not
  "this episode will fail". If the network *does* know at `t = 0` that a *training* episode fails, it has memorised —
  see [step 5](steps/05_memorization_check.md).
- With the `−1`-per-step reward, a failed episode that drags on also has `G_t` increasing toward `−C_fail/Z`. This is
  inherent to the paper's design; the large `C_fail` keeps failure frames far below success frames.

## 4. Reference implementations (none official)

| Project | Value backbone | Bins | C_fail | Notes |
|---|---|---|---|---|
| **[RLinf](https://github.com/RLinf/RLinf)** (`examples/offline_rl/advantage_labeling/recap`, `rlinf/models/embodiment/value_model/recap`) | SigLIP2-so400m-224 + Gemma-3-270M; value read from a learned CLS token through a tiny "critic expert" | 201 on [−1,0] | −300 (LIBERO) | all frames, openpi-style augmentation on by default, no robot state, LR 5e-5 backbone / 1e-4 head, batch 256, 8k steps, positive quantile 0.3, normalises by \|min return\| |
| [LeRobot PR #3245](https://github.com/huggingface/lerobot/pull/3245) | PaliGemma (π0.5 base), mostly frozen | 50 | −500 | unmerged |
| [OpenTau](https://opentau.readthedocs.io/en/latest/tutorials/RECAP.html) | — | 201 | −1000 (normaliser 1600) | N = 50, 30 % positive |

We re-implement the RLinf recipe in a small single-GPU PyTorch codebase rather than adopting RLinf itself (it pulls
in FSDP/Hydra/openpi/LIBERO-specific plumbing that does not fit a 3-camera bimanual robot on one RTX 5090).

## 5. Our dataset at a glance

`DreamMachines/20h_fullft_eval_success` (LeRobot v3.0, `bi_dk1_follower`):

| | |
|---|---|
| Episodes | 800 (423 success / 377 failure) — `success` column in `meta/episodes/…parquet` |
| Frames | 634 838 at **50 fps** (~3.5 h) |
| Episode length (frames) | success: median 586, min 382, max 2077 · failure: median 749, min 157, max **2229** |
| Cameras | `observation.images.top`, `observation.images.left_wrist`, `observation.images.right_wrist` — 224×224 H.264, one mp4 per camera containing **all episodes concatenated** |
| State / action | 14-D joint positions (6 joints + gripper per arm) |
| Task | single prompt: "Take an actuator from the box and place it in the tray" |
| Export flags | `trim_idle=true`, `interventions=false` (no human-correction flag in the data) |
| Size | ~5.3 GB (videos ~5.2 GB, tabular 59 MB) |
