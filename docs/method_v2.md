# Method v2 — cross-fitted teacher + bootstrapped critic

**Status:** proposed, **not implemented** (no time; decided 2026-10-12). This page explains how to improve the
baseline's labels and why. The detailed implementation specs are steps 8–12 in [`docs/steps/`](steps/).
Everything not mentioned here stays as in the baseline (backbone, 201 bins on [−1, 0], reward, `C_fail = 2229`,
`Z = 4457`, `N = 50` advantage, 40 % positive threshold, optimiser).

**In one paragraph.** The baseline's Monte-Carlo targets give every frame of an episode that episode's outcome, so
each label says "this episode failed", not "this action caused the failure". The trained critic still learns *where*
things go wrong, because the same-looking states lead to different outcomes across episodes: on held-out episodes V
drops at the mistake. On training episodes, though, the network memorises the outcomes and the localisation is lost.
The fix has two stages:
1. Score every episode with a model that never saw it (**cross-fitting**).
2. Optionally, distil those values into one critic trained on **bootstrapped targets**, i.e. the teacher's value
   one second later instead of the episode's outcome. These targets drop at the mistake and stay high before it.

## 1. Why: what the baseline does and does not do

Numbers from `runs/vf_c2229_s0` (workstation: `metrics.jsonl`, `eval/val_{heldout,train}_step*.parquet`, stride 10)
and the comparison videos in `runs/viz_compare/` (checkpoints 250 / 500 / 750 / 1000).

**On held-out episodes the Monte-Carlo (MC) critic does what we want.**
- V drops within ~1 s of the visible mistake in held-out failures 409 (handover drop, ~4.5 s), 588 (~5.5 s),
  750 (actuator taken sideways, ~7.5 s) and 155 (~9 s). It misses 723, where no cause is visible either.
- V dips at recovered fumbles in held-out successes and rises when they are fixed: 707 (7–13 s), 771 (6–8 s),
  218 (5–7 s).
- V(o₀) ≈ −0.41 for both outcomes. That is the outcome-weighted average, as an honest MC value should be.

**On training episodes, which are 85 % of what we label, it partly memorises each episode's outcome.**

| signal (train / held-out) | step 500 | step 1000 |
|---|---|---|
| mean V over the first 10 % of the episode, success − failure | 0.008 / 0.002 | **0.14 / 0.03** |
| auc@0.25 (success predicted from V at 25 % of the episode) | 0.53 / 0.60 | **0.89 / 0.56** |
| MAE on success frames | 0.16 / 0.17 | **0.10 / 0.20** |
| successes whose V has a dip > 0.2 | 16 % / 29 % | **55 % / 79 %** |

Consequences:
- Bad actions inside training successes are masked: V stays high through a fumble.
- Training failures look bad before the mistake happens.

The cause is that the MC target of every frame is the episode's own outcome: a straight line per episode with no
step at the mistake. Localisation comes only from generalising across episodes, and memorisation undoes it.

**Other observations.**
- *Checkpoint trade-off:* step 250 underfits (it stays near the average). Step 1000 is overconfident: held-out
  entropy falls from 3.92 to 3.07, held-out failure CE rises from 4.67 to 5.68, and 89 % of held-out successes touch
  the failure level. On held-out ep 750, step 1000 gives V = −0.75 at t = 0.
- *Flicker:* in ambiguous states V jumps between ≈ −0.1 and ≈ −0.55. In the 20 eval episodes, 10–40 % of frames
  have |A| > 0.1, while the progress term is only N/Z = 0.011. Step 500 flickers least.
- *Failure end:* failed episodes end at the human takeover, not at the mistake. The frames in between (ep 155: 18 s
  of fumbling; ep 364: retract, a human hand, then ~20 s of a still scene) get advantages near the threshold, so
  noise decides their labels. This is addressed by step 12 (gated), not by v2.

## 2. The method

**Stage 1 — cross-fitting (step 9).** Split the 800 episodes into the 5 folds of `splits.json` (160 each,
stratified). Fold model *k* trains with the baseline recipe on the other 640 episodes for a fixed budget `S` steps,
chosen in step 8. It then scores fold *k*. Every frame thus gets an **out-of-fold (OOF)** value from a model that
never saw its episode: held-out quality for all 800 episodes. Labels **v1** = step-6 labels from these values.
This is the RECAP estimator unchanged; only the train/label split differs.

**Stage 2 — bootstrapped critic (step 10).** Train one new critic (the "student") on targets built from the OOF
teacher one second ahead instead of from the episode's outcome. With teacher distribution `p_t` (201 bins),
`N = 50`, `Z = 4457`:

```
t + N <  T :  q_t = Proj( p^oof_{t+N} shifted by −N/Z )      # distributional N-step TD target
t + N >= T :  q_t = two-hot( g_t )                           # true return; the episode ends inside the window
loss       =  − Σ_b q_t,b · log softmax(ℓ(o_t))_b
```

`E[q_t] = −N/Z + V^oof(o_{t+N})`. If the teacher equals V^π, then `E[target | o_t] = V^π(o_t)`. The student
therefore estimates the **same quantity as RECAP's MC critic** (the value of the data-collecting policy), but from
lower-variance targets that do not contain the episode's outcome:
- Before a mistake, the target is "what the teacher thinks 1 s later", which is high.
- At the mistake, the target drops.
- During a recovered fumble, the target is low.

Memorising these targets is harmless, because they are functions of observations, not of the secret outcome.
Labels **v2** = step-6 labels from the student's values.

**What it cannot do.** The student cannot know more than the teacher about what a failure looks like. The
held-out curves are roughly the ceiling. The expected gains are:
1. training-episode labels at held-out quality (most of this already comes with v1);
2. smoother curves;
3. one reusable critic for the next round of rollouts. For new data, averaging the 5 fold models is an alternative
   that needs no distillation.

## 3. Deviations from the paper, and sources

| Choice | Source / reason |
|---|---|
| Cross-fitting | Not in the paper. The paper has far more data and co-trains the value function on web data "to prevent overfitting". Our memorisation numbers above are the justification. |
| Bootstrapped (TD) targets | The paper uses MC and calls it "less optimal than a more classic off-policy Q-function estimator", but "simple and highly reliable", and leaves off-policy estimators to future work. [ROVE](https://arxiv.org/abs/2606.17011) pre-trains an MC critic, then fine-tunes it with 16-step TD plus expectile regression; qualitatively, its MC critic is "noisy, less discriminative". [ALOE](https://arxiv.org/abs/2602.12691): TD beat "MC returns as RECAP" (65.1 vs 58.3 % policy success), but that ablation also removes ALOE's off-policy component. |
| Shift-and-project targets on the 201 bins | C51 projection (Bellemare et al. 2017); two-hot as in MuZero / DreamerV3. These keep the target mean exact. A nearest-bin one-hot of a continuous target would bias it by up to ½ bin ≈ 11 frames. |
| Student initialised from the pretrained backbones | Same as the baseline and the paper ("both the value function and policy are finetuned from the pre-trained checkpoint"). ROVE instead initialises its TD critic from the MC critic. |
| **Not adopted (need Saad's decision)** | Expectile / optimistic value (ROVE): it changes what V estimates and needs a design for a distributional head. Iterating stage 2. TD(λ) mixtures. Penalty at the mistake (step 12). Web-data co-training. |

## 4. Implementation plan (specified, not built)

| Step | What | GPU (RTX 5090) |
|---|---|---|
| [8](steps/08_failure_annotations.md) | Blind failure-time and fumble annotations of 240 val episodes; localisation yardstick; choose `S` | none |
| [9](steps/09_crossfit.md) | 5 fold critics → OOF values + distributions → labels v1 | ~7 h train (S = 500) + ~3 h inference |
| [10](steps/10_bootstrapped_critic.md) | Bootstrapped targets → student critic → labels v2 | ≤ 5.5 h train + ~3 h inference |
| [11](steps/11_label_comparison.md) | Baseline vs v1 vs v2 on the yardstick, memorisation signals, label diagnostics, videos, blind sheets | none (renders on CPU) |
| [12](steps/12_failure_time_penalty.md) | **Gated**: failure penalty at the mistake instead of at the takeover | — |

Steps 9 and 10 were meant to run unattended as one resumable chain (`scripts/run_v2_pipeline.sh S`), about 16–18 h
of RTX 5090 time with S = 500. None of it exists yet.

## 5. How it would be judged (step 11)
1. On the **val_train** episodes, v1 and v2 should reach the held-out numbers of the yardstick: drop-at-mistake hit
   rate, drop size, pre-failure V and dips at fumbles. The baseline in-sample values should not.
2. On held-out episodes, v1 and v2 should be at least as good as the baseline.
3. Labels should be less noisy: longer runs, a lower robust jitter ratio, and fewer frames decided by flicker.
4. Saad's blind choice between the curves on 30 episodes.
