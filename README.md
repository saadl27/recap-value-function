# RECAP Value Function — Advantage Labelling for Robot Rollouts

Implementation of the **value function and advantage estimation** from the RECAP method of
[π\*0.6: a VLA That Learns From Its Experience](https://arxiv.org/abs/2511.14759) (Physical Intelligence, 2025),
applied to DreamMachines' bimanual actuator-unboxing rollouts.

**Goal:** for every timestep of every episode in
[`DreamMachines/20h_fullft_eval_success`](https://huggingface.co/datasets/DreamMachines/20h_fullft_eval_success),
decide whether the robot's behaviour at that moment has **positive advantage** (it makes progress toward success
faster/more reliably than the policy usually does) or not. These per-frame labels (`I_t ∈ {True, False}`) are what
an advantage-conditioned policy is later trained on.

> **Scope:** we train the *value function* and produce the *advantage labels*. We do **not** train a policy.

---

## Pipeline

```
 HF dataset (800 episodes, 50 fps, 3 cams, success flag)
        │
        ▼
 ┌────────────────────┐   ┌─────────────────────┐
 │ 1. Data pipeline   │   │ 2. Rewards & returns│   (independent, run in parallel)
 │ frames → JPEG store│   │ G_t, normalisation, │
 │ splits, Dataset    │   │ 201-bin targets     │
 └─────────┬──────────┘   └──────────┬──────────┘
           │      ┌────────────────────────────┐
           │      │ 3. Model                   │  (can be built in parallel on dummy data)
           │      │ SigLIP2-so400m + Gemma3-270M│
           │      │ → 201-bin value head       │
           │      └─────────────┬──────────────┘
           ▼                    ▼
        ┌─────────────────────────────────┐
        │ 4. Value-function training      │  RTX 5090
        │ cross-entropy on return bins    │
        └────────────────┬────────────────┘
                         ▼
        ┌─────────────────────────────────┐
        │ 5. Memorisation check           │  train vs held-out → earlier ckpt / regularise?
        └────────────────┬────────────────┘
                         ▼
        ┌─────────────────────────────────┐
        │ 6. Advantage labelling          │  A_t = Σr + V(o_{t+50}) − V(o_t), top-40% → I_t
        └────────────────┬────────────────┘
                         ▼
        ┌─────────────────────────────────┐
        │ 7. Empirical evaluation         │  does V drop before failures / rise on recovery?
        └─────────────────────────────────┘
```

| Step | Doc | Depends on | Status |
|---|---|---|---|
| 0 | [Environment setup](docs/steps/00_setup.md) | — | ✅ laptop · ✅ workstation (torch 2.11+cu128 on RTX 5090, sm_120) |
| 1 | [Data pipeline](docs/steps/01_data_pipeline.md) | 0 | ✅ done — 800 eps / 634 838 frames, 36 GB store, 10k items/s (16 workers) on workstation — [report](reports/step_01_data.md) |
| 2 | [Rewards & returns](docs/steps/02_returns.md) | 0 | ✅ done — [report](reports/step_02_returns.md) |
| 3 | [Value model](docs/steps/03_model.md) | 0 | ✅ done — 681 M params; 5090 full FT + grad-ckpt batch 32 = 11.8 GiB, overfit 5.30→0.0003 — [report](reports/step_03_model.md) |
| 4 | [Training](docs/steps/04_training.md) | 1, 2, 3 | ✅ `vf_c2229_s0` early-stopped at step 1000 (2.6 h); best held-out CE at step 250; train/held-out gap grows from step 500 |
| 5 | [Memorisation check](docs/steps/05_memorization_check.md) | 4 | ⬜ todo |
| 6 | [Advantage labelling](docs/steps/06_advantage.md) | 4 (5) | 🟨 code done (validated on fake values), real labels pending — [report](reports/step_06_advantage.md) |
| 7 | [Evaluation](docs/steps/07_evaluation.md) | 6 | 🟨 tooling done, evaluation pending — [report](reports/step_07_evaluation.md) |

**Method v2** ([explanation](docs/method_v2.md)): cross-fitted MC teacher → bootstrapped critic. **Proposed, not implemented** (no time).

| Step | Doc | Depends on | Status |
|---|---|---|---|
| 8 | [Failure annotations & yardstick](docs/steps/08_failure_annotations.md) | 1, 4 | 📝 specified, not implemented |
| 9 | [Cross-fitted critics → labels v1](docs/steps/09_crossfit.md) | 4, 6, 8 (S) | 📝 specified, not implemented |
| 10 | [Bootstrapped critic → labels v2](docs/steps/10_bootstrapped_critic.md) | 9 | 📝 specified, not implemented |
| 11 | [Label comparison](docs/steps/11_label_comparison.md) | 8, 9, 10 | 📝 specified, not implemented |
| 12 | [Failure penalty at the mistake](docs/steps/12_failure_time_penalty.md) | 8–11 | 📝 idea only (gated) |

Each agent writes its results to `reports/step_XX.md` (see [conventions](docs/conventions.md#reports)).

---

## Repository layout

```
.
├── README.md                 ← you are here
├── CLAUDE.md                 ← short rules for coding agents working in this repo
├── pyproject.toml            ← single uv-managed environment
├── docs/
│   ├── background.md         ← RECAP paper summary, verified facts, reference implementations
│   ├── conventions.md        ← code style, paths, file contracts between steps, reports
│   ├── open_questions.md     ← things to confirm with the DreamMachines team
│   └── steps/                ← one detailed spec per pipeline step (hand these to agents)
├── src/recap_vf/             ← the python package
│   ├── constants.py          ← dataset id, cameras, fps, prompt (shared, do not duplicate)
│   ├── paths.py              ← where data / runs live (env-overridable)
│   └── bins.py               ← 201-bin value support, targets, expected value (shared)
├── scripts/                  ← entry points, numbered by step (01_*.py, …) + remote.sh (laptop ↔ workstation)
├── configs/                  ← YAML configs for returns / training / advantage
├── tests/                    ← pytest unit tests
├── reports/                  ← one markdown report per step (results, plots, decisions)
├── data/      (git-ignored)  ← raw HF download + derived caches
└── runs/      (git-ignored)  ← checkpoints, tensorboard logs, value/advantage outputs
```

## Quickstart

Development and smoke tests happen on a laptop; full runs on the office RTX 5090 after `git pull`.

```bash
uv sync                          # .venv with torch cu128 (works on RTX 40xx and 50xx)
uv run pytest -q                 # unit tests (CPU)
uv run python scripts/01_download.py   # then follow docs/steps/ in order
```
Full commands for each step (smoke scale and full scale) are in its doc and its report. Workstation rules and the
optional `scripts/remote.sh` helper are in [conventions → Machines](docs/conventions.md#machines).

## Key decisions (so far)

| Topic | Decision | Why |
|---|---|---|
| Backbone | SigLIP2-so400m-patch14-224 + Gemma-3-270M (≈670M) | Matches paper ("670M VLM initialised from Gemma 3"); same as RLinf's open implementation |
| Value head | 201 bins on [−1, 0], cross-entropy | Paper |
| Estimator | baseline: Monte-Carlo returns (paper) · proposed v2: student on 50-step distributional TD targets from the out-of-fold MC teacher | same estimand, lower-variance targets ([method v2](docs/method_v2.md)) |
| Reward | −1 per frame, 0 at success end, −C_fail at failure end | Paper |
| Normalisation | divide by \|min return in dataset\| | RLinf; maps to [−1, 0] |
| Frames | all frames, uniformly sampled, training measured in steps | RLinf/OpenTau use all frames |
| Training budget | ≤ one pass over train frames (2 100 steps × 256, ~4–5 h), constant LR, early stop on held-out CE | 8 000 steps (RLinf) would take ~18 h on one 5090 |
| Augmentation | none | not in paper; decided 2026-10-12 |
| Cross-fitting | baseline: not used · proposed v2: 5 folds, every episode labelled by a model that never saw it | baseline memorises train outcomes from step ~750 ([method v2](docs/method_v2.md)) |
| Advantage | N = 50 frames (1 s) lookahead; top 40 % positive | Paper (post-training on rollouts) |
| Human interventions ([DAgger data](https://huggingface.co/datasets/DreamMachines/actuator_unboxing_1h_HQ_dagger_rollouts), `RECAP_DATASET=actuator_unboxing_1h_HQ_dagger_rollouts`) | −C before each takeover, −1 elsewhere, return-to-go **cut at the next takeover**; advantage window stops there; human frames forced positive; run with `scripts/run_dagger_pipeline.sh` | each penalty blames only the frames that led to it ([interventions](docs/interventions.md)) |
| Evaluation | qualitative on held-out episodes: V drops before failure, rises on recovery | Paper only evaluates qualitatively |

## References

- Paper: [arXiv 2511.14759](https://arxiv.org/abs/2511.14759) · [blog](https://www.pi.website/blog/pistar06)
- RLinf RECAP (closest open implementation): [docs](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/recap.html) · [code](https://github.com/RLinf/RLinf)
- LeRobot RECAP PR (unmerged): [#3245](https://github.com/huggingface/lerobot/pull/3245)
- OpenTau RECAP tutorial: [docs](https://opentau.readthedocs.io/en/latest/tutorials/RECAP.html)
- No official implementation: [openpi #857](https://github.com/Physical-Intelligence/openpi/issues/857)
