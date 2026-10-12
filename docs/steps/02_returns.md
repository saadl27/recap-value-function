# Step 2 — Rewards, returns and value targets

**Depends on:** step 0 (only `meta/episodes` is needed — download it with
`snapshot_download(..., allow_patterns=["meta/*"])` if step 1 is not done). **Parallel with:** steps 1 and 3.
**Owns:** `src/recap_vf/returns.py`, `scripts/02_compute_returns.py`, `configs/returns.yaml`,
`tests/test_returns.py`, `reports/step_02_returns.md`, `reports/figures/step_02/`.

## Goal
For every frame, the Monte-Carlo **return-to-go** in raw and normalised units and its **target bin** for the
201-bin value head, for one or more choices of the failure penalty `C_fail`. These are the training labels.

## Definitions (from the paper; see [background §3](../background.md#3-the-maths-we-implement))
Per episode with length `T` and success `s`, frames `t = 0 … T−1`, γ = 1:
```
r_t      = −1 for t < T−1;   r_{T−1} = 0 (success) or −C_fail (failure)
G_t      = −(T−1−t) − C_fail · 1[failure]
Z        = |min over all frames of G_t|          (one constant for the dataset)
g_t      = G_t / Z ∈ [−1, 0]                    → column `return_norm`
bin_t    = bins.value_to_bin_np(g_t)            → column `bin_index`
```
Use `recap_vf.bins` for the bin mapping (do not re-implement it). Reward is per **frame** (50 Hz), same as the
paper's 50 Hz control steps, so `N = 50` later means 1 s.

## Choosing `C_fail`
The paper only says "large". Its effect:
- Small `C_fail` → a failure looks like a slow success; V mostly measures time-to-go.
- Huge `C_fail` → failure frames all collapse near −1; inside a failed episode almost no temporal resolution;
  V ≈ −(C_fail/Z)·P(failure).
- References: RLinf −300 (LIBERO), LeRobot PR −500, OpenTau −1000 with normaliser 1600 (≈ C_fail + max length).

Our longest episode is a **failure** of 2229 frames; the longest success is 2077 frames.
**Default: `C_fail = 2229` (= T_max), tag `c2229`** — *our* choice, not the paper's (it gives no value). Any
`C_fail > T_max_success − 1 = 2076` makes every failure frame worth less than every success frame (the strictest
reading of "ensure that failed episodes have low values"); `C_fail = T_max` is a round, dataset-agnostic rule with
that property. OpenTau's values (C = 1000, normaliser 1600) imply the same property.
With it → `Z = 2228 + 2229 = 4457`; successes occupy
`g ∈ [−0.466, 0]` (bins 107–200), failures `g ∈ [−1, −0.5]` (bins 0–100) — the two never overlap, and each still
spans ~95–100 bins.
Also produce **`c1115` (0.5 × T_max)** and **`c4458` (2 × T_max)** for a later ablation (cheap; only labels change).

## Tasks
1. `src/recap_vf/returns.py`
   - `compute_episode_returns(length: int, success: bool, c_fail: float) -> dict[str, np.ndarray]`
     returning `reward`, `return_raw` (float64 internally, store float32).
   - `compute_returns_table(episodes: pd.DataFrame, c_fail: float) -> tuple[pd.DataFrame, dict]` building the full
     table (schema in [conventions](../conventions.md#file-contracts-between-steps)) and the metadata dict
     `{tag, c_fail, Z, num_bins, v_min, v_max, t_max, t_max_success, t_max_failure}`.
   - `load_returns(tag) -> (DataFrame, dict)` helper used by steps 4 and 6.
   - Global `index` for frame `t` of episode `e` = `dataset_from_index[e] + t` (verify against the frame table if
     step 1 has produced it).
2. `scripts/02_compute_returns.py --c-fail 2229 1115 4458` → `data/cache/returns_{tag}.parquet` + `.json`
   (tag = `c{int(c_fail)}`).
3. Figures in `reports/figures/step_02/`:
   - histogram of `return_norm` for success vs failure frames (per tag);
   - `return_norm` vs `t / T` for ~10 example episodes of each class;
   - bin occupancy: number of frames per bin (shows how much of the 201-bin support is used and how imbalanced it is).
4. Report the class balance in **frames** (failure frames are ~56 % of all frames because failures are longer) and
   per-bin counts — step 4 needs this to decide whether to re-weight.

## Tests (`tests/test_returns.py`)
- Success episode: last frame `G = 0`, first frame `G = −(T−1)`, increments of exactly +1 per frame.
- Failure episode: last frame `G = −C_fail`, first frame `G = −(T−1) − C_fail`.
- `min(return_norm) == −1` exactly (at frame 0 of the longest failure), `max == 0`, all within [−1, 0].
- `bin_index ∈ [0, 200]`; for `c2229`: success frames only in bins 107–200, failure frames only in bins 0–100.
- Row count = sum of lengths; `index` unique and contiguous.

## Definition of done
- [ ] Three returns tables + metadata JSONs written (default `c2229`)
- [ ] Numbers in report: Z per tag, ranges of `return_norm` per class, frame counts per class, bin occupancy
- [ ] Figures saved and described
- [ ] Tests pass, ruff clean
- [ ] `reports/step_02_returns.md` written
