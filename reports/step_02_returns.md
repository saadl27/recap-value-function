# Step 02 — Rewards, returns and value targets
**Status:** done        **Agent/date:** Claude (step-2 agent), 2026-10-12

## Summary
- Built `src/recap_vf/returns.py` (per-episode rewards / Monte-Carlo return-to-go, full returns table, metadata,
  save/load helpers, index verification, summary + bin counts), the thin entry point
  `scripts/02_compute_returns.py`, `configs/returns.yaml` and `tests/test_returns.py` (8 tests).
- Wrote three returns tables + JSONs for the full dataset (800 episodes, 634 838 frames, 7 s on the laptop CPU):
  `data/cache/returns_c2229.{parquet,json}` (**default**), `returns_c1115.*`, `returns_c4458.*`.
- Global `index = dataset_from_index + frame_index` verified **row-for-row** against the raw LeRobot frame table
  (`data/raw/.../data/chunk-000/file-000.parquet`): all 634 838 `(index, episode_index, frame_index)` triples match.
- All numbers given in the brief are reproduced exactly: **c2229 → Z = 4457**, success `g ∈ [−0.4658, 0]`
  (bins 107–200), failure `g ∈ [−1, −0.5001]` (bins 0–100), no overlap.
- Frame class balance: **55.5 % failure frames** (352 251) vs 44.5 % success (282 587). Within each class the bins
  are very imbalanced too (up to ~650× between the fullest and the emptiest used bin); see "Bin occupancy".

## How to reproduce
```bash
# needs only data/raw/20h_fullft_eval_success/meta/episodes/... (and data/... for the index check)
env -u PYTHONPATH uv run pytest -q tests/test_returns.py
env -u PYTHONPATH uv run python scripts/02_compute_returns.py --max-episodes 10 --no-figures   # smoke: writes returns_c*_smoke10.*
rm -f data/cache/returns_c*_smoke10.*                                                          # clean up smoke output
env -u PYTHONPATH uv run python scripts/02_compute_returns.py                                  # full: tags from configs/returns.yaml
# equivalent explicit form:
env -u PYTHONPATH uv run python scripts/02_compute_returns.py --c-fail 2229 1115 4458 --seed 0
```
Flags: `--config`, `--c-fail …` (overrides config), `--max-episodes N` (smoke scale, tags get suffix `_smoke{N}` so
real tables are never overwritten), `--raw-dir`, `--no-cache` (ignore `data/cache/episodes.parquet` and read raw meta),
`--no-verify`, `--no-figures`, `--seed` (only used to pick the example episodes in the plot).
Episode source: `data/cache/episodes.parquet` from step 1 if it exists, else the raw `meta/episodes/*.parquet`
(it did not exist at run time, so the raw meta was used). Frame-table check: `data/cache/frames.parquet` if it exists,
else the raw `data/chunk-*/file-*.parquet`.

Downstream usage (steps 4 and 6):
```python
from recap_vf.returns import load_returns
table, meta = load_returns("c2229")   # table keyed by `index`, sorted, 634 838 rows; meta["Z"] == 4457.0
```

## Results

### Dataset (from `meta/episodes`)
| | count | longest |
|---|---|---|
| Episodes | 800 (423 success / 377 failure) | 2229 frames (failure) |
| Frames | 634 838 | longest success 2077 |

### Normaliser and `return_norm` ranges per tag
`Z = |min G_t| = (2229 − 1) + C_fail` in every tag (the longest episode is a failure).

| tag | C_fail | Z | success `g` range | success bins | failure `g` range | failure bins | classes overlap? |
|---|---|---|---|---|---|---|---|
| **c2229** | 2229 | **4457** | [−0.4658, 0] | 107–200 (94 bins) | [−1, −0.5001] | 0–100 (101 bins) | no (bins 101–106 empty) |
| c1115 | 1115 | 3343 | [−0.6210, 0] | 76–200 (125 bins) | [−1, −0.3335] | 0–133 (134 bins) | **yes**: bins 76–133 |
| c4458 | 4458 | 6686 | [−0.3105, 0] | 138–200 (63 bins) | [−1, −0.6668] | 0–67 (68 bins) | no (bins 68–137 empty) |

`min(return_norm) = −1` exactly (frame 0 of episode with T = 2229), `max = 0` exactly (last frame of every success).

### Frame class balance (identical for every tag)
| class | episodes | frames | share of frames |
|---|---|---|---|
| success | 423 (52.9 %) | 282 587 | 44.5 % |
| failure | 377 (47.1 %) | 352 251 | **55.5 %** |

### Bin occupancy (frames per bin; full per-bin counts in `reports/figures/step_02/bin_counts_{tag}.csv`)
| tag | bins used / 201 | empty | min non-empty | median non-empty | max (bin) | bins < 100 frames | bins < 1000 frames | per-bin median success / failure |
|---|---|---|---|---|---|---|---|---|
| **c2229** | 195 | 6 | 15 (bin 107) | 1842 | 9729 (186) | 18 | 64 | 798 / 2592 |
| c1115 | 201 | 0 | 30 | 2306 | 7191 (178) | 1 | 50 | 613 / 1938 |
| c4458 | 131 | 70 | 21 | 2646 | 14 382 (190) | 9 | 32 | 1213 / 3656 |

Observations relevant for step 4 (re-weighting decision):
- **Success mass is concentrated near 0**: 94 % of success frames have `g ≥ −0.2` (c2229) — most successes are short
  (median 586 frames ≈ 0.13 Z). The low-success bins 107–~125 (long, slow successes) have only 15–100 frames each.
- **Failure mass is concentrated near −0.5 … −0.6** (the end of failed episodes); the bins near −1 (start of the
  longest failures) have ~700–900 frames each.
- Edge bins hold less mass by construction (nearest-centre rounding covers only half a bin width at the ends):
  bin 200 has 5076 frames vs ~9300 in its neighbours, bin 0 only a few dozen (c2229), and for c2229 bin 100 only gets
  values in [−0.5025, −0.5001].
- With one-hot targets every bin is supported by ≥ 15 frames, so plain cross-entropy is well-posed; the imbalance is
  ~650× between extremes. Whether to re-weight is step 4's call; HL-Gauss soft targets (`bins.hl_gauss_targets`)
  would also smooth the sparse low-success bins.

### Figures (`reports/figures/step_02/`)
| file | content |
|---|---|
| `return_hist_{tag}.png` | histogram of `return_norm`, success (blue) vs failure (orange) frames, one bar = one value bin. For c2229 the two classes are separated by a gap at ≈ [−0.50, −0.47]; success frames pile up in [−0.1, 0], failure frames in [−0.65, −0.5]. |
| `return_examples_{tag}.png` | `return_norm` vs `t/T` for 10 random success and 10 random failure episodes (seed 0). Each is a straight line with slope 1/Z per frame; successes end at 0, failures at −C_fail/Z (−0.500 for c2229, −0.334 for c1115, −0.667 for c4458). Long failures start near −1, short failures barely below their end value. |
| `bin_occupancy_{tag}.png` | stacked frames-per-bin bars (linear and log scale), showing the unused gap between classes and the imbalance. |
| `bin_counts_{tag}.csv` | per-bin frame counts (`bin_index, success, failure, total`) — machine-readable for step 4. |
| `summary.json` | the key numbers above for all three tags. |

### Output files
| file | rows / size |
|---|---|
| `data/cache/returns_c2229.parquet` + `.json` | 634 838 rows, 4.9 MB |
| `data/cache/returns_c1115.parquet` + `.json` | 634 838 rows, 4.3 MB |
| `data/cache/returns_c4458.parquet` + `.json` | 634 838 rows, 4.9 MB |

Parquet schema (contract): `index:int64, episode_index:int64, frame_index:int64, success:bool, episode_length:int64,
reward:float32, return_raw:float32, return_norm:float32, bin_index:int64`, sorted by `index` (0 … 634 837, contiguous).
`return_raw` is computed in float64 and stored as float32 (integers ≤ 6686 → exact). `bin_index` is computed from the
float64 `return_norm` via `bins.value_to_bin_np`.

JSON for c2229:
```json
{"tag": "c2229", "c_fail": 2229.0, "Z": 4457.0, "num_bins": 201, "v_min": -1.0, "v_max": 0.0,
 "t_max": 2229, "t_max_success": 2077, "t_max_failure": 2229, "seed": 0, "num_episodes": 800, "num_frames": 634838}
```

## Deviations from the spec
- The metadata JSON is a **superset** of the contract: besides `{tag, c_fail, Z, num_bins, v_min, v_max, t_max}` and the
  spec's `t_max_success, t_max_failure`, it records `seed`, `num_episodes`, `num_frames`. No keys removed.
- Extra (non-required) helpers in `returns.py`: `load_episodes`, `verify_index`, `save_returns`, `summarize_returns`,
  `bin_counts`, `returns_tag`. Extra artefacts: `bin_counts_{tag}.csv` and `summary.json` in the figures dir.
- Smoke runs (`--max-episodes N`) write tags with a `_smoke{N}` suffix (and Z is computed on the subset) so they cannot
  overwrite the real tables.

## Open issues / questions for the orchestrator
- **c1115 lets the classes overlap** (success bins 76–200, failure bins 0–133; a slow success's start looks worse than a
  short failure's end). That is the intended contrast for the ablation, but worth knowing when reading its results.
- The `I_t` advantage step will need `Z` from the JSON (`meta["Z"]`), not recomputed; `load_returns` gives it.
- `env -u PYTHONPATH uv run ruff check .` currently reports one E501 in `src/recap_vf/model.py:347` (step 3's file,
  being edited concurrently) — not touched by me. All step-2 files pass `ruff check` and `ruff format --check`.
- No new dependencies; no shared files changed.

## Definition-of-done checklist
- [x] Three returns tables + metadata JSONs written (default `c2229`)
- [x] Numbers in report: Z per tag, ranges of `return_norm` per class, frame counts per class, bin occupancy
- [x] Figures saved and described
- [x] Tests pass, ruff clean (step-2 files; full suite 12 passed; one unrelated E501 in step 3's `model.py`)
- [x] `reports/step_02_returns.md` written
