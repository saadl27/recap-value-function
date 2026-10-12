"""N-step advantages and binary improvement indicators (RECAP, step 6).

Per episode of length T, with g_t = G_t / Z (normalised return-to-go) and V_t the predicted value, lookahead N:

    t + N <  T:  reward_sum_t = g_t - g_{t+N}  (= -N/Z exactly)      value_next_t = V_{t+N}
    t + N >= T:  reward_sum_t = g_t            (incl. terminal reward) value_next_t = 0
    A_t = reward_sum_t + value_next_t - V_t

which is the paper's  sum_{t'=t}^{t+N-1} r_t' + V(o_{t+N}) - V(o_t)  in normalised units (gamma = 1).
With human interventions the returns table carries `stretch_frame` / `stretch_length` (see `recap_vf.returns`): the
window then stops at the next takeover instead of the episode end (T above becomes the stretch length), consistent
with returns that are cut there. Human frames are labelled positive with `forced_positive` (`intervention_indices`).
One dataset-wide threshold eps = quantile(A, 1 - p_pos) turns A into the indicator I_t = 1[A_t >= eps].
Everything is vectorised over the whole (index-sorted, episode-contiguous) table: no Python loop over frames.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

VALUE_COLUMNS = ["index", "episode_index", "frame_index", "value", "value_std", "model_id"]
ADVANTAGE_COLUMNS = [
    "index",
    "episode_index",
    "frame_index",
    "value",
    "value_next",
    "reward_sum",
    "advantage",
    "indicator",
]
SEGMENT_COLUMNS = [
    "index",
    "episode_index",
    "segment_index",
    "frame_start",
    "frame_end",
    "num_frames",
    "indicator",
    "advantage_mean",
    "frame_positive_fraction",
]

# Advantages closer than this to the threshold count as ties (and are labelled positive, `>=`). In normalised units
# one frame of progress is 1/Z ~ 2.2e-4, so 1e-6 is ~0.005 frames: far below anything meaningful, but above the
# float32 resolution of V (ulp(0.5) ~ 6e-8), so float32 round-off of V - V' can never flip a label at the threshold.
TIE_ATOL = 1e-6


# ---------------------------------------------------------------------------------------------------------------------
# Advantage
# ---------------------------------------------------------------------------------------------------------------------
def compute_advantages(
    values: pd.DataFrame, returns: pd.DataFrame, lookahead: int, *, z: float | None = None
) -> pd.DataFrame:
    """N-step advantage for every frame of the episodes present in `values` (complete episodes only).

    `values`: at least `index, episode_index, frame_index, value` (contract schema of values.parquet).
    `returns`: the step-2 returns table (`index, episode_index, frame_index, episode_length, return_norm`, and
    `return_raw` when `z` is given). With `z` (the returns JSON's Z, never recomputed) g_t = return_raw / Z in float64,
    which makes reward_sum = -N/Z exactly; without it the float32 `return_norm` column is used.
    If `returns` has `stretch_frame` and `stretch_length` (returns cut at human interventions), the lookahead never
    crosses a cut: a window that reaches it sums the rewards up to the cut and adds no value after it.
    Joined on the global `index`. Returns the contract columns minus `indicator`, sorted by `index`.
    """
    lookahead = int(lookahead)
    if lookahead < 1:
        raise ValueError(f"lookahead must be >= 1, got {lookahead}")
    for name, df, cols in (
        ("values", values, ["index", "episode_index", "frame_index", "value"]),
        ("returns", returns, ["index", "episode_index", "frame_index", "episode_length", "return_norm"]),
    ):
        if missing := [c for c in cols if c not in df.columns]:
            raise KeyError(f"{name} is missing columns {missing}")
    if z is not None and "return_raw" not in returns.columns:
        raise KeyError("returns is missing 'return_raw' (needed when z is given)")
    if values["index"].duplicated().any():
        raise ValueError(f"values has {int(values['index'].duplicated().sum())} duplicated index values")

    r_cols = ["index", "episode_index", "frame_index", "episode_length", "return_norm"]
    if z is not None:
        r_cols.append("return_raw")
    cut = {"stretch_frame", "stretch_length"} <= set(returns.columns)
    if cut:
        r_cols += ["stretch_frame", "stretch_length"]
    merged = values[["index", "episode_index", "frame_index", "value"]].merge(
        returns[r_cols].rename(columns={"episode_index": "_ep", "frame_index": "_fi"}),
        on="index",
        how="left",
        validate="one_to_one",
    )
    if merged["_ep"].isna().any():
        raise KeyError(f"{int(merged['_ep'].isna().sum())} value rows have an index absent from the returns table")
    merged = merged.sort_values("index", kind="stable").reset_index(drop=True)

    ep = merged["episode_index"].to_numpy(np.int64)
    fi = merged["frame_index"].to_numpy(np.int64)
    if not (np.array_equal(ep, merged["_ep"].to_numpy(np.int64)) and np.array_equal(fi, merged["_fi"].to_numpy())):
        raise ValueError("episode_index / frame_index of values disagree with the returns table for the same index")
    length = merged["episode_length"].to_numpy(np.int64)
    # Complete, contiguous episodes: sorted by index, each episode's rows must be frames 0..T-1 in order. Then row
    # i + N is frame t + N of the same episode whenever t + N < T.
    position = merged.groupby("episode_index", sort=False).cumcount().to_numpy(np.int64)
    counts = merged.groupby("episode_index", sort=False)["index"].transform("size").to_numpy(np.int64)
    if not (np.array_equal(position, fi) and np.array_equal(counts, length)):
        bad = np.unique(ep[(position != fi) | (counts != length)])
        raise ValueError(f"{len(bad)} episodes are incomplete or not contiguous in values (e.g. {bad[:5].tolist()})")

    v = merged["value"].to_numpy(np.float64)
    if not np.isfinite(v).all():
        raise ValueError(f"{int((~np.isfinite(v)).sum())} non-finite values")
    if z is not None:
        z = float(z)
        g_raw = merged["return_raw"].to_numpy(np.float64)
        g = g_raw / z
    else:
        g = merged["return_norm"].to_numpy(np.float64)

    n = len(merged)
    if cut:  # stretches are contiguous runs of frames, so row i + N is frame t + N of the same stretch here
        has_next = merged["stretch_frame"].to_numpy(np.int64) + lookahead < merged["stretch_length"].to_numpy(np.int64)
    else:
        has_next = fi + lookahead < length
    j = np.minimum(np.arange(n) + lookahead, max(n - 1, 0))
    value_next = np.where(has_next, v[j], 0.0)
    if z is not None:
        reward_sum = np.where(has_next, (g_raw - g_raw[j]) / z, g)  # integer difference / Z: exactly -N/Z
    else:
        reward_sum = np.where(has_next, g - g[j], g)
    advantage = reward_sum + value_next - v

    return pd.DataFrame(
        {
            "index": merged["index"].to_numpy(np.int64),
            "episode_index": ep,
            "frame_index": fi,
            "value": merged["value"].to_numpy(np.float32),
            "value_next": value_next.astype(np.float32),
            "reward_sum": reward_sum,  # float64: keeps `advantage >= epsilon` reproducible from the file
            "advantage": advantage,
        }
    )


def intervention_indices(returns: pd.DataFrame) -> set[int]:
    """Global indices of human-controlled frames (RECAP labels human corrections positive); empty if no flag."""
    if "intervention" not in returns.columns:
        return set()
    return set(returns.loc[returns["intervention"].astype(bool), "index"].astype(np.int64).tolist())


# ---------------------------------------------------------------------------------------------------------------------
# Threshold & labels
# ---------------------------------------------------------------------------------------------------------------------
def threshold(advantages: np.ndarray, p_pos: float) -> float:
    """eps such that a fraction `p_pos` of `advantages` satisfies A >= eps (exactly, up to ties).

    Order-statistic form of quantile(A, 1 - p_pos): eps is the k-th largest A with k = round(p_pos * n), so
    `(A >= eps).sum() == k` when there are no ties (more with ties). p_pos = 0 gives +inf (nothing positive).
    """
    a = np.asarray(advantages, dtype=np.float64).ravel()
    if not 0.0 <= p_pos <= 1.0:
        raise ValueError(f"p_pos must be in [0, 1], got {p_pos}")
    if a.size == 0:
        raise ValueError("cannot threshold an empty advantage array")
    if not np.isfinite(a).all():
        raise ValueError("advantages contain non-finite values")
    n = a.size
    k = int(round(p_pos * n))
    if k == 0:
        return float("inf")
    if k >= n:
        return float(a.min())
    return float(np.partition(a, n - k)[n - k])


def label(
    adv_df: pd.DataFrame,
    p_pos: float,
    forced_positive: set[int] | None = None,
    *,
    tie_atol: float = TIE_ATOL,
) -> tuple[pd.DataFrame, dict]:
    """Add `indicator = A >= eps` (one dataset-wide eps) and return (labelled df, threshold info).

    `forced_positive`: global frame indices labelled True regardless of A (human corrections in RECAP); eps is
    computed over the other frames. A within `tie_atol` of eps counts as a tie (positive). The info dict flags a
    **degenerate** threshold (A has ~no spread, or ties push the positive fraction well above `p_pos`) instead of
    failing, so e.g. a perfect value function (A = 0 everywhere) yields all-positive labels plus a warning.
    """
    a = adv_df["advantage"].to_numpy(np.float64)
    idx = adv_df["index"].to_numpy(np.int64)
    forced = np.zeros(len(a), dtype=bool)
    n_forced_missing = 0
    if forced_positive:
        wanted = np.fromiter((int(i) for i in forced_positive), dtype=np.int64)
        forced = np.isin(idx, wanted)
        n_forced_missing = int(len(np.unique(wanted)) - forced.sum())
    eligible = ~forced
    n_elig = int(eligible.sum())

    warnings: list[str] = []
    if n_elig:
        a_el = a[eligible]
        eps = threshold(a_el, p_pos)
        indicator = a >= eps - tie_atol if np.isfinite(eps) else np.zeros(len(a), dtype=bool)
        ties = int((np.abs(a_el - eps) <= tie_atol).sum()) if np.isfinite(eps) else 0
        spread = float(a_el.max() - a_el.min())
        q = np.quantile(a_el, [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0])
    else:
        eps, ties, spread, q = float("nan"), 0, 0.0, np.full(9, np.nan)
        indicator = np.zeros(len(a), dtype=bool)
        warnings.append("no non-forced frames: threshold undefined")
    indicator = indicator | forced

    pos_frac_elig = float(indicator[eligible].mean()) if n_elig else float("nan")
    degenerate = False
    if n_elig and spread <= tie_atol:
        degenerate = True
        warnings.append(
            f"advantage has (near-)zero spread ({spread:.3g} <= tie_atol {tie_atol:g}): every frame is a tie, the "
            f"threshold cannot separate frames -> {pos_frac_elig:.1%} positive instead of {p_pos:.0%}"
        )
    elif n_elig and abs(pos_frac_elig - p_pos) > max(2.0 / n_elig, 0.01):
        degenerate = True
        warnings.append(
            f"{ties} frames tie at eps = {eps:.6g}: {pos_frac_elig:.1%} positive instead of the requested {p_pos:.0%}"
        )
    if n_forced_missing:
        warnings.append(f"{n_forced_missing} forced_positive indices are not in the table (ignored)")
    for w in warnings:
        logger.warning("label(p_pos=%g): %s", p_pos, w)

    out = adv_df.copy()
    out["indicator"] = indicator
    info = {
        "p_pos": float(p_pos),
        "epsilon": float(eps),
        "tie_atol": float(tie_atol),
        "num_frames": int(len(a)),
        "num_positive": int(indicator.sum()),
        "positive_fraction": float(indicator.mean()) if len(a) else float("nan"),
        "positive_fraction_unforced": pos_frac_elig,
        "ties_at_epsilon": ties,
        "num_forced_positive": int(forced.sum()),
        "advantage_spread": spread,
        "advantage_quantiles": dict(
            zip(["min", "q01", "q05", "q25", "median", "q75", "q95", "q99", "max"], map(float, q), strict=True)
        ),
        "degenerate": degenerate,
        "warnings": warnings,
    }
    return out, info


# ---------------------------------------------------------------------------------------------------------------------
# Optional transforms
# ---------------------------------------------------------------------------------------------------------------------
def smooth_values(values: pd.DataFrame, k: int) -> pd.DataFrame:
    """Centred median of `value` over `k` frames (odd), within each episode (edges repeat the boundary frame).

    A flagged, non-default option of step 6 to suppress frame-to-frame jitter before computing advantages.
    """
    from scipy.ndimage import median_filter

    k = int(k)
    if k < 1 or k % 2 == 0:
        raise ValueError(f"smoothing window must be a positive odd integer, got {k}")
    out = values.sort_values("index", kind="stable").reset_index(drop=True)
    if k == 1:
        return out
    v = out["value"].to_numpy(np.float64)
    ep = out["episode_index"].to_numpy(np.int64)
    bounds = np.flatnonzero(np.diff(ep)) + 1
    smoothed = np.concatenate([median_filter(chunk, size=k, mode="nearest") for chunk in np.split(v, bounds)])
    out["value"] = smoothed.astype(np.float32)
    return out


def segment_labels(labelled: pd.DataFrame, segment_len: int) -> pd.DataFrame:
    """Non-overlapping `segment_len`-frame chunks per episode (last one may be shorter).

    Chunk label = `indicator` at its first frame; also the mean advantage and the fraction of positive frames.
    """
    segment_len = int(segment_len)
    if segment_len < 1:
        raise ValueError("segment_len must be >= 1")
    df = labelled.sort_values("index", kind="stable")
    seg = (df["frame_index"].to_numpy(np.int64) // segment_len).astype(np.int64)
    grouped = df.assign(segment_index=seg).groupby(["episode_index", "segment_index"], sort=True)
    out = grouped.agg(
        index=("index", "first"),
        frame_start=("frame_index", "first"),
        num_frames=("frame_index", "size"),
        indicator=("indicator", "first"),
        advantage_mean=("advantage", "mean"),
        frame_positive_fraction=("indicator", "mean"),
    ).reset_index()
    out["frame_end"] = out["frame_start"] + out["num_frames"]
    out["indicator"] = out["indicator"].astype(bool)
    return out[SEGMENT_COLUMNS]


# ---------------------------------------------------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------------------------------------------------
def run_lengths(indicator: np.ndarray, episode_index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lengths (frames) and values of runs of identical consecutive `indicator`, broken at episode boundaries.

    Inputs must be sorted by global index.
    """
    ind = np.asarray(indicator, dtype=bool)
    ep = np.asarray(episode_index)
    if ind.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=bool)
    change = np.r_[True, (ind[1:] != ind[:-1]) | (ep[1:] != ep[:-1])]
    starts = np.flatnonzero(change)
    return np.diff(np.r_[starts, ind.size]).astype(np.int64), ind[starts]


def label_agreement(a: pd.DataFrame, b: pd.DataFrame) -> float:
    """Fraction of frames (joined on `index`) with the same indicator in both labellings."""
    m = a[["index", "indicator"]].merge(b[["index", "indicator"]], on="index", validate="one_to_one")
    if len(m) != len(a) or len(m) != len(b):
        raise ValueError(f"labellings cover different frames ({len(a)} vs {len(b)}, {len(m)} shared)")
    return float((m["indicator_x"] == m["indicator_y"]).mean())


def per_episode_summary(labelled: pd.DataFrame, episode_success: Mapping[int, bool] | pd.Series) -> pd.DataFrame:
    """One row per episode: success, length, positive_fraction, advantage_mean, mean_run_length."""
    df = labelled.sort_values("index", kind="stable")
    lengths, _ = run_lengths(df["indicator"].to_numpy(), df["episode_index"].to_numpy())
    ep = df["episode_index"].to_numpy(np.int64)
    run_ep = ep[np.r_[0, np.cumsum(lengths)[:-1]]] if len(lengths) else ep[:0]
    runs = pd.Series(lengths).groupby(run_ep).mean()
    out = df.groupby("episode_index").agg(
        length=("frame_index", "size"),
        positive_fraction=("indicator", "mean"),
        advantage_mean=("advantage", "mean"),
        advantage_min=("advantage", "min"),
    )
    out["mean_run_length"] = runs.reindex(out.index).to_numpy()
    success = pd.Series(episode_success) if not isinstance(episode_success, pd.Series) else episode_success
    out.insert(0, "success", success.reindex(out.index).astype(bool).to_numpy())
    return out.reset_index()


def diagnostics(labelled: pd.DataFrame, episode_success: Mapping[int, bool] | pd.Series) -> dict:
    """Label-quality numbers for the report (positive fractions by outcome, value jitter, label flicker)."""
    df = labelled.sort_values("index", kind="stable")
    ep = df["episode_index"].to_numpy(np.int64)
    ind = df["indicator"].to_numpy(bool)
    a = df["advantage"].to_numpy(np.float64)
    v = df["value"].to_numpy(np.float64)
    success = pd.Series(episode_success) if not isinstance(episode_success, pd.Series) else episode_success
    succ_frame = success.reindex(ep).astype(bool).to_numpy()

    same_ep = ep[1:] == ep[:-1]
    dv = np.diff(v)[same_ep]
    jitter = float(dv.std()) if dv.size else float("nan")
    a_std = float(a.std()) if a.size else float("nan")
    jitter_r = float(1.4826 * np.median(np.abs(dv - np.median(dv)))) if dv.size else float("nan")
    a_r = float(np.subtract(*np.quantile(a, [0.75, 0.25])) / 1.349) if a.size else float("nan")
    lengths, run_vals = run_lengths(ind, ep)
    per_ep = per_episode_summary(df, success)

    def frac(mask: np.ndarray) -> float:
        return float(ind[mask].mean()) if mask.any() else float("nan")

    out: dict = {
        "num_frames": int(len(df)),
        "num_episodes": int(len(per_ep)),
        "positive_fraction": float(ind.mean()) if len(ind) else float("nan"),
        "positive_fraction_success_frames": frac(succ_frame),
        "positive_fraction_failure_frames": frac(~succ_frame),
        "value_jitter_std": jitter,  # std of V_{t+1} - V_t within episodes
        "advantage_std": a_std,
        "jitter_to_advantage_std": jitter / a_std if a_std > 0 else float("inf"),
        # Robust twins (heavy-tailed A, e.g. a few large failure drops, inflate std(A) and hide noisy labels):
        # 1.4826 * MAD(dV) and IQR(A) / 1.349 both estimate a Gaussian sigma.
        "value_jitter_robust": jitter_r,
        "advantage_robust_std": a_r,
        "jitter_to_advantage_robust": jitter_r / a_r if a_r > 0 else float("inf"),
        "mean_run_length": float(lengths.mean()) if lengths.size else float("nan"),
        "median_run_length": float(np.median(lengths)) if lengths.size else float("nan"),
        "mean_run_length_positive": float(lengths[run_vals].mean()) if run_vals.any() else float("nan"),
        "mean_run_length_negative": float(lengths[~run_vals].mean()) if (~run_vals).any() else float("nan"),
        "fraction_runs_le5": float((lengths <= 5).mean()) if lengths.size else float("nan"),
        "fraction_frames_in_runs_le5": float(lengths[lengths <= 5].sum() / lengths.sum()) if lengths.size else 0.0,
        "advantage_mean_success": float(a[succ_frame].mean()) if succ_frame.any() else float("nan"),
        "advantage_mean_failure": float(a[~succ_frame].mean()) if (~succ_frame).any() else float("nan"),
    }
    for name, flag in (("success", True), ("failure", False)):
        pf = per_ep.loc[per_ep["success"] == flag, "positive_fraction"]
        out[f"episode_positive_fraction_{name}"] = {
            "episodes": int(len(pf)),
            "mean": float(pf.mean()) if len(pf) else float("nan"),
            "median": float(pf.median()) if len(pf) else float("nan"),
            "q10": float(pf.quantile(0.1)) if len(pf) else float("nan"),
            "q90": float(pf.quantile(0.9)) if len(pf) else float("nan"),
            "num_all_positive": int((pf == 1.0).sum()),
            "num_all_negative": int((pf == 0.0).sum()),
        }
    return out
