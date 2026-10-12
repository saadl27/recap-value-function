"""Validation metrics for the distributional value function (step 4; reused by step 5).

All functions take per-frame NumPy arrays of one validation set (any frame order) and return plain floats, so the
same code scores the in-training evals and any offline re-evaluation. See docs/steps/04_training.md, "Metrics".
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from recap_vf import bins

AUC_FRACTIONS: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 0.75)
FAIL_DELTA = 0.02


def log_softmax_np(logits: np.ndarray) -> np.ndarray:
    """Numerically stable log-softmax over the last axis (float64)."""
    x = np.asarray(logits, dtype=np.float64)
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


def summarize_logits(logits: np.ndarray, target_bin: np.ndarray) -> dict[str, np.ndarray]:
    """Per-frame value E[V], std, cross-entropy vs `target_bin` and entropy from logits [N, NUM_BINS]."""
    logp = log_softmax_np(logits)
    p = np.exp(logp)
    centers = np.linspace(bins.V_MIN, bins.V_MAX, bins.NUM_BINS)
    value = p @ centers
    var = np.clip(p @ centers**2 - value**2, 0.0, None)
    target_bin = np.asarray(target_bin, dtype=np.int64)
    ce = -logp[np.arange(len(target_bin)), target_bin]
    entropy = -(p * logp).sum(axis=-1)
    return {"value": value, "value_std": np.sqrt(var), "ce": ce, "entropy": entropy}


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    if labels.all() or (~labels).all():
        return float("nan")  # undefined with one class
    return float(roc_auc_score(labels, scores))


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")  # undefined for constant input
    return float(spearmanr(a, b).statistic)


def episode_frame_at(
    episode_index: np.ndarray, frame_index: np.ndarray, episode_length: np.ndarray, fraction: float
) -> np.ndarray:
    """Row position, for every episode, of the available frame nearest to `fraction * T` (ties -> earlier frame)."""
    episode_index = np.asarray(episode_index)
    frame_index = np.asarray(frame_index, dtype=np.float64)
    episode_length = np.asarray(episode_length, dtype=np.float64)
    dist = np.abs(frame_index - fraction * episode_length)
    # sort by (episode, distance, frame) and take the first row per episode
    order = np.lexsort((frame_index, dist, episode_index))
    ep_sorted = episode_index[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = ep_sorted[1:] != ep_sorted[:-1]
    return order[first]


def fail_detect_times(
    value: np.ndarray,
    episode_index: np.ndarray,
    frame_index: np.ndarray,
    episode_length: np.ndarray,
    threshold: float,
) -> dict[int, float]:
    """{episode: first t/T with V < threshold, or inf if never} for the episodes given (caller filters failures)."""
    out: dict[int, float] = {}
    order = np.lexsort((frame_index, episode_index))
    ep, fi, v, length = (np.asarray(a)[order] for a in (episode_index, frame_index, value, episode_length))
    starts = np.flatnonzero(np.r_[True, ep[1:] != ep[:-1]])
    ends = np.r_[starts[1:], len(ep)]
    for s, e in zip(starts, ends, strict=True):
        hit = np.flatnonzero(v[s:e] < threshold)
        out[int(ep[s])] = float(fi[s + hit[0]] / length[s]) if len(hit) else math.inf
    return out


def compute_metrics(
    logits: np.ndarray,
    target_bin: np.ndarray,
    return_norm: np.ndarray,
    episode_index: np.ndarray,
    frame_index: np.ndarray,
    episode_length: np.ndarray,
    success: np.ndarray,
    *,
    fail_value: float,
    fail_delta: float = FAIL_DELTA,
    fractions: Sequence[float] = AUC_FRACTIONS,
) -> dict[str, float]:
    """All step-4 metrics for one validation set.

    `fail_value` = -C_fail / Z (the normalised return of the last frame of a failure); "confident of failure" means
    V < fail_value + fail_delta. `fail_detect_time` is the median over failure episodes of the first such t/T, with
    never-detected episodes counted as 1.0 (conservative); `fail_detect_frac` is the share ever detected and
    `false_alarm_frac` the share of *success* episodes that ever cross the threshold.
    """
    s = summarize_logits(logits, target_bin)
    value = s["value"]
    return_norm = np.asarray(return_norm, dtype=np.float64)
    success = np.asarray(success, dtype=bool)
    episode_index = np.asarray(episode_index, dtype=np.int64)
    frame_index = np.asarray(frame_index, dtype=np.int64)
    episode_length = np.asarray(episode_length, dtype=np.int64)

    abs_err = np.abs(value - return_norm)
    out: dict[str, float] = {
        "num_frames": float(len(value)),
        "num_episodes": float(len(np.unique(episode_index))),
        "ce": float(s["ce"].mean()),
        "mae": float(abs_err.mean()),
        "spearman": _spearman(value, return_norm),
        "entropy": float(s["entropy"].mean()),
        "value_std": float(s["value_std"].mean()),
        "acc_top1": float((np.argmax(logits, axis=-1) == np.asarray(target_bin)).mean()),
    }
    for name, mask in (("success", success), ("failure", ~success)):
        out[f"ce_{name}"] = float(s["ce"][mask].mean()) if mask.any() else float("nan")
        out[f"mae_{name}"] = float(abs_err[mask].mean()) if mask.any() else float("nan")

    for f in fractions:
        rows = episode_frame_at(episode_index, frame_index, episode_length, f)
        out[f"auc@{f:g}"] = _auc(success[rows], value[rows])

    threshold = fail_value + fail_delta
    times = fail_detect_times(value, episode_index, frame_index, episode_length, threshold)
    ep_success = dict(zip(episode_index.tolist(), success.tolist(), strict=True))
    fail_t = np.array([t for e, t in times.items() if not ep_success[e]])
    succ_t = np.array([t for e, t in times.items() if ep_success[e]])
    if len(fail_t):
        out["fail_detect_time"] = float(np.median(np.minimum(fail_t, 1.0)))
        out["fail_detect_frac"] = float(np.isfinite(fail_t).mean())
    else:
        out["fail_detect_time"] = out["fail_detect_frac"] = float("nan")
    out["false_alarm_frac"] = float(np.isfinite(succ_t).mean()) if len(succ_t) else float("nan")
    return out
