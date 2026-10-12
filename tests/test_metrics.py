"""Step 4 metrics on synthetic predictions with known answers."""

import math

import numpy as np
import pytest

from recap_vf import bins
from recap_vf.metrics import compute_metrics, episode_frame_at, fail_detect_times, summarize_logits

FAIL_VALUE = -0.5  # -C_fail / Z for c2229


def synthetic(stride: int = 1):
    """4 episodes (2 success, 2 failure) with exact-on-bin returns: g_t = -(T-1-t)/Z (- 0.5 for failures)."""
    z = 200.0
    rows = []
    for e, (length, success) in enumerate([(40, True), (60, True), (50, False), (80, False)]):
        for t in range(0, length, stride):
            g = -(length - 1 - t) / z + (0.0 if success else FAIL_VALUE)
            rows.append((e, t, length, success, g))
    ep, fi, length, succ, g = (np.array(c) for c in zip(*rows, strict=True))
    tb = bins.value_to_bin_np(g)
    return ep, fi, length, succ, g, tb


def onehot_logits(tb: np.ndarray, scale: float = 60.0) -> np.ndarray:
    logits = np.zeros((len(tb), bins.NUM_BINS))
    logits[np.arange(len(tb)), tb] = scale
    return logits


def test_perfect_predictions():
    ep, fi, length, succ, g, tb = synthetic(stride=5)
    g = np.linspace(-1, 0, bins.NUM_BINS)[tb]  # make returns exactly the bin centres -> MAE 0
    m = compute_metrics(onehot_logits(tb), tb, g, ep, fi, length, succ, fail_value=FAIL_VALUE)
    assert m["ce"] < 1e-6 and m["mae"] < 1e-9 and m["entropy"] < 1e-6
    assert m["spearman"] == pytest.approx(1.0)
    assert m["acc_top1"] == 1.0
    for f in (0, 0.1, 0.25, 0.5, 0.75):
        assert m[f"auc@{f:g}"] == 1.0
    assert m["fail_detect_time"] == 0.0 and m["fail_detect_frac"] == 1.0  # failures sit below -0.48 from t = 0
    assert m["false_alarm_frac"] == 0.0
    assert m["num_episodes"] == 4


def test_uniform_predictions():
    ep, fi, length, succ, g, tb = synthetic()
    m = compute_metrics(np.zeros((len(tb), bins.NUM_BINS)), tb, g, ep, fi, length, succ, fail_value=FAIL_VALUE)
    assert m["ce"] == pytest.approx(math.log(bins.NUM_BINS))
    assert m["entropy"] == pytest.approx(math.log(bins.NUM_BINS))
    assert m["mae"] == pytest.approx(np.abs(-0.5 - g).mean())
    assert m["auc@0.5"] == 0.5  # constant score
    # V = -0.5 < -0.5 + 0.02 everywhere: every episode "detected" at t = 0, every success is a false alarm
    assert m["fail_detect_time"] == 0.0 and m["fail_detect_frac"] == 1.0 and m["false_alarm_frac"] == 1.0
    assert math.isnan(m["spearman"])  # constant prediction


def test_never_confident_of_failure():
    ep, fi, length, succ, g, tb = synthetic()
    m = compute_metrics(onehot_logits(np.full_like(tb, 150)), tb, g, ep, fi, length, succ, fail_value=FAIL_VALUE)
    assert m["fail_detect_frac"] == 0.0 and m["fail_detect_time"] == 1.0  # never detected counts as 1.0
    assert m["false_alarm_frac"] == 0.0


def test_inverted_predictions_give_auc_zero():
    ep, fi, length, succ, g, tb = synthetic()
    flipped = bins.NUM_BINS - 1 - tb  # success frames now predicted low, failure frames high
    m = compute_metrics(onehot_logits(flipped), tb, g, ep, fi, length, succ, fail_value=FAIL_VALUE)
    assert m["auc@0"] == 0.0 and m["spearman"] < 0


def test_summarize_logits_matches_bins():
    import torch

    rng = np.random.default_rng(0)
    logits = rng.normal(size=(5, bins.NUM_BINS)) * 3
    s = summarize_logits(logits, np.array([0, 50, 100, 150, 200]))
    t = torch.tensor(logits, dtype=torch.float32)
    np.testing.assert_allclose(s["value"], bins.expected_value(t).numpy(), atol=1e-5)
    np.testing.assert_allclose(s["value_std"], bins.value_std(t).numpy(), atol=1e-4)
    ref_ce = torch.nn.functional.cross_entropy(t, torch.tensor([0, 50, 100, 150, 200]), reduction="none")
    np.testing.assert_allclose(s["ce"], ref_ce.numpy(), atol=1e-5)


def test_episode_frame_at_nearest_available_frame():
    ep = np.array([0, 0, 0, 1, 1, 1])
    fi = np.array([20, 0, 10, 0, 10, 20])  # unordered rows, stride 10
    length = np.array([25, 25, 25, 30, 30, 30])
    rows = episode_frame_at(ep, fi, length, 0.5)  # targets 12.5 and 15 (tie 10/20 -> earlier frame)
    assert sorted(fi[rows].tolist()) == [10, 10]
    rows = episode_frame_at(ep, fi, length, 0.75)  # 18.75 -> 20, 22.5 -> 20
    assert fi[rows].tolist() == [20, 20]
    assert fi[episode_frame_at(ep, fi, length, 0.0)].tolist() == [0, 0]


def test_fail_detect_times():
    ep = np.array([3, 3, 3, 3, 4, 4])
    fi = np.array([0, 10, 20, 30, 0, 10])
    length = np.array([40, 40, 40, 40, 20, 20])
    v = np.array([-0.3, -0.4, -0.6, -0.2, -0.3, -0.3])
    out = fail_detect_times(v, ep, fi, length, threshold=-0.48)
    assert out[3] == 0.5 and math.isinf(out[4])
