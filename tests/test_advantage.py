"""CPU tests for N-step advantages, thresholds and labels (tiny synthetic episodes, no dataset)."""

import numpy as np
import pandas as pd
import pytest

from recap_vf.advantage import (
    ADVANTAGE_COLUMNS,
    SEGMENT_COLUMNS,
    TIE_ATOL,
    compute_advantages,
    diagnostics,
    label,
    label_agreement,
    per_episode_summary,
    run_lengths,
    segment_labels,
    smooth_values,
    threshold,
)
from recap_vf.returns import compute_returns_table

N = 5


def make_returns(lengths: list[int], successes: list[bool], c_fail: float = 50.0) -> tuple[pd.DataFrame, dict]:
    lengths_arr = np.asarray(lengths, dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(lengths_arr)[:-1]])
    episodes = pd.DataFrame(
        {
            "episode_index": np.arange(len(lengths), dtype=np.int64),
            "length": lengths_arr,
            "success": np.asarray(successes, dtype=bool),
            "dataset_from_index": starts,
            "dataset_to_index": starts + lengths_arr,
        }
    )
    return compute_returns_table(episodes, c_fail)


def values_from(returns: pd.DataFrame, value: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "index": returns["index"].to_numpy(),
            "episode_index": returns["episode_index"].to_numpy(),
            "frame_index": returns["frame_index"].to_numpy(),
            "value": np.asarray(value, dtype=np.float32),
            "value_std": np.zeros(len(returns), dtype=np.float32),
            "model_id": "test",
        }
    )


def adv_frame(a: np.ndarray) -> pd.DataFrame:
    n = len(a)
    return pd.DataFrame(
        {
            "index": np.arange(n),
            "episode_index": np.zeros(n, dtype=np.int64),
            "frame_index": np.arange(n),
            "advantage": a,
        }
    )


# --- advantage formula ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("use_z", [True, False])
def test_perfect_value_success_gives_zero_advantage(use_z):
    returns, meta = make_returns([3, 20, 40], [False, True, True])
    vals = values_from(returns, returns["return_norm"])
    adv = compute_advantages(vals, returns, N, z=meta["Z"] if use_z else None)
    succ = adv["episode_index"].isin([1, 2]).to_numpy()
    np.testing.assert_allclose(adv.loc[succ, "advantage"], 0.0, atol=1e-6)  # incl. the tail t + N >= T
    # Any episode (also failures) gets A = 0 under the perfect value: the MC return is exactly realised.
    np.testing.assert_allclose(adv["advantage"], 0.0, atol=1e-6)


def test_constant_value_failure():
    T, c = 23, 50.0
    returns, meta = make_returns([T, 30], [False, True], c_fail=c)
    z = meta["Z"]
    const = -0.3
    vals = values_from(returns, np.full(len(returns), const))
    adv = compute_advantages(vals, returns, N, z=z)
    ep0 = adv[adv["episode_index"] == 0].reset_index(drop=True)
    g = returns.loc[returns["episode_index"] == 0, "return_raw"].to_numpy(np.float64) / z
    head = ep0["frame_index"] + N < T
    np.testing.assert_allclose(ep0.loc[head, "advantage"], -N / z, rtol=0, atol=1e-7)
    np.testing.assert_array_equal(ep0.loc[head, "reward_sum"], np.full(int(head.sum()), -N / z))  # exact
    np.testing.assert_allclose(ep0.loc[~head, "advantage"], g[~head.to_numpy()] - const, atol=1e-7)
    np.testing.assert_array_equal(ep0.loc[~head, "value_next"], 0.0)
    # tail of a failure includes the terminal -C_fail reward
    assert ep0["reward_sum"].iloc[-1] == pytest.approx(-c / z)


def test_tail_boundary_uses_terminal_branch():
    T = 12
    returns, meta = make_returns([T], [True])
    rng = np.random.default_rng(0)
    v = rng.uniform(-1, 0, T)
    adv = compute_advantages(values_from(returns, v), returns, N, z=meta["Z"])
    g = returns["return_raw"].to_numpy(np.float64) / meta["Z"]
    v32 = v.astype(np.float32).astype(np.float64)
    t = T - N  # t + N == T -> terminal branch
    assert adv["value_next"].iloc[t] == 0.0
    assert adv["reward_sum"].iloc[t] == pytest.approx(g[t])
    assert adv["advantage"].iloc[t] == pytest.approx(g[t] - v32[t])
    t = T - N - 1  # t + N == T - 1 -> bootstrap branch
    assert adv["value_next"].iloc[t] == pytest.approx(v32[t + N])
    assert adv["advantage"].iloc[t] == pytest.approx(-N / meta["Z"] + v32[t + N] - v32[t])


def test_short_episode_all_tail_and_lookahead_one():
    returns, meta = make_returns([3, 8], [False, True])
    vals = values_from(returns, np.zeros(len(returns)))
    adv = compute_advantages(vals, returns, 10, z=meta["Z"])
    assert (adv["value_next"] == 0).all()
    np.testing.assert_allclose(adv["advantage"], returns["return_raw"] / meta["Z"], atol=1e-12)
    adv1 = compute_advantages(vals, returns, 1, z=meta["Z"])
    head = adv1["frame_index"] + 1 < returns["episode_length"]
    np.testing.assert_allclose(adv1.loc[head, "reward_sum"], -1 / meta["Z"], atol=1e-15)


def test_join_by_index_not_row_order_and_output_schema():
    returns, meta = make_returns([10, 15], [True, False])
    v = np.random.default_rng(1).uniform(-1, 0, len(returns))
    vals = values_from(returns, v)
    ref = compute_advantages(vals, returns, N, z=meta["Z"])
    shuffled = vals.sample(frac=1.0, random_state=0)
    got = compute_advantages(shuffled, returns.sample(frac=1.0, random_state=1), N, z=meta["Z"])
    pd.testing.assert_frame_equal(ref, got)
    assert list(ref.columns) == ADVANTAGE_COLUMNS[:-1]
    assert ref["index"].is_monotonic_increasing


def test_subset_of_episodes_ok_but_incomplete_episode_rejected():
    returns, meta = make_returns([10, 15, 7], [True, False, True])
    vals = values_from(returns, np.zeros(len(returns)))
    sub = vals[vals["episode_index"].isin([1])]
    adv = compute_advantages(sub, returns, N, z=meta["Z"])
    assert len(adv) == 15
    with pytest.raises(ValueError, match="incomplete"):
        compute_advantages(vals.drop(index=12), returns, N)
    bad = vals.copy()
    bad.loc[0, "frame_index"] = 3
    with pytest.raises(ValueError, match="disagree"):
        compute_advantages(bad, returns, N)
    with pytest.raises(ValueError, match="duplicated"):
        compute_advantages(pd.concat([vals, vals.iloc[:1]]), returns, N)
    nan = vals.copy()
    nan.loc[3, "value"] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        compute_advantages(nan, returns, N)


# --- threshold & labels -----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("n", [1, 7, 100, 1001])
@pytest.mark.parametrize("p", [0.0, 0.3, 0.4, 1.0])
def test_threshold_positive_fraction(n, p):
    a = np.random.default_rng(n).normal(size=n)
    eps = threshold(a, p)
    assert abs(int((a >= eps).sum()) - p * n) <= 1  # +- one frame
    _, info = label(adv_frame(a), p)
    assert abs(info["num_positive"] - p * n) <= 1


def test_threshold_ties_handled_with_ge():
    a = np.array([0.0] * 30 + [1.0] * 20 + [-1.0] * 50)  # 100 frames, 40 % -> 40 frames wanted
    eps = threshold(a, 0.4)
    assert eps == 0.0
    labelled, info = label(adv_frame(a), 0.4)
    ind = labelled["indicator"].to_numpy()
    assert ind[a == 0.0].all() and ind[a == 1.0].all() and not ind[a == -1.0].any()  # all ties are positive
    assert info["num_positive"] == 50 and info["ties_at_epsilon"] == 30
    assert info["degenerate"] and info["warnings"]


def test_degenerate_constant_advantage():
    a = np.zeros(50) + np.random.default_rng(0).normal(scale=TIE_ATOL / 10, size=50)  # float32-level noise
    labelled, info = label(adv_frame(a), 0.4)
    assert labelled["indicator"].all()
    assert info["degenerate"] and "zero spread" in info["warnings"][0]


def test_non_degenerate_info():
    a = np.random.default_rng(0).normal(size=1000)
    labelled, info = label(adv_frame(a), 0.4)
    assert not info["degenerate"] and not info["warnings"]
    assert info["positive_fraction"] == pytest.approx(0.4, abs=1e-3)
    assert labelled["indicator"].dtype == bool
    assert (labelled["indicator"] == (a >= info["epsilon"])).all()


def test_forced_positive():
    a = np.linspace(-1, 1, 100)
    forced = {0, 1, 2, 10_000}  # the most negative frames, plus one unknown index
    labelled, info = label(adv_frame(a), 0.4, forced_positive=forced)
    ind = labelled["indicator"].to_numpy()
    assert ind[[0, 1, 2]].all()
    assert info["num_forced_positive"] == 3
    assert abs(int(ind[3:].sum()) - 0.4 * 97) <= 1  # threshold computed over the non-forced frames
    assert any("not in the table" in w for w in info["warnings"])


def test_threshold_rejects_bad_input():
    with pytest.raises(ValueError):
        threshold(np.array([]), 0.4)
    with pytest.raises(ValueError):
        threshold(np.array([0.0, np.nan]), 0.4)
    with pytest.raises(ValueError):
        threshold(np.array([0.0]), 1.5)


# --- optional transforms & diagnostics --------------------------------------------------------------------------------
def test_smooth_values_within_episodes():
    returns, _ = make_returns([6, 6], [True, False])
    v = np.array([0, 0, -1, 0, 0, 0, -0.5, -0.5, -0.5, -0.5, -0.5, -0.9], dtype=np.float64)
    out = smooth_values(values_from(returns, v), 3)
    np.testing.assert_allclose(out["value"].iloc[:6], 0.0)  # spike removed
    np.testing.assert_allclose(out["value"].iloc[6:11], -0.5)  # episode boundary not crossed
    assert out["value"].iloc[11] == pytest.approx(-0.9)  # edge repeats the boundary frame: median(-.5, -.9, -.9)
    pd.testing.assert_frame_equal(smooth_values(values_from(returns, v), 1), values_from(returns, v))
    with pytest.raises(ValueError):
        smooth_values(values_from(returns, v), 4)


def test_run_lengths_break_at_episode_boundary():
    ind = np.array([1, 1, 0, 0, 0, 0, 1, 1, 1], dtype=bool)
    ep = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])
    lengths, vals = run_lengths(ind, ep)
    np.testing.assert_array_equal(lengths, [2, 2, 2, 3])
    np.testing.assert_array_equal(vals, [True, False, False, True])


def test_segments_and_agreement_and_diagnostics():
    returns, meta = make_returns([12, 7], [True, False])
    v = np.random.default_rng(3).uniform(-1, 0, len(returns))
    adv = compute_advantages(values_from(returns, v), returns, N, z=meta["Z"])
    labelled, _ = label(adv, 0.4)
    seg = segment_labels(labelled, 5)
    assert list(seg.columns) == SEGMENT_COLUMNS
    assert seg["num_frames"].tolist() == [5, 5, 2, 5, 2]
    assert seg["frame_start"].tolist() == [0, 5, 10, 0, 5]
    first = labelled.set_index("index").loc[seg["index"], "indicator"].to_numpy()
    np.testing.assert_array_equal(seg["indicator"].to_numpy(), first)
    assert seg["advantage_mean"].iloc[0] == pytest.approx(labelled["advantage"].iloc[:5].mean())

    assert label_agreement(labelled, labelled) == 1.0
    flipped = labelled.assign(indicator=~labelled["indicator"])
    assert label_agreement(labelled, flipped) == 0.0
    with pytest.raises(ValueError):
        label_agreement(labelled, labelled.iloc[:5])

    success = returns.groupby("episode_index")["success"].first()
    d = diagnostics(labelled, success)
    assert d["num_frames"] == 19 and d["num_episodes"] == 2
    assert d["positive_fraction"] == pytest.approx(labelled["indicator"].mean())
    per_ep = per_episode_summary(labelled, success)
    assert per_ep["success"].tolist() == [True, False]
    assert per_ep["length"].tolist() == [12, 7]
