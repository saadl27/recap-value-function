"""Step 7: rendering helpers (frame composition, panel, keyframe/episode selection) on synthetic data."""

import json

import numpy as np
import pandas as pd
import pytest

from recap_vf.constants import CAMERAS, FPS, IMAGE_SIZE
from recap_vf.viz import (
    Mp4Writer,
    SeriesPanel,
    compose_camera_strip,
    compose_video_frame,
    load_episode_series,
    read_episode_list,
    select_eval_episodes,
    select_keyframes,
    true_runs,
    v_dip_depth,
)


# ---------------------------------------------------------------------------------------------------------------------
# Synthetic fixtures (independent of step 6's code: values = g + noise, A per docs/background.md §3)
# ---------------------------------------------------------------------------------------------------------------------
def fake_returns(
    lengths: dict[int, int], success: dict[int, bool], c_fail: float = 300.0
) -> tuple[pd.DataFrame, float]:
    rows, start = [], 0
    z = (max(lengths.values()) - 1) + c_fail
    for e, T in lengths.items():
        t = np.arange(T)
        g = (-(T - 1 - t) - (0 if success[e] else c_fail)) / z
        rows.append(pd.DataFrame({"index": start + t, "episode_index": e, "frame_index": t, "return_norm": g}))
        start += T
    return pd.concat(rows, ignore_index=True), z


def fake_values(returns: pd.DataFrame, rng: np.random.Generator, noise: float = 0.01) -> pd.DataFrame:
    v = returns["return_norm"].to_numpy() + rng.normal(0, noise, len(returns))
    return returns[["index", "episode_index", "frame_index"]].assign(
        value=np.clip(v, -1, 0), value_std=0.0, model_id="fake"
    )


def fake_advantages(values: pd.DataFrame, returns: pd.DataFrame, n: int = 50, p_pos: float = 0.4) -> pd.DataFrame:
    df = values.merge(returns[["index", "return_norm"]], on="index").sort_values("index")
    out = []
    for _, grp in df.groupby("episode_index", sort=True):
        g, v = grp["return_norm"].to_numpy(), grp["value"].to_numpy()
        T = len(g)
        inside = np.arange(T) + n < T
        nxt = np.minimum(np.arange(T) + n, T - 1)
        reward_sum = np.where(inside, g - g[nxt], g)
        value_next = np.where(inside, v[nxt], 0.0)
        out.append(grp.assign(value_next=value_next, reward_sum=reward_sum, advantage=reward_sum + value_next - v))
    adv = pd.concat(out, ignore_index=True)
    eps = float(np.quantile(adv["advantage"], 1 - p_pos))
    adv["indicator"] = adv["advantage"] >= eps
    return adv[["index", "episode_index", "frame_index", "value", "value_next", "reward_sum", "advantage", "indicator"]]


@pytest.fixture()
def run(tmp_path):
    lengths, success = {0: 120, 1: 90, 2: 75}, {0: True, 1: False, 2: True}
    ret, z = fake_returns(lengths, success)
    vals = fake_values(ret, np.random.default_rng(0))
    adv = fake_advantages(vals, ret)
    vals.to_parquet(tmp_path / "values.parquet")
    adv.to_parquet(tmp_path / "advantages_50_0.4.parquet")
    ret.to_parquet(tmp_path / "returns.parquet")
    return tmp_path, ret, adv


# ---------------------------------------------------------------------------------------------------------------------
def test_load_episode_series(run):
    rd, ret, adv = run
    s = load_episode_series(rd, 1, "advantages_50_0.4", returns_path=rd / "returns.parquet")
    assert list(s.columns) == ["index", "frame_index", "value", "return_norm", "advantage", "indicator"]
    assert len(s) == 90 and np.array_equal(s["frame_index"], np.arange(90))
    exp = adv[adv.episode_index == 1].sort_values("frame_index")["advantage"].to_numpy()
    np.testing.assert_allclose(s["advantage"], exp)
    s2 = load_episode_series(rd, 2, None, returns_path=rd / "returns.parquet")  # values only
    assert s2["advantage"].isna().all() and not s2["indicator"].any()


def test_camera_strip_and_frame_shapes(run):
    rd, *_ = run
    s = load_episode_series(rd, 0, "advantages_50_0.4", returns_path=rd / "returns.parquet")
    rng = np.random.default_rng(0)
    cams = {c: rng.integers(0, 255, (3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8) for c in CAMERAS}
    strip = compose_camera_strip(cams, labels=False)
    assert strip.shape == (IMAGE_SIZE, 3 * IMAGE_SIZE, 3) and strip.dtype == np.uint8
    # camera order follows constants.CAMERAS
    for i, c in enumerate(CAMERAS):
        np.testing.assert_array_equal(strip[:, i * IMAGE_SIZE : (i + 1) * IMAGE_SIZE], cams[c].transpose(1, 2, 0))
    panel = SeriesPanel(s, 3 * IMAGE_SIZE, 300, epsilon=0.0)
    assert panel.shape == (300, 3 * IMAGE_SIZE, 3)
    f = compose_video_frame(cams, panel.frame(10), "ep", "t")
    assert f.shape == (24 + IMAGE_SIZE + 300, 3 * IMAGE_SIZE, 3) and f.shape[0] % 2 == 0
    # the cursor moves: frames differ between two times, and only around the two cursor positions
    a, b = panel.frame(10), panel.frame(100)
    cols = np.flatnonzero((a != b).any(axis=(0, 2)))
    assert len(cols) and cols.max() - cols.min() > 100
    np.testing.assert_array_equal(panel.frame(None), panel.background)


def test_mp4_writer(tmp_path):
    import av

    p = tmp_path / "x.mp4"
    with Mp4Writer(p, 64, 48, fps=25, crf=28) as w:
        for i in range(10):
            w.write(np.full((48, 64, 3), i * 20, np.uint8))
    with av.open(str(p)) as c:
        frames = list(c.decode(video=0))
    assert len(frames) == 10 and (frames[0].width, frames[0].height) == (64, 48)
    with pytest.raises(ValueError):
        Mp4Writer(tmp_path / "odd.mp4", 63, 48)


def test_true_runs():
    assert true_runs(np.array([0, 1, 1, 0, 1], bool)) == [(1, 3), (4, 5)]
    assert true_runs(np.zeros(3, bool)) == []


def test_select_keyframes_spacing_and_order():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 0.01, 2000)
    a[[500, 520, 900, 1500]] = [-1.0, -0.9, -0.8, -0.7]  # 520 is too close to 500
    a[[100, 1700]] = [2.0, 1.0]
    a[50] = np.nan
    sep = 2 * FPS
    drops = select_keyframes(a, 3, sep, "drop")
    assert drops == [500, 900, 1500]
    rises = select_keyframes(a, 3, sep, "rise")
    assert rises[:2] == [100, 1700]
    for sel, sign in ((drops, 1), (rises, -1)):
        assert all(abs(i - j) >= sep for k, i in enumerate(sel) for j in sel[k + 1 :])
        vals = sign * a[sel]
        assert np.all(np.diff(vals) >= 0)  # most extreme first
    assert len(select_keyframes(a[:150], 5, sep, "drop")) == 2  # short episode: fewer than k fit
    neg = -np.abs(a[:400])  # no positive A at all: no "rise" picks unless the sign check is off
    assert select_keyframes(neg, 3, sep, "rise") == []
    assert len(select_keyframes(neg, 3, sep, "rise", require_sign=False)) == 3
    with pytest.raises(ValueError):
        select_keyframes(a, 1, sep, "sideways")


def test_v_dip_depth():
    v = np.array([-0.5, -0.4, -0.3, -0.6, -0.2, -0.1])
    d, peak, trough = v_dip_depth(v)
    assert d == pytest.approx(0.3) and (peak, trough) == (2, 3)
    assert v_dip_depth(np.linspace(-1, 0, 50))[0] == 0.0  # monotone rise: no dip
    spike = np.linspace(-0.6, 0, 101)
    spike[50] = -0.9
    assert v_dip_depth(spike, smooth=5)[0] < 0.05  # one-frame glitch removed by the median


def test_select_eval_episodes_deterministic():
    rng = np.random.default_rng(0)
    E = 60
    lengths = {e: int(rng.integers(100, 200)) for e in range(E)}
    success = {e: bool(e % 2) for e in range(E)}
    ret, _ = fake_returns(lengths, success)
    vals = fake_values(ret, rng, noise=0.0)
    deep = {11: 0.4, 23: 0.3, 35: 0.2}  # planted dips in successful episodes
    for e, depth in deep.items():
        m = (vals.episode_index == e) & vals.frame_index.between(40, 60)
        vals.loc[m, "value"] -= depth
    episodes = pd.DataFrame(
        {"episode_index": list(lengths), "length": list(lengths.values()), "success": [success[e] for e in lengths]}
    )
    pool = list(range(E))
    sel = select_eval_episodes(episodes, pool, vals, seed=3, n_fail=10, n_dip=3, n_success=5, smooth=1)
    again = select_eval_episodes(
        episodes, pool[::-1], vals.sample(frac=1, random_state=0), seed=3, n_fail=10, n_dip=3, n_success=5, smooth=1
    )
    assert sel == again  # independent of pool / row order
    groups = pd.DataFrame(sel)
    assert groups.group.value_counts().to_dict() == {
        "failure_random": 10,
        "success_deepest_dip": 3,
        "success_random": 5,
    }
    assert groups.episode_index.is_unique
    assert groups[groups.group == "success_deepest_dip"].episode_index.tolist() == [11, 23, 35]
    assert not groups[groups.group == "failure_random"].success.any()
    assert groups[groups.group != "failure_random"].success.all()
    other = select_eval_episodes(episodes, pool, vals, seed=4, n_dip=3, smooth=1)
    assert [x["episode_index"] for x in other] != [x["episode_index"] for x in sel]
    # tiny pool: groups are capped, never fail
    small = select_eval_episodes(episodes, [0, 1, 2], vals, seed=0)
    assert len(small) == 3


def test_read_episode_list(tmp_path):
    p = tmp_path / "sel.json"
    p.write_text(json.dumps({"episodes": [{"episode_index": 4}, {"episode_index": 9}]}))
    assert read_episode_list(p) == [4, 9]
    p.write_text("[1, 2]")
    assert read_episode_list(p) == [1, 2]
