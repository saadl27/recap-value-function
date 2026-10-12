import numpy as np
import pandas as pd
import pytest

from recap_vf import bins
from recap_vf.returns import (
    INTERVENTION_COLUMNS,
    RETURNS_COLUMNS,
    compute_episode_returns,
    compute_episode_returns_interventions,
    compute_returns_table,
    default_intervention_penalty,
    intervention_onsets,
    load_episodes,
    load_returns,
    save_returns,
    verify_index,
)


def make_episodes(lengths: list[int], successes: list[bool]) -> pd.DataFrame:
    lengths_arr = np.asarray(lengths, dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(lengths_arr)[:-1]])
    return pd.DataFrame(
        {
            "episode_index": np.arange(len(lengths), dtype=np.int64),
            "length": lengths_arr,
            "success": np.asarray(successes, dtype=bool),
            "dataset_from_index": starts,
            "dataset_to_index": starts + lengths_arr,
        }
    )


# Mimics the real dataset's extremes: longest episode is a failure of 2229 frames, longest success 2077.
REAL_LIKE = make_episodes([624, 2077, 230, 2229, 382, 157, 900], [False, True, False, False, True, False, True])


def test_success_episode_returns():
    T = 7
    out = compute_episode_returns(T, True, c_fail=100)
    G = out["return_raw"]
    assert G[-1] == 0
    assert G[0] == -(T - 1)
    np.testing.assert_array_equal(np.diff(G), np.ones(T - 1))
    np.testing.assert_array_equal(out["reward"], [-1] * (T - 1) + [0])


def test_failure_episode_returns():
    T, c = 5, 42.0
    out = compute_episode_returns(T, False, c_fail=c)
    G = out["return_raw"]
    assert G[-1] == -c
    assert G[0] == -(T - 1) - c
    np.testing.assert_array_equal(np.diff(G), np.ones(T - 1))
    assert out["reward"][-1] == -c


def test_single_frame_episode():
    assert compute_episode_returns(1, True, 10)["return_raw"].tolist() == [0.0]
    assert compute_episode_returns(1, False, 10)["return_raw"].tolist() == [-10.0]
    with pytest.raises(ValueError):
        compute_episode_returns(0, True, 10)


def test_table_normalisation_and_metadata():
    table, meta = compute_returns_table(REAL_LIKE, c_fail=2229)
    assert meta["Z"] == 2228 + 2229 == 4457
    assert meta["t_max"] == 2229 and meta["t_max_success"] == 2077 and meta["t_max_failure"] == 2229
    assert meta["num_bins"] == bins.NUM_BINS and meta["v_min"] == -1.0 and meta["v_max"] == 0.0
    assert meta["tag"] == "c2229"
    g = table["return_norm"].to_numpy()
    assert g.min() == -1.0 and g.max() == 0.0
    assert ((g >= -1) & (g <= 0)).all()
    # The minimum sits at frame 0 of the longest failure.
    row = table.loc[table["return_norm"].idxmin()]
    assert row["episode_index"] == 3 and row["frame_index"] == 0
    np.testing.assert_allclose(table["return_raw"] / meta["Z"], g, rtol=0, atol=1e-6)


def test_bins_c2229_classes_do_not_overlap():
    table, _ = compute_returns_table(REAL_LIKE, c_fail=2229)
    b = table["bin_index"].to_numpy()
    assert b.min() >= 0 and b.max() <= bins.NUM_BINS - 1
    succ, fail = b[table["success"]], b[~table["success"]]
    assert succ.min() == 107 and succ.max() == 200
    assert fail.min() == 0 and fail.max() == 100
    np.testing.assert_array_equal(b, bins.value_to_bin_np(table["return_norm"].to_numpy(np.float64)))


def test_index_unique_contiguous_and_schema():
    table, _ = compute_returns_table(REAL_LIKE, c_fail=1115)
    assert list(table.columns) == RETURNS_COLUMNS
    assert len(table) == REAL_LIKE["length"].sum()
    idx = table["index"].to_numpy()
    np.testing.assert_array_equal(idx, np.arange(len(table)))
    assert table["index"].is_unique
    # Global index = dataset_from_index + frame_index.
    starts = REAL_LIKE.set_index("episode_index")["dataset_from_index"]
    np.testing.assert_array_equal(idx, starts.loc[table["episode_index"]].to_numpy() + table["frame_index"].to_numpy())
    assert (
        table["episode_length"] == REAL_LIKE.set_index("episode_index").loc[table["episode_index"], "length"].to_numpy()
    ).all()
    assert table["reward"].sum() == -(REAL_LIKE["length"] - 1).sum() - 1115 * (~REAL_LIKE["success"]).sum()


def test_z_when_longest_is_success():
    eps = make_episodes([10, 50], [False, True])
    table, meta = compute_returns_table(eps, c_fail=5)
    # min(G) = max(-(10-1)-5, -(50-1)) -> -49 from the success episode.
    assert meta["Z"] == 49
    assert table["return_norm"].min() == -1.0


def test_verify_index_and_roundtrip(tmp_path, monkeypatch):
    table, meta = compute_returns_table(REAL_LIKE, c_fail=4458)
    frames = table[["index", "episode_index", "frame_index"]].sample(frac=1, random_state=0)
    verify_index(table, frames)
    bad = frames.copy()
    bad.loc[bad.index[0], "frame_index"] += 1
    with pytest.raises(ValueError):
        verify_index(table, bad)

    from recap_vf import paths

    monkeypatch.setattr(paths, "CACHE_DIR", tmp_path)
    save_returns(table, meta)
    loaded, loaded_meta = load_returns("c4458")
    assert loaded_meta == meta
    pd.testing.assert_frame_equal(loaded, table)
    assert loaded["return_raw"].dtype == np.float32 and loaded["bin_index"].dtype == np.int64


# --- human interventions: penalty before each takeover, return cut at the next one ------------------------------------
C = 100.0


def human(length: int, *runs: tuple[int, int]) -> np.ndarray:
    h = np.zeros(length, dtype=bool)
    for a, b in runs:
        h[a:b] = True
    return h


def test_two_interventions_penalise_once_each():
    # robot 0-4, human 5-7, robot 8-11, human 12-14, robot 15-19, success
    h = human(20, (5, 8), (12, 15))
    out = compute_episode_returns_interventions(20, True, c_fail=999, intervention=h, c_interv=C)
    r, G = out["reward"], out["return_raw"]
    np.testing.assert_array_equal(np.flatnonzero(r == -C), [4, 11])  # the robot's last frame before each takeover
    assert r[-1] == 0 and (r[np.r_[0:4, 5:11, 12:19]] == -1).all()
    assert G[0] == -4 - C  # only the first penalty, not -2C
    assert G[4] == -C and G[11] == -C and G[-1] == 0
    assert G[5] == -6 - C  # a human frame belongs to the stretch that ends at the next takeover
    assert G[12] == -7  # last stretch: runs to the episode end
    np.testing.assert_array_equal(out["stretch_index"], [0] * 5 + [1] * 7 + [2] * 8)
    np.testing.assert_array_equal(out["stretch_length"], [5] * 5 + [7] * 7 + [8] * 8)
    np.testing.assert_array_equal(out["stretch_frame"], np.r_[0:5, 0:7, 0:8])
    # within a stretch the return rises by exactly one per frame
    for s in range(3):
        assert (np.diff(G[out["stretch_index"] == s]) == 1).all()


@pytest.mark.parametrize("success", [True, False])
def test_no_intervention_equals_plain_returns(success):
    plain = compute_episode_returns(37, success, c_fail=80)
    cut = compute_episode_returns_interventions(37, success, 80, np.zeros(37, bool), c_interv=C)
    np.testing.assert_array_equal(cut["reward"], plain["reward"])
    np.testing.assert_array_equal(cut["return_raw"], plain["return_raw"])
    assert (cut["stretch_index"] == 0).all() and (cut["stretch_length"] == 37).all()


def test_start_under_human_control_and_failure_end():
    h = human(12, (0, 3), (6, 8))
    np.testing.assert_array_equal(intervention_onsets(h), [6])  # no onset at frame 0: no robot action caused it
    out = compute_episode_returns_interventions(12, False, c_fail=500, intervention=h, c_interv=C)
    assert out["reward"][5] == -C and out["reward"][-1] == -500
    assert out["return_raw"][0] == -5 - C and out["return_raw"][6] == -5 - 500
    with pytest.raises(ValueError):
        compute_episode_returns_interventions(12, True, 1, h[:5], C)


def intervention_table(lengths, successes, runs_per_episode, c_interv=None):
    eps = make_episodes(lengths, successes)
    flags = np.concatenate([human(n, *runs) for n, runs in zip(lengths, runs_per_episode, strict=True)])
    series = pd.Series(flags, index=np.arange(len(flags)))
    c = default_intervention_penalty(eps, series) if c_interv is None else c_interv
    return eps, series, *compute_returns_table(eps, c_fail=c, interventions=series, c_interv=c)


def test_table_with_interventions_and_default_penalty():
    lengths, runs = [20, 30, 12, 25], [[(5, 8), (12, 15)], [], [(4, 6)], [(20, 23)]]
    eps, series, table, meta = intervention_table(lengths, [True] * 4, runs)
    # longest stretch ending in success: episode 1 has no intervention -> 30 frames
    assert meta["c_interv"] == 30 and meta["tag"] == "i30"
    assert list(table.columns[-4:]) == INTERVENTION_COLUMNS
    assert meta["num_interventions"] == 4 and meta["num_stretches"] == 8
    assert meta["episodes_by_num_interventions"] == {"0": 1, "1": 2, "2": 1}
    assert meta["Z"] == -table["return_raw"].min() and table["return_norm"].min() == -1.0
    # every frame of a stretch that ends in a takeover scores below every frame of a stretch that ends in success
    last = table["stretch_index"] == table.groupby("episode_index")["stretch_index"].transform("max")
    assert table.loc[~last, "return_raw"].max() < table.loc[last, "return_raw"].min()
    np.testing.assert_array_equal(table["intervention"].to_numpy(), series.to_numpy())


def test_table_without_interventions_has_no_extra_columns():
    table, meta = compute_returns_table(REAL_LIKE, c_fail=2229)
    assert list(table.columns) == RETURNS_COLUMNS and "c_interv" not in meta and meta["tag"] == "c2229"
    with pytest.raises(ValueError):
        compute_returns_table(REAL_LIKE, 2229, c_interv=10)


def test_intervention_returns_roundtrip_and_missing_success(tmp_path):
    _, _, table, meta = intervention_table([20, 9], [True, True], [[(5, 8)], []], c_interv=C)
    save_returns(table, meta, cache_dir=tmp_path)
    loaded, loaded_meta = load_returns(meta["tag"], cache_dir=tmp_path)
    pd.testing.assert_frame_equal(loaded, table)
    assert loaded_meta == meta

    raw = tmp_path / "raw"
    (raw / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    make_episodes([20, 9], [True, True]).drop(columns="success").to_parquet(
        raw / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    )
    with pytest.raises(KeyError):
        load_episodes(raw, prefer_cache=False)
    assert load_episodes(raw, prefer_cache=False, assume_success=True)["success"].all()
