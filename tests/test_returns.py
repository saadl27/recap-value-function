import numpy as np
import pandas as pd
import pytest

from recap_vf import bins
from recap_vf.returns import (
    RETURNS_COLUMNS,
    compute_episode_returns,
    compute_returns_table,
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
