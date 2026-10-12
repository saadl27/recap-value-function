"""Step 1: stratified held-out set and cross-fitting folds."""

import numpy as np
import pandas as pd
import pytest

from recap_vf.data.splits import length_strata, load_splits, make_splits, save_splits, split_summary


def _episodes(n=800, n_success=423, seed=1):
    rng = np.random.default_rng(seed)
    success = np.zeros(n, bool)
    success[rng.choice(n, n_success, replace=False)] = True
    length = np.where(success, rng.integers(382, 2078, n), rng.integers(157, 2230, n))
    return pd.DataFrame({"episode_index": np.arange(n), "length": length, "success": success})


@pytest.fixture(scope="module")
def eps():
    return _episodes()


@pytest.fixture(scope="module")
def splits(eps):
    return make_splits(eps, seed=0)


def test_disjoint_and_complete(eps, splits):
    held, train = set(splits["heldout"]), set(splits["train"])
    assert not held & train
    assert held | train == set(eps["episode_index"])
    assert len(held) == 120 and len(train) == 680
    fold_sets = [set(v) for v in splits["folds"].values()]
    assert sorted(splits["folds"]) == ["0", "1", "2", "3", "4"]
    assert sum(len(s) for s in fold_sets) == 800 and set().union(*fold_sets) == set(eps["episode_index"])


def test_stratified_within_one_episode(eps, splits):
    succ = eps.set_index("episode_index")["success"]
    n_s, n_f = int(succ.sum()), int((~succ).sum())
    held_s = int(succ.loc[splits["heldout"]].sum())
    assert abs(held_s - 0.15 * n_s) <= 1 and abs((120 - held_s) - 0.15 * n_f) <= 1
    # length strata are represented proportionally in held-out
    strata = length_strata(eps).to_numpy()
    for s in np.unique(strata):
        n = (strata == s).sum()
        k = np.isin(eps["episode_index"][strata == s], splits["heldout"]).sum()
        assert abs(k - 0.15 * n) <= 1, (s, k, n)
    # folds: sizes, per-class and per-stratum counts differ by at most 1
    for labels in (succ.to_numpy(), strata, np.zeros(len(eps), int)):
        for v in np.unique(labels):
            members = set(eps["episode_index"][labels == v])
            counts = [len(members & set(f)) for f in splits["folds"].values()]
            assert max(counts) - min(counts) <= 1, (v, counts)


def test_deterministic_and_seeded(eps, splits):
    again = make_splits(eps, seed=0)
    assert again == splits
    other = make_splits(eps, seed=1)
    assert other["heldout"] != splits["heldout"]
    assert other["seed"] == 1


def test_small_dataset_and_io(tmp_path):
    small = _episodes(10, 4)
    sp = make_splits(small, seed=0)
    assert len(sp["heldout"]) == 2 and len(sp["train"]) == 8
    save_splits(sp, tmp_path / "s.json")
    assert load_splits(tmp_path / "s.json") == sp
    summary = split_summary(small, sp)
    assert summary.loc["all", "episodes"] == 10 and summary.loc["heldout", "episodes"] == 2


def test_stratify_on_another_class_column():
    # all-success data (DAgger): stratify on the number of interventions instead of success
    rng = np.random.default_rng(3)
    eps = pd.DataFrame(
        {
            "episode_index": np.arange(240),
            "length": rng.integers(800, 1700, 240),
            "success": True,
            "interventions": rng.choice([0, 1, 2], 240, p=[0.12, 0.58, 0.30]),
        }
    )
    sp = make_splits(eps, seed=0, class_column="interventions")
    assert len(sp["heldout"]) == 36 and "interventions" in sp["stratified_on"]
    held = eps.set_index("episode_index").loc[sp["heldout"], "interventions"].value_counts()
    expected = eps["interventions"].value_counts() * 0.15
    assert (abs(held.reindex(expected.index, fill_value=0) - expected) <= 1).all()
    for fold in sp["folds"].values():
        counts = eps.set_index("episode_index").loc[fold, "interventions"].value_counts()
        assert (abs(counts.reindex(expected.index, fill_value=0) - eps["interventions"].value_counts() / 5) <= 1).all()
