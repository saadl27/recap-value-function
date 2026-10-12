"""Step 1.4 — stratified held-out set (15 %) + 5 cross-fitting folds -> data/cache/splits.json.

    env -u PYTHONPATH uv run python scripts/01_make_splits.py --seed 0
Needs data/cache/episodes.parquet (scripts/01_extract_frames.py --tables-only). Splits always cover all episodes.
DAgger data (RECAP_DATASET=actuator_unboxing_1h_HQ_dagger_rollouts): add `--stratify interventions`.
"""

import argparse

import pandas as pd

from recap_vf import paths
from recap_vf.data.splits import HELDOUT_FRAC, LENGTH_BINS, NUM_FOLDS, make_splits, save_splits, split_summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--heldout-frac", type=float, default=HELDOUT_FRAC)
    ap.add_argument("--folds", type=int, default=NUM_FOLDS)
    ap.add_argument("--length-bins", type=int, default=LENGTH_BINS)
    ap.add_argument("--out", default=str(paths.SPLITS_FILE))
    ap.add_argument(
        "--stratify",
        choices=["success", "interventions"],
        default="success",
        help="outer stratum: the success flag, or the number of human interventions per episode (0/1/2+, DAgger data)",
    )
    args = ap.parse_args()

    episodes = pd.read_parquet(paths.EPISODE_TABLE)
    class_column = "success"
    if args.stratify == "interventions":
        from recap_vf.returns import intervention_onsets, load_interventions

        flags = load_interventions(paths.RAW_DIR)
        episodes["interventions"] = [
            min(len(intervention_onsets(flags.loc[a : b - 1].to_numpy())), 2)
            for a, b in zip(episodes["dataset_from_index"], episodes["dataset_to_index"], strict=True)
        ]
        class_column = "interventions"
        print(
            "episodes by interventions (capped at 2):", episodes["interventions"].value_counts().sort_index().to_dict()
        )
    splits = make_splits(episodes, args.seed, args.heldout_frac, args.folds, args.length_bins, class_column)
    save_splits(splits, args.out)
    pd.set_option("display.width", 160)
    print(split_summary(episodes, splits).round(3).to_string())
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
