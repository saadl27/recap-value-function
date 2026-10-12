"""Step 4: train the value function (RECAP Monte-Carlo targets, 201-bin cross-entropy).

Workstation (main run):  uv run python scripts/04_train_value.py --config configs/train_default.yaml
Resume / extend:         uv run python scripts/04_train_value.py --config configs/train_default.yaml --resume \
                             [--set steps=3000]
Short smoke run:         ... --run-name smoke --set steps=20 eval_every=10 val_heldout_episodes=0,19 \
                             val_train_num_episodes=2
See docs/steps/04_training.md and src/recap_vf/train.py.
"""

from __future__ import annotations

import argparse
import logging

from recap_vf.train import resolve_config, train


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/train_default.yaml", help="YAML config (TrainConfig fields)")
    p.add_argument("--set", nargs="+", action="extend", default=[], metavar="KEY=VALUE",
                   help="config overrides, YAML-parsed values, dotted keys for model.*")  # fmt: skip
    p.add_argument("--run-name", default=None, help="override run_name (output: runs/<run_name>/)")
    p.add_argument("--resume", action="store_true", help="resume runs/<run_name>/checkpoints/last (saved config)")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = resolve_config(args.config, args.set, resume=args.resume, run_name=args.run_name)
    train(cfg, resume=args.resume)


if __name__ == "__main__":  # required: DataLoader workers use forkserver
    main()
