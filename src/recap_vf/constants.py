"""Dataset-level constants shared by every step. Import from here; do not redefine.

The dataset is chosen with the env var RECAP_DATASET (default: the 800-episode evaluation rollouts). Every dataset
listed here has the same robot, cameras, state, fps and task; only the id, the size and the success flag differ.
"""

import os

DATASETS = {
    "20h_fullft_eval_success": {
        "repo_id": "DreamMachines/20h_fullft_eval_success",
        "num_episodes": 800,
        "num_frames": 634_838,
        "all_succeed": False,
    },
    # DAgger rollouts with human interventions (docs/interventions.md). No `success` column: every episode ends with
    # the task done (the human steps in until it is), so all episodes are successes.
    "actuator_unboxing_1h_HQ_dagger_rollouts": {
        "repo_id": "DreamMachines/actuator_unboxing_1h_HQ_dagger_rollouts",
        "num_episodes": 240,
        "num_frames": 256_537,
        "all_succeed": True,
    },
}
DEFAULT_DATASET = "20h_fullft_eval_success"
DATASET_NAME = os.environ.get("RECAP_DATASET", DEFAULT_DATASET)
if DATASET_NAME not in DATASETS:
    raise ValueError(f"RECAP_DATASET={DATASET_NAME!r} is not one of {sorted(DATASETS)}")

DATASET_REPO_ID = DATASETS[DATASET_NAME]["repo_id"]
NUM_EPISODES = DATASETS[DATASET_NAME]["num_episodes"]
NUM_FRAMES = DATASETS[DATASET_NAME]["num_frames"]
ALL_EPISODES_SUCCEED = DATASETS[DATASET_NAME]["all_succeed"]

FPS = 50
STATE_DIM = 14
IMAGE_SIZE = 224

# Order matters: it is the order in which camera tokens are fed to the model.
CAMERAS = ("top", "left_wrist", "right_wrist")
VIDEO_KEYS = {cam: f"observation.images.{cam}" for cam in CAMERAS}

PROMPT = "Take an actuator from the box and place it in the tray"

# Backbones (see docs/steps/03_model.md). unsloth/gemma-3-270m is an ungated mirror of google/gemma-3-270m.
VISION_BACKBONE = "google/siglip2-so400m-patch14-224"
LANGUAGE_BACKBONE = "google/gemma-3-270m"  # gated: needs licence acceptance on the HF account
LANGUAGE_BACKBONE_MIRROR = "unsloth/gemma-3-270m"
