"""Dataset-level constants shared by every step. Import from here; do not redefine."""

DATASET_REPO_ID = "DreamMachines/20h_fullft_eval_success"
DATASET_NAME = "20h_fullft_eval_success"

FPS = 50
NUM_EPISODES = 800
NUM_FRAMES = 634_838
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
