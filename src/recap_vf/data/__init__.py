"""Step 1 — data pipeline: raw-dataset tables, JPEG/HDF5 frame store, splits and `FrameDataset`.

Exports are resolved lazily so that importing `recap_vf.data.frames` (e.g. in the video-decoder processes) does not
pull in torch.
"""

from importlib import import_module

_EXPORTS = {
    "FrameDataset": "recap_vf.data.dataset",
    "decode_jpeg": "recap_vf.data.dataset",
    "load_splits": "recap_vf.data.splits",
    "make_splits": "recap_vf.data.splits",
    "save_splits": "recap_vf.data.splits",
    "split_summary": "recap_vf.data.splits",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name in _EXPORTS:
        return getattr(import_module(_EXPORTS[name]), name)
    raise AttributeError(f"module 'recap_vf.data' has no attribute {name!r}")
