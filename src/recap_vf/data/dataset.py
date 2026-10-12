"""`FrameDataset`: random access to (3 camera images, state) for every frame of a set of episodes.

Backed by the per-episode HDF5 JPEG store written by `recap_vf.data.frames` and by `frames.parquet` for state and
bookkeeping. HDF5 handles are opened lazily inside each dataloader worker (never inherited across fork) and kept in a
small per-process LRU cache.
"""

from __future__ import annotations

import io
import os
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from recap_vf import paths
from recap_vf.constants import CAMERAS, STATE_DIM

DECODERS = ("torchvision", "pil")


def decode_jpeg(data: np.ndarray, decoder: str = "torchvision") -> torch.Tensor:
    """JPEG bytes (uint8 array) -> uint8 Tensor[3, H, W] RGB."""
    if decoder == "torchvision":
        from torchvision.io import ImageReadMode
        from torchvision.io import decode_jpeg as tv_decode

        if not data.flags.writeable:  # e.g. np.frombuffer(bytes); h5py reads are already writable copies
            data = data.copy()
        return tv_decode(torch.from_numpy(data), mode=ImageReadMode.RGB)
    if decoder == "pil":
        from PIL import Image

        rgb = np.asarray(Image.open(io.BytesIO(data.tobytes())).convert("RGB"))
        return torch.from_numpy(rgb.copy()).permute(2, 0, 1).contiguous()
    raise ValueError(f"unknown decoder {decoder!r}, expected one of {DECODERS}")


class FrameDataset(Dataset):
    """All frames (optionally every `frame_stride`-th) of `episodes`, ordered by global `index`.

    Item (see docs/conventions.md, "Batch contract"):
      images: {cam: uint8 Tensor[3, 224, 224]}, state: float32 Tensor[14],
      index / episode_index / frame_index: int64 0-d Tensors.
    """

    def __init__(
        self,
        episodes: Sequence[int],
        cameras: Sequence[str] = CAMERAS,
        frame_stride: int = 1,
        *,
        frames_dir: Path | str = paths.FRAMES_DIR,
        frame_table: Path | str = paths.FRAME_TABLE,
        decoder: str = "torchvision",
        max_open_files: int = 64,
    ) -> None:
        if frame_stride < 1:
            raise ValueError("frame_stride must be >= 1")
        if decoder not in DECODERS:
            raise ValueError(f"unknown decoder {decoder!r}")
        self.episodes = sorted({int(e) for e in episodes})
        self.cameras = tuple(cameras)
        self.frame_stride = frame_stride
        self.frames_dir = Path(frames_dir)
        self.decoder = decoder
        self.max_open_files = max_open_files

        table = pq.read_table(
            frame_table,
            columns=["index", "episode_index", "frame_index", "state"],
            filters=[("episode_index", "in", self.episodes)] if self.episodes else None,
        ).sort_by("index")
        found = set(np.unique(table["episode_index"].to_numpy()).tolist())
        if missing := [e for e in self.episodes if e not in found]:
            raise KeyError(f"episodes not in {frame_table}: {missing[:10]}")
        if missing := [e for e in self.episodes if not self._path(e).is_file()]:
            raise FileNotFoundError(
                f"{len(missing)} episodes have no frame store file in {self.frames_dir}: {missing[:10]}"
            )

        ep = table["episode_index"].to_numpy()
        fi = table["frame_index"].to_numpy()
        state = pc.list_flatten(table["state"].combine_chunks()).to_numpy().reshape(-1, STATE_DIM)
        self._all = {  # every frame of the selected episodes (used by get_episode)
            "index": table["index"].to_numpy().astype(np.int64),
            "episode_index": ep.astype(np.int64),
            "frame_index": fi.astype(np.int64),
            "state": state.astype(np.float32),
        }
        keep = np.flatnonzero(fi % frame_stride == 0)
        self._items = {k: v[keep] for k, v in self._all.items()}
        self._handles: OrderedDict[int, h5py.File] = OrderedDict()
        self._pid = os.getpid()

    # ---------------------------------------------------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._items["index"])

    @property
    def index(self) -> np.ndarray:
        """Global frame `index` of every item, in item order."""
        return self._items["index"]

    @property
    def episode_index(self) -> np.ndarray:
        return self._items["episode_index"]

    @property
    def frame_index(self) -> np.ndarray:
        return self._items["frame_index"]

    def __getitem__(self, i: int) -> dict:
        if i < 0:
            i += len(self)
        if not 0 <= i < len(self):
            raise IndexError(i)
        e = int(self._items["episode_index"][i])
        fi = int(self._items["frame_index"][i])
        f = self._file(e)
        return {
            "images": {cam: decode_jpeg(f[cam][fi], self.decoder) for cam in self.cameras},
            "state": torch.from_numpy(self._items["state"][i].copy()),
            "index": torch.tensor(self._items["index"][i], dtype=torch.int64),
            "episode_index": torch.tensor(e, dtype=torch.int64),
            "frame_index": torch.tensor(fi, dtype=torch.int64),
        }

    def get_episode(
        self, episode_index: int, frames: Sequence[int] | None = None, cameras: Sequence[str] | None = None
    ) -> dict:
        """All frames (or the given `frames`) of one selected episode, ignoring `frame_stride`.

        Returns images {cam: uint8 Tensor[T, 3, 224, 224]}, state float32 [T, 14], index / frame_index int64 [T],
        episode_index (int), success (bool). Note: a whole long episode is ~0.3 GB per camera decoded.
        """
        e = int(episode_index)
        sel = np.flatnonzero(self._all["episode_index"] == e)
        if not len(sel):
            raise KeyError(f"episode {e} is not part of this dataset")
        if frames is not None:
            frames = np.asarray(frames, dtype=np.int64)
            if frames.min(initial=0) < 0 or frames.max(initial=0) >= len(sel):
                raise IndexError(f"frame indices out of range for episode {e} (length {len(sel)})")
            sel = sel[frames]
        f = self._file(e)
        fis = self._all["frame_index"][sel]
        cams = self.cameras if cameras is None else tuple(cameras)
        images = {}
        for cam in cams:
            ds = f[cam]
            images[cam] = (
                torch.stack([decode_jpeg(ds[int(k)], self.decoder) for k in fis])
                if len(fis)
                else torch.empty(0, 3, 0, 0, dtype=torch.uint8)
            )
        return {
            "images": images,
            "state": torch.from_numpy(self._all["state"][sel].copy()),
            "index": torch.from_numpy(self._all["index"][sel].copy()),
            "frame_index": torch.from_numpy(fis.copy()),
            "episode_index": e,
            "success": bool(f.attrs["success"]),
        }

    # ---------------------------------------------------------------------------------------------------------------
    def _path(self, e: int) -> Path:
        return self.frames_dir / paths.episode_frames_path(e).name

    def _file(self, e: int) -> h5py.File:
        """Per-process lazily opened handle (reset after fork so handles are never shared between processes)."""
        if self._pid != os.getpid():
            self._handles = OrderedDict()
            self._pid = os.getpid()
        f = self._handles.get(e)
        if f is not None:
            self._handles.move_to_end(e)
            return f
        f = h5py.File(self._path(e), "r")
        self._handles[e] = f
        if len(self._handles) > self.max_open_files:
            _, old = self._handles.popitem(last=False)
            old.close()
        return f

    def close(self) -> None:
        for f in self._handles.values():
            f.close()
        self._handles.clear()

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_handles"] = OrderedDict()  # never pickle open HDF5 handles (spawn / forkserver workers)
        return state

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001 - interpreter shutdown
            pass
