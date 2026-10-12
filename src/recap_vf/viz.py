"""Step 7 — rendering helpers for the qualitative evaluation of the value function.

Building blocks shared by `scripts/07_render_episode.py`, `scripts/07_keyframes.py` and
`scripts/07_select_episodes.py`:

* `load_episode_series` — one episode's V(t), g(t), A(t), I(t) from the step-6 artefacts + the step-2 returns table.
* `compose_camera_strip` — the three cameras side by side (order of `constants.CAMERAS`) with a header line.
* `SeriesPanel` — the V/g + A time-series figure rendered **once** with matplotlib; per video frame only a cursor
  is painted into a copy of the cached pixels (fast enough to render an episode in seconds on CPU).
* `Mp4Writer` — H.264 (libx264, CRF) writer through PyAV.
* `render_episode_video` / `render_keyframe_strip` — the two deliverables.
* `select_keyframes`, `v_dip_depth`, `select_eval_episodes` — pure NumPy/pandas selection logic (unit-tested).

Matplotlib is used with the Agg backend only (no display needed).
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from recap_vf import paths  # noqa: E402
from recap_vf.constants import CAMERAS, FPS, IMAGE_SIZE  # noqa: E402

log = logging.getLogger(__name__)

CAMERA_LABELS = {"top": "top", "left_wrist": "left wrist", "right_wrist": "right wrist"}

# Colours (light chart surface). V and A carry identity; g is a neutral reference; I_t=True is a green wash.
COLORS = {
    "surface": "#fcfcfb",
    "ink": "#0b0b0b",
    "ink2": "#52514e",
    "grid": "#e4e3df",
    "value": "#2a78d6",  # V(t)
    "return": "#52514e",  # g(t), dashed
    "advantage": "#eb6834",  # A(t)
    "positive": "#1baf7a",  # I_t = True shading
    "cursor": "#0b0b0b",
    "header_bg": "#1a1a19",
    "header_fg": "#f0efec",
    "success": "#1baf7a",
    "failure": "#e34948",
}

SERIES_COLUMNS = ["index", "frame_index", "value", "return_norm", "advantage", "indicator"]


# ---------------------------------------------------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------------------------------------------------
def _read_episode(path: Path, episode: int, columns: Sequence[str] | None = None) -> pd.DataFrame:
    table = pq.read_table(path, columns=list(columns) if columns else None, filters=[("episode_index", "==", episode)])
    return table.to_pandas()


def load_episode_series(
    run_dir: Path | str,
    episode: int,
    advantages: str | None = "advantages_50_0.4",
    returns_tag: str = "c2229",
    values_file: str = "values.parquet",
    returns_path: Path | str | None = None,
) -> pd.DataFrame:
    """One episode's per-frame series, sorted by `frame_index`.

    Columns: `index, frame_index, value, return_norm, advantage, indicator`. Values (and A, I) come from
    `runs/<run>/<advantages>.parquet` when `advantages` is given, else from `values.parquet` (then `advantage` is
    NaN and `indicator` False). `return_norm` (g_t) is joined from the step-2 returns table on the global `index`.
    """
    run_dir = Path(run_dir)
    e = int(episode)
    if advantages:
        name = advantages if advantages.endswith(".parquet") else f"{advantages}.parquet"
        df = _read_episode(run_dir / name, e, ["index", "frame_index", "value", "advantage", "indicator"])
    else:
        df = _read_episode(run_dir / values_file, e, ["index", "frame_index", "value"])
        df["advantage"] = np.nan
        df["indicator"] = False
    if not len(df):
        raise KeyError(f"episode {e} not found in {run_dir}")
    rpath = Path(returns_path) if returns_path is not None else paths.returns_paths(returns_tag)[0]
    ret = _read_episode(rpath, e, ["index", "return_norm"])
    df = df.merge(ret, on="index", how="left", validate="one_to_one")
    if df["return_norm"].isna().any():
        raise ValueError(f"episode {e}: {int(df['return_norm'].isna().sum())} frames missing from {rpath}")
    df = df.sort_values("frame_index").reset_index(drop=True)
    if not np.array_equal(df["frame_index"].to_numpy(), np.arange(len(df))):
        raise ValueError(f"episode {e}: frame_index is not 0..T-1 in {run_dir}")
    df["indicator"] = df["indicator"].astype(bool)
    return df[SERIES_COLUMNS]


def load_advantage_meta(run_dir: Path | str, advantages: str | None) -> dict:
    """The step-6 JSON next to the advantages parquet ({} if absent)."""
    if not advantages:
        return {}
    p = Path(run_dir) / f"{advantages.removesuffix('.parquet')}.json"
    return json.loads(p.read_text()) if p.is_file() else {}


def to_hwc(img) -> np.ndarray:
    """uint8 CHW Tensor/array (or HWC array) -> contiguous HWC uint8 NumPy array."""
    a = img.numpy() if hasattr(img, "numpy") else np.asarray(img)
    if a.ndim == 3 and a.shape[0] == 3 and a.shape[-1] != 3:
        a = a.transpose(1, 2, 0)
    return np.ascontiguousarray(a, dtype=np.uint8)


# ---------------------------------------------------------------------------------------------------------------------
# Text / camera strip
# ---------------------------------------------------------------------------------------------------------------------
@lru_cache(maxsize=16)
def get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """DejaVu Sans shipped with matplotlib (always available, no system fonts needed)."""
    path = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold" if bold else "normal"))
    return ImageFont.truetype(path, size)


def draw_label(
    img: Image.Image, xy: tuple[int, int], text: str, size: int = 12, fg: str = "#ffffff", bg=(0, 0, 0, 150)
) -> None:
    """Text with a translucent box behind it (legible on any camera image)."""
    font = get_font(size)
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    x0, y0, x1, y1 = d.textbbox(xy, text, font=font)
    d.rectangle((x0 - 3, y0 - 2, x1 + 3, y1 + 2), fill=bg)
    d.text(xy, text, font=font, fill=fg)
    img.paste(Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB"))


def header_bar(width: int, height: int, left: str, right: str = "", size: int = 13) -> np.ndarray:
    """Dark one-line header (left- and right-aligned text) -> HWC uint8."""
    img = Image.new("RGB", (width, height), COLORS["header_bg"])
    d = ImageDraw.Draw(img)
    font = get_font(size)
    ty = (height - size) // 2 - 1
    d.text((8, ty), left, font=font, fill=COLORS["header_fg"])
    if right:
        w = d.textlength(right, font=font)
        d.text((width - 8 - w, ty), right, font=font, fill=COLORS["header_fg"])
    return np.asarray(img)


def compose_camera_strip(
    images: dict[str, object] | Sequence[object],
    cameras: Sequence[str] = CAMERAS,
    labels: bool = True,
    label_size: int = 11,
) -> np.ndarray:
    """Cameras side by side in `cameras` order (default top | left wrist | right wrist) -> HWC uint8 [H, n*W, 3]."""
    tiles = [to_hwc(images[c]) for c in cameras] if isinstance(images, dict) else [to_hwc(x) for x in images]
    h = tiles[0].shape[0]
    if any(t.shape[0] != h for t in tiles):
        raise ValueError("all camera images must have the same height")
    strip = np.concatenate(tiles, axis=1)
    if not labels:
        return strip
    img = Image.fromarray(strip)
    x = 0
    for cam, t in zip(cameras, tiles, strict=True):
        draw_label(img, (x + 5, 4), CAMERA_LABELS.get(cam, cam), size=label_size)
        x += t.shape[1]
    return np.asarray(img)


def blank_tile(size: int = IMAGE_SIZE, text: str = "after end") -> np.ndarray:
    img = Image.new("RGB", (size, size), "#3a3a38")
    d = ImageDraw.Draw(img)
    font = get_font(13)
    w = d.textlength(text, font=font)
    d.text(((size - w) / 2, size / 2 - 8), text, font=font, fill="#c3c2b7")
    return np.asarray(img)


# ---------------------------------------------------------------------------------------------------------------------
# Time-series panel
# ---------------------------------------------------------------------------------------------------------------------
def true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Half-open [start, end) runs of True in a 1-D boolean array."""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.diff(m.astype(np.int8))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist(), strict=True))


class SeriesPanel:
    """V(t) solid + g(t) dashed (top axes), A(t) (bottom axes, shared time axis), I_t=True shaded green.

    The figure is drawn once; `frame(t)` returns the cached pixels with a cursor (vertical line + dots on V and A)
    painted at frame `t`. Shown as two stacked axes rather than a twin-y axis so the two scales never mix.
    """

    def __init__(
        self,
        series: pd.DataFrame,
        width: int,
        height: int,
        *,
        title: str | None = None,
        epsilon: float | None = None,
        highlight: tuple[float, float] | None = None,
        marks: Sequence[float] = (),
        dpi: int = 100,
        font_size: float = 8.5,
    ) -> None:
        self.n = len(series)
        t = series["frame_index"].to_numpy() / FPS
        v = series["value"].to_numpy(dtype=float)
        g = series["return_norm"].to_numpy(dtype=float)
        a = series["advantage"].to_numpy(dtype=float)
        ind = series["indicator"].to_numpy(dtype=bool)
        self.has_adv = bool(np.isfinite(a).any())

        rc = {
            "font.size": font_size,
            "axes.edgecolor": COLORS["ink2"],
            "axes.labelcolor": COLORS["ink2"],
            "xtick.color": COLORS["ink2"],
            "ytick.color": COLORS["ink2"],
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
        with plt.rc_context(rc):
            fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi, facecolor=COLORS["surface"])
            if self.has_adv:
                gs = fig.add_gridspec(2, 1, height_ratios=(3, 2), hspace=0.12)
                ax_v = fig.add_subplot(gs[0])
                ax_a = fig.add_subplot(gs[1], sharex=ax_v)
                axes = [ax_v, ax_a]
            else:
                ax_v = fig.add_subplot(111)
                ax_a = None
                axes = [ax_v]
            for ax in axes:
                ax.set_facecolor(COLORS["surface"])
                ax.grid(True, color=COLORS["grid"], lw=0.6)
                ax.set_axisbelow(True)
                for s, e in true_runs(ind):
                    # span [s, e) frames; shade to the next frame's start so 1-frame runs stay visible
                    ax.axvspan(
                        t[s], t[min(e, self.n - 1)] if e < self.n else t[-1], color=COLORS["positive"], alpha=0.18, lw=0
                    )
                if highlight is not None:
                    ax.axvspan(*highlight, color=COLORS["ink2"], alpha=0.10, lw=0)
                for m in marks:
                    ax.axvline(m, color=COLORS["ink2"], lw=0.8, ls=":")
            ax_v.plot(t, g, color=COLORS["return"], lw=1.2, ls=(0, (4, 3)), label="g(t) true return")
            ax_v.plot(t, v, color=COLORS["value"], lw=1.6, label="V(t)")
            lo = float(np.nanmin([np.nanmin(v), np.nanmin(g)]))
            hi = float(np.nanmax([np.nanmax(v), np.nanmax(g)]))
            pad = 0.06 * max(hi - lo, 0.05)
            ax_v.set_ylim(max(lo - pad, -1.05), min(hi + pad, 0.05))
            ax_v.set_xlim(t[0], t[-1] if self.n > 1 else t[0] + 1 / FPS)
            ax_v.set_ylabel("V(t), g(t)")
            handles, labels = ax_v.get_legend_handles_labels()
            if ind.any():
                handles.append(Patch(color=COLORS["positive"], alpha=0.35, lw=0))
                labels.append("I_t = True")
            if ax_a is not None:
                handles.append(Line2D([], [], color=COLORS["advantage"], lw=1.2))
                labels.append("A(t)")
            # legend in a row above the plot so it never hides data
            ax_v.legend(
                handles,
                labels,
                loc="lower right",
                bbox_to_anchor=(1.0, 1.0),
                ncol=len(labels),
                frameon=False,
                fontsize=font_size - 0.5,
                borderpad=0.1,
                borderaxespad=0.1,
                handlelength=1.8,
                columnspacing=1.2,
            )
            if title:
                ax_v.set_title(title, loc="left", fontsize=font_size + 0.5, color=COLORS["ink"], pad=4)
            if ax_a is not None:
                ax_a.plot(t, a, color=COLORS["advantage"], lw=1.2, label="A(t)")
                ax_a.axhline(0, color=COLORS["ink2"], lw=0.7)
                if epsilon is not None and np.isfinite(epsilon):
                    ax_a.axhline(epsilon, color=COLORS["positive"], lw=0.9, ls=(0, (2, 2)))
                    ax_a.annotate(
                        "ε",
                        (1.0, epsilon),
                        xycoords=("axes fraction", "data"),
                        xytext=(2, 0),
                        textcoords="offset points",
                        va="center",
                        color=COLORS["ink2"],
                    )
                q = np.nanpercentile(np.abs(a), 99.5) if np.isfinite(a).any() else 1.0
                amax = max(float(q) * 1.15, abs(epsilon or 0) * 1.5, 1e-4)
                ax_a.set_ylim(-amax, amax)
                ax_a.set_ylabel("A(t)")
                ax_a.set_xlabel("time [s]")
                plt.setp(ax_v.get_xticklabels(), visible=False)
            else:
                ax_v.set_xlabel("time [s]")
            # margins in pixels so the layout holds for any panel size
            fig.subplots_adjust(left=78 / width, right=1 - 16 / width, top=1 - 20 / height, bottom=40 / height)
            for ax in axes:  # y-labels at a fixed 14 px from the left edge, whatever the tick-label width
                ax.yaxis.set_label_coords(-(78 - 14) / (width - 78 - 16), 0.5)
            fig.canvas.draw()
            self.background = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()

            H = self.background.shape[0]

            def px(ax, xs, ys):
                xy = ax.transData.transform(np.column_stack([xs, ys]))
                return np.round(xy[:, 0]).astype(int), np.round(H - xy[:, 1]).astype(int)

            self._x, self._vy = px(ax_v, t, np.clip(v, *ax_v.get_ylim()))
            self._spans = []
            for ax in axes:
                bb = ax.get_window_extent()
                self._spans.append((int(round(H - bb.y1)), int(round(H - bb.y0))))
            self._ay = None
            if ax_a is not None:
                _, ay = px(ax_a, t, np.clip(np.nan_to_num(a), *ax_a.get_ylim()))
                self._ay = np.where(np.isfinite(a), ay, -1)
            plt.close(fig)
        self.cursor_rgb = np.array(to_rgb(COLORS["cursor"])) * 255
        self.dot_rgb = {
            "v": np.array(to_rgb(COLORS["value"])) * 255,
            "a": np.array(to_rgb(COLORS["advantage"])) * 255,
        }

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.background.shape

    def _dot(self, img: np.ndarray, x: int, y: int, rgb: np.ndarray, r: int = 4) -> None:
        H, W = img.shape[:2]
        y0, y1, x0, x1 = max(y - r - 1, 0), min(y + r + 2, H), max(x - r - 1, 0), min(x + r + 2, W)
        if y0 >= y1 or x0 >= x1:
            return
        yy, xx = np.mgrid[y0:y1, x0:x1]
        d2 = (yy - y) ** 2 + (xx - x) ** 2
        img[y0:y1, x0:x1][d2 <= (r + 1.5) ** 2] = 255  # white ring separates the dot from the line
        img[y0:y1, x0:x1][d2 <= r**2] = rgb

    def frame(self, t: int | None) -> np.ndarray:
        """Panel pixels with the cursor at frame `t` (None -> no cursor)."""
        img = self.background.copy()
        if t is None:
            return img
        t = int(np.clip(t, 0, self.n - 1))
        x = int(self._x[t])
        for y0, y1 in self._spans:
            img[y0:y1, max(x - 1, 0) : x + 1] = self.cursor_rgb
        self._dot(img, x, int(self._vy[t]), self.dot_rgb["v"])
        if self._ay is not None and self._ay[t] >= 0:
            self._dot(img, x, int(self._ay[t]), self.dot_rgb["a"], r=3)
        return img


# ---------------------------------------------------------------------------------------------------------------------
# Video
# ---------------------------------------------------------------------------------------------------------------------
class Mp4Writer:
    """Minimal video writer (PyAV, yuv420p, constant quality): H.264 for `.mp4`, VP9 for `.webm`.

    WebM/VP9 plays in browsers and VS Code (whose players lack H.264); MP4/H.264 is smaller and plays in desktop
    players. The file is written as `<name>.tmp<suffix>` and renamed on close, so a reader never sees a half-written
    video; MP4s get `+faststart` (index first) for players that stream the file.
    """

    def __init__(self, path: Path | str, width: int, height: int, fps: int = 25, crf: int = 28, preset: str = "medium"):
        import av

        if width % 2 or height % 2:
            raise ValueError(f"yuv420p needs even frame dimensions, got {width}x{height}")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._tmp = self.path.with_name(f"{self.path.stem}.tmp{self.path.suffix}")
        webm = self.path.suffix.lower() == ".webm"
        options = {"movflags": "+faststart"} if not webm else {}
        self._container = av.open(str(self._tmp), mode="w", options=options)
        self._stream = self._container.add_stream("libvpx-vp9" if webm else "libx264", rate=fps)
        self._stream.width, self._stream.height = width, height
        self._stream.pix_fmt = "yuv420p"
        if webm:  # VP9's CRF scale is 0-63; +6 gives about the same quality as the x264 value
            self._stream.options = {
                "crf": str(crf + 6),
                "b": "0",
                "deadline": "realtime",
                "cpu-used": "8",
                "row-mt": "1",
            }
        else:
            self._stream.options = {"crf": str(crf), "preset": preset}
        self._av = av
        self.frames = 0

    def write(self, rgb: np.ndarray) -> None:
        frame = self._av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb), format="rgb24")
        for packet in self._stream.encode(frame):
            self._container.mux(packet)
        self.frames += 1

    def close(self) -> None:
        for packet in self._stream.encode():
            self._container.mux(packet)
        self._container.close()
        os.replace(self._tmp, self.path)

    def __enter__(self) -> Mp4Writer:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _even(x: int) -> int:
    return x + (x % 2)


def episode_title(episode: int, success: bool | None, length: int) -> str:
    outcome = "" if success is None else (" · SUCCESS" if success else " · FAILURE")
    return f"episode {episode:04d}{outcome} · {length} frames ({length / FPS:.1f} s)"


def compose_video_frame(
    cams: dict[str, object],
    panel: np.ndarray,
    header_left: str,
    header_right: str,
    cameras: Sequence[str] = CAMERAS,
    header_h: int = 24,
) -> np.ndarray:
    """header bar / camera strip / time-series panel stacked vertically -> HWC uint8 (width = camera strip width)."""
    strip = compose_camera_strip(cams, cameras)
    W = strip.shape[1]
    if panel.shape[1] != W:
        raise ValueError(f"panel width {panel.shape[1]} != camera strip width {W}")
    head = header_bar(W, header_h, header_left, header_right)
    return np.concatenate([head, strip, panel], axis=0)


def frame_readout(series: pd.DataFrame, t: int) -> str:
    row = series.iloc[t]
    s = f"V {row['value']:+.3f}   g {row['return_norm']:+.3f}"
    if np.isfinite(row["advantage"]):
        s += f"   A {row['advantage']:+.4f}   I {'+' if row['indicator'] else '-'}"
    return s


def render_episode_video(
    ds,
    episode: int,
    series: pd.DataFrame,
    out_path: Path | str,
    *,
    success: bool | None = None,
    epsilon: float | None = None,
    stride: int = 2,
    fps: int = 25,
    crf: int = 28,
    panel_height: int = 300,
    chunk: int = 64,
    max_frames: int | None = None,
    preset: str = "medium",
) -> dict:
    """Render one episode to mp4: header + three cameras + V/g/A/I panel with a moving cursor.

    `ds` is a `FrameDataset` containing `episode`; frames are decoded in chunks of `chunk` (bounded memory even for
    the 2229-frame episodes). Every `stride`-th frame is written at `fps` (stride 2 @ 25 fps = real time at 50 fps).
    """
    t0 = time.perf_counter()
    T = len(series)
    frames = np.arange(0, T, stride)
    if max_frames is not None:
        frames = frames[:max_frames]
    width = IMAGE_SIZE * len(ds.cameras)
    panel = SeriesPanel(series, width, _even(panel_height), title=None, epsilon=epsilon)
    header_left = episode_title(episode, success, T)
    height = 24 + IMAGE_SIZE + panel.shape[0]
    with Mp4Writer(out_path, width, height, fps=fps, crf=crf, preset=preset) as w:
        for c0 in range(0, len(frames), chunk):
            sel = frames[c0 : c0 + chunk]
            ep = ds.get_episode(episode, frames=sel)
            for j, t in enumerate(sel):
                cams = {cam: ep["images"][cam][j] for cam in ds.cameras}
                right = f"t {t / FPS:6.2f} s   {frame_readout(series, int(t))}"
                w.write(compose_video_frame(cams, panel.frame(int(t)), header_left, right, ds.cameras))
    dt = time.perf_counter() - t0
    size = Path(out_path).stat().st_size
    return {
        "episode": int(episode),
        "frames_written": int(len(frames)),
        "seconds": dt,
        "bytes": size,
        "width": width,
        "height": height,
        "path": str(out_path),
    }


# ---------------------------------------------------------------------------------------------------------------------
# Keyframes
# ---------------------------------------------------------------------------------------------------------------------
def select_keyframes(
    advantage: np.ndarray, k: int, min_sep: int, mode: str = "drop", require_sign: bool = True
) -> list[int]:
    """Greedy: the `k` most negative (`drop`) or most positive (`rise`) A_t, pairwise at least `min_sep` frames apart.

    Returned in order of extremeness (index 0 = most extreme). NaNs are ignored; ties are broken by earlier frame.
    With `require_sign`, a drop must have A < 0 and a rise A > 0 (short episodes may then yield fewer than k).
    """
    a = np.asarray(advantage, dtype=float)
    if mode not in ("drop", "rise"):
        raise ValueError(f"mode must be 'drop' or 'rise', got {mode!r}")
    ok = np.isfinite(a)
    if require_sign:
        ok &= (a < 0) if mode == "drop" else (a > 0)
    valid = np.flatnonzero(ok)
    key = a[valid] if mode == "drop" else -a[valid]
    order = valid[np.argsort(key, kind="stable")]
    chosen: list[int] = []
    for i in order:
        if len(chosen) >= k:
            break
        if all(abs(int(i) - c) >= min_sep for c in chosen):
            chosen.append(int(i))
    return chosen


def render_keyframe_strip(
    ds,
    episode: int,
    series: pd.DataFrame,
    t: int,
    out_path: Path | str,
    *,
    label: str = "",
    success: bool | None = None,
    epsilon: float | None = None,
    offsets_s: Sequence[float] = (-1.0, 0.0, 1.0, 2.0),
    panel_height: int = 220,
) -> np.ndarray:
    """PNG: rows = cameras (CAMERAS order), columns = t + offsets, each column headed by time, V and A; below, the
    episode's V/g/A panel with the window shaded and the key time marked. Returns the image array."""
    T = len(series)
    fis = [int(round(t + o * FPS)) for o in offsets_s]
    inside = [0 <= f < T for f in fis]
    got = ds.get_episode(episode, frames=[f for f, ok in zip(fis, inside, strict=True) if ok])
    cols, j = [], 0
    for f, ok in zip(fis, inside, strict=True):
        if ok:
            col = np.concatenate([to_hwc(got["images"][c][j]) for c in ds.cameras], axis=0)
            j += 1
        else:
            col = np.concatenate([blank_tile(text="after end" if f >= T else "before start") for _ in ds.cameras], 0)
        cols.append(col)
    grid = Image.fromarray(np.concatenate(cols, axis=1))
    for r, cam in enumerate(ds.cameras):
        draw_label(grid, (5, r * IMAGE_SIZE + 4), CAMERA_LABELS.get(cam, cam), size=11)
    # column headers
    W = grid.size[0]
    col_head_h = 38
    head = Image.new("RGB", (W, col_head_h), COLORS["surface"])
    d = ImageDraw.Draw(head)
    f_b, f_n = get_font(12, bold=True), get_font(11)
    for c, (o, f, ok) in enumerate(zip(offsets_s, fis, inside, strict=True)):
        x = c * IMAGE_SIZE + 6
        rel = "t" if o == 0 else f"t{o:+g} s"
        d.text((x, 3), f"{rel}  ({f / FPS:.2f} s)", font=f_b, fill=COLORS["ink"] if o == 0 else COLORS["ink2"])
        if ok:
            row = series.iloc[f]
            txt = f"V {row['value']:+.3f}"
            if np.isfinite(row["advantage"]):
                txt += f"   A {row['advantage']:+.4f} {'I+' if row['indicator'] else 'I-'}"
            d.text((x, 20), txt, font=f_n, fill=COLORS["ink2"])
        if o == 0:  # frame the key column
            d.rectangle((c * IMAGE_SIZE, 0, (c + 1) * IMAGE_SIZE - 1, col_head_h - 1), outline=COLORS["ink"], width=2)
    title = header_bar(W, 26, f"{episode_title(episode, success, T)}   {label}".rstrip(), size=13)
    lo, hi = (min(fis) / FPS, max(fis) / FPS)
    panel = SeriesPanel(series, W, _even(panel_height), epsilon=epsilon, highlight=(lo, hi), marks=[t / FPS])
    img = np.concatenate([title, np.asarray(head), np.asarray(grid), panel.frame(t)], axis=0)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_png(img, out_path)
    return img


def save_png(img: np.ndarray, path: Path | str, colors: int = 256) -> None:
    """Palette PNG (~3x smaller than RGB; strips go to reports/figures, which is committed)."""
    Image.fromarray(img).quantize(colors=colors, method=Image.Quantize.FASTOCTREE).save(path, optimize=True)


# ---------------------------------------------------------------------------------------------------------------------
# Episode selection
# ---------------------------------------------------------------------------------------------------------------------
def v_dip_depth(values: np.ndarray, smooth: int = 1) -> tuple[float, int, int]:
    """Deepest dip of V: max over t of (running max of V up to t) − V_t (a peak-to-later-trough drawdown).

    `smooth` > 1 applies a centred rolling median first so a single noisy frame cannot define the dip.
    Returns (depth >= 0, peak frame, trough frame).
    """
    v = np.asarray(values, dtype=float)
    if not len(v):
        return 0.0, 0, 0
    if smooth > 1:
        v = pd.Series(v).rolling(smooth, center=True, min_periods=1).median().to_numpy()
    peak = np.maximum.accumulate(v)
    dd = peak - v
    trough = int(np.argmax(dd))
    peak_at = int(np.argmax(v[: trough + 1]))
    return float(dd[trough]), peak_at, trough


def select_eval_episodes(
    episodes: pd.DataFrame,
    pool: Sequence[int],
    values: pd.DataFrame,
    *,
    seed: int = 0,
    n_fail: int = 10,
    n_dip: int = 5,
    n_success: int = 5,
    smooth: int = 25,
) -> list[dict]:
    """Step-7 evaluation set from the episodes in `pool` (normally the held-out split).

    `n_fail` random failures; the `n_dip` successes with the deepest V dip (`v_dip_depth`); `n_success` random
    successes among the remaining ones. Deterministic for a given seed (pool and candidates are sorted first; one
    `default_rng(seed)` draws failures then successes). Groups are capped at what the pool contains.
    `values` needs `episode_index, frame_index, value` for (at least) the successful pool episodes.
    """
    eps = episodes.set_index("episode_index")
    pool = sorted({int(e) for e in pool})
    if missing := [e for e in pool if e not in eps.index]:
        raise KeyError(f"pool episodes not in the episode table: {missing[:10]}")
    succ = [e for e in pool if bool(eps.at[e, "success"])]
    fail = [e for e in pool if not bool(eps.at[e, "success"])]
    rng = np.random.default_rng(seed)

    def info(e: int, group: str, **extra) -> dict:
        return {
            "episode_index": e,
            "success": bool(eps.at[e, "success"]),
            "length": int(eps.at[e, "length"]),
            "group": group,
            **extra,
        }

    picked_fail = sorted(rng.choice(fail, size=min(n_fail, len(fail)), replace=False).tolist()) if fail else []

    vals = values[values["episode_index"].isin(succ)].sort_values(["episode_index", "frame_index"])
    dips: dict[int, tuple[float, int, int]] = {}
    for e, grp in vals.groupby("episode_index", sort=True):
        dips[int(e)] = v_dip_depth(grp["value"].to_numpy(), smooth)
    if missing := [e for e in succ if e not in dips]:
        raise KeyError(f"no values for successful pool episodes: {missing[:10]}")
    ranked = sorted(succ, key=lambda e: (-dips[e][0], e))
    picked_dip = ranked[: min(n_dip, len(ranked))]
    rest = [e for e in succ if e not in set(picked_dip)]
    picked_succ = sorted(rng.choice(rest, size=min(n_success, len(rest)), replace=False).tolist()) if rest else []

    def dip_info(e: int) -> dict:
        d, p, tr = dips[e]
        return {"dip_depth": round(d, 5), "dip_peak_frame": p, "dip_trough_frame": tr}

    out = [info(e, "failure_random") for e in picked_fail]
    out += [info(e, "success_deepest_dip", **dip_info(e)) for e in picked_dip]
    out += [info(e, "success_random", **dip_info(e)) for e in picked_succ]
    return out


def read_episode_list(path: Path | str) -> list[int]:
    """Episode ids from a `07_select_episodes.py` JSON (or a bare JSON list of ints)."""
    data = json.loads(Path(path).read_text())
    items = data["episodes"] if isinstance(data, dict) else data
    return [int(x["episode_index"]) if isinstance(x, dict) else int(x) for x in items]


def episode_success(episode: int, episode_table: Path | str = paths.EPISODE_TABLE) -> bool | None:
    try:
        df = pd.read_parquet(episode_table, columns=["episode_index", "success"])
    except FileNotFoundError:
        return None
    row = df.loc[df["episode_index"] == episode, "success"]
    return bool(row.iloc[0]) if len(row) else None
