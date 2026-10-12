"""Step 1.6 — visual sanity checks of the frame store.

1. For 3 success + 3 failure episodes: 3 cameras x {0, 25, 50, 75 %, last frame} -> ep_XXXX_grid.png
2. Video <-> state alignment: gripper state of one episode with the 3 sharpest gripper changes marked, and the
   matching wrist-camera frames around each change -> alignment_ep_XXXX.png

    env -u PYTHONPATH uv run python scripts/01_sanity_figures.py --episodes 0-9
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.signal import find_peaks  # noqa: E402

from recap_vf import paths  # noqa: E402
from recap_vf.constants import CAMERAS, FPS  # noqa: E402
from recap_vf.data.dataset import FrameDataset  # noqa: E402
from recap_vf.data.frames import parse_episodes  # noqa: E402

OUT = paths.FIGURES_DIR / "step_01"
OFFSETS = (-20, -10, 0, 10, 20)  # frames around a gripper event (0.2 s steps)


def _save(fig, path) -> None:
    """Save as a 256-colour palette PNG (~3x smaller than RGB; figures are committed to git)."""
    import io

    from PIL import Image

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90)
    plt.close(fig)
    Image.open(buf).convert("RGB").quantize(colors=256, method=Image.Quantize.MEDIANCUT).save(path, optimize=True)


def _hwc(img) -> np.ndarray:
    return img.permute(1, 2, 0).numpy()


def episode_grid(ds: FrameDataset, e: int, length: int, success: bool) -> None:
    frames = sorted({0, round(0.25 * (length - 1)), round(0.5 * (length - 1)), round(0.75 * (length - 1)), length - 1})
    ep = ds.get_episode(e, frames=frames)
    fig, axes = plt.subplots(len(CAMERAS), len(frames), figsize=(2.3 * len(frames), 2.45 * len(CAMERAS)), squeeze=False)
    for r, cam in enumerate(CAMERAS):
        for c, fi in enumerate(frames):
            ax = axes[r, c]
            ax.imshow(_hwc(ep["images"][cam][c]))
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"frame {fi} ({fi / FPS:.1f}s)", fontsize=9)
            if c == 0:
                ax.set_ylabel(cam, fontsize=9)
    fig.suptitle(f"episode {e} — {'SUCCESS' if success else 'FAILURE'} — {length} frames", fontsize=11)
    fig.tight_layout()
    _save(fig, OUT / f"ep_{e:04d}_grid.png")


def gripper_events(state: np.ndarray, grip_cols: dict[str, int], n: int = 3, min_sep: int = 50) -> list[tuple]:
    """The `n` sharpest gripper changes: (frame, side, signed speed per s), at least `min_sep` frames apart."""
    cands = []
    for side, col in grip_cols.items():
        g = pd.Series(state[:, col]).rolling(5, center=True, min_periods=1).mean().to_numpy()
        v = np.gradient(g) * FPS
        peaks, _ = find_peaks(np.abs(v), distance=min_sep)
        cands += [(int(p), side, float(v[p])) for p in peaks]
    cands.sort(key=lambda x: -abs(x[2]))
    chosen: list[tuple] = []
    for c in cands:
        if all(abs(c[0] - o[0]) >= min_sep for o in chosen):
            chosen.append(c)
        if len(chosen) == n:
            break
    return sorted(chosen)


def alignment_figure(ds: FrameDataset, e: int, grip_cols: dict[str, int]) -> list[tuple]:
    full = ds.get_episode(e, cameras=[])  # state only
    state = full["state"].numpy()
    length = len(state)
    events = gripper_events(state, grip_cols)
    fig = plt.figure(figsize=(2.6 * len(OFFSETS), 3.0 + 2.6 * len(events)))
    gs = fig.add_gridspec(1 + len(events), len(OFFSETS), height_ratios=[1.3] + [1] * len(events))
    ax = fig.add_subplot(gs[0, :])
    t = np.arange(length) / FPS
    for side, col in grip_cols.items():
        ax.plot(t, state[:, col], label=f"{side} gripper state", lw=1.2)
    for k, (fi, side, _) in enumerate(events):
        ax.axvline(fi / FPS, color="k", ls="--", lw=0.8)
        ax.text(fi / FPS, ax.get_ylim()[1], f" E{k + 1} ({side})", va="top", fontsize=8)
    ax.set_xlabel("time in episode [s]")
    ax.set_ylabel("gripper position")
    ax.legend(fontsize=8, loc="best")
    ax.set_title(f"episode {e}: gripper state vs wrist cameras (columns: event frame {list(OFFSETS)} frames)")
    for r, (fi, side, _) in enumerate(events, start=1):
        cam = f"{side}_wrist"
        frames = [int(np.clip(fi + o, 0, length - 1)) for o in OFFSETS]
        imgs = ds.get_episode(e, frames=frames, cameras=[cam])["images"][cam]
        for c, (f, img) in enumerate(zip(frames, imgs, strict=True)):
            a = fig.add_subplot(gs[r, c])
            a.imshow(_hwc(img))
            a.set_xticks([])
            a.set_yticks([])
            a.set_title(f"E{r} {cam} f{f}  g={state[f, grip_cols[side]]:.2f}", fontsize=7)
    fig.tight_layout()
    _save(fig, OUT / f"alignment_ep_{e:04d}.png")
    return events


TRAY_BOX = (90, 150, 224, 215)  # top camera crop (x0, y0, x1, y1) showing the tray's front row


def tray_figure(ds: FrameDataset, eps: list[int], episodes: pd.DataFrame) -> None:
    """Top-camera crop of the tray's front row at the first and last frame of each episode (count the actuators)."""
    fig, axes = plt.subplots(len(eps), 2, figsize=(6.4, 1.25 * len(eps)), squeeze=False)
    x0, y0, x1, y1 = TRAY_BOX
    for r, e in enumerate(eps):
        length = int(episodes.loc[e, "length"])
        imgs = ds.get_episode(e, frames=[0, length - 1], cameras=["top"])["images"]["top"]
        for c in range(2):
            ax = axes[r, c]
            ax.imshow(_hwc(imgs[c])[y0:y1, x0:x1])
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(
                    f"ep {e}\n{'S' if episodes.loc[e, 'success'] else 'F'}", fontsize=8, rotation=0, labelpad=14
                )
            if r == 0:
                ax.set_title(["first frame", "last frame"][c], fontsize=9)
    fig.suptitle("tray front row (top camera crop); S = success, F = failure", fontsize=9)
    fig.tight_layout()
    _save(fig, OUT / "tray_first_last.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", default="all", help="candidate episodes (must be in the store)")
    ap.add_argument("--n-per-class", type=int, default=3)
    ap.add_argument("--alignment-episode", type=int, default=None, help="default: first success episode")
    ap.add_argument("--seed", type=int, default=0, help="unused (deterministic choice); recorded for uniformity")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    episodes = pd.read_parquet(paths.EPISODE_TABLE).set_index("episode_index")
    cand = [e for e in parse_episodes(args.episodes, len(episodes)) if paths.episode_frames_path(e).is_file()]
    succ = [e for e in cand if episodes.loc[e, "success"]][: args.n_per_class]
    fail = [e for e in cand if not episodes.loc[e, "success"]][: args.n_per_class]
    names = json.loads((paths.RAW_DIR / "meta/info.json").read_text())["features"]["observation.state"]["names"]
    grip_cols = {"left": names.index("left_gripper.pos"), "right": names.index("right_gripper.pos")}

    ds = FrameDataset(succ + fail)
    for e in succ + fail:
        episode_grid(ds, e, int(episodes.loc[e, "length"]), bool(episodes.loc[e, "success"]))
        print(f"wrote {OUT / f'ep_{e:04d}_grid.png'}")
    tray_eps = cand[:10]
    tray_figure(FrameDataset(tray_eps), tray_eps, episodes)
    print(f"wrote {OUT / 'tray_first_last.png'}")
    ae = args.alignment_episode if args.alignment_episode is not None else succ[0]
    if ae not in ds.episodes:
        ds = FrameDataset([ae])
    events = alignment_figure(ds, ae, grip_cols)
    print(f"wrote {OUT / f'alignment_ep_{ae:04d}.png'}; events (frame, side, speed/s): {events}")


if __name__ == "__main__":
    main()
