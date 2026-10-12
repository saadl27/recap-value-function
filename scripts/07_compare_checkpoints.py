"""Compare checkpoints on the same episodes: one mp4 per episode (3 cameras + one V(t) panel per checkpoint, shared
cursor) and one PNG per split with every checkpoint's V(t) overlaid.

Values come from `06_infer_values.py` runs, one per checkpoint, scored on the same episodes. With human interventions
(returns table with an `intervention` column) the panels shade human control and mark the takeover penalties.

    export RECAP_DATASET=actuator_unboxing_1h_HQ_dagger_rollouts
    uv run python scripts/07_compare_checkpoints.py --select 2 --print-episodes          # pick episodes, print them
    uv run python scripts/07_compare_checkpoints.py --episodes 3,17,40 --returns-tag i1060 \
        --out-dir runs/viz_dagger 250:viz_dagger_step250 500:viz_dagger_step500

`--select K`: K held-out and K training episodes per number of takeovers (0, 1, 2+), or per outcome when the dataset
has no interventions; fixed seed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from recap_vf import paths  # noqa: E402
from recap_vf.constants import CAMERAS, FPS, IMAGE_SIZE  # noqa: E402
from recap_vf.data.dataset import FrameDataset  # noqa: E402
from recap_vf.viz import Mp4Writer, compose_camera_strip, header_bar, load_episode_series, true_runs  # noqa: E402

COLORS = ["#2a7de1", "#e8a33d", "#c0392b", "#27ae60", "#8e44ad", "#16a085"]
HUMAN = "#f0a46a"
PANEL_H = 190
HEADER_H = 24


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoints", nargs="*", help="label:run pairs, e.g. 250:viz_dagger_step250")
    p.add_argument("--episodes", default=None, help="comma-separated episode list")
    p.add_argument("--select", type=int, default=None, help="pick K episodes per split and group (see above)")
    p.add_argument("--print-episodes", action="store_true", help="only print the selected episodes and exit")
    p.add_argument("--returns-tag", default="c2229")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def episode_groups(returns: pd.DataFrame) -> pd.Series:
    """Group label per episode: '0/1/2+ takeovers' with interventions, else 'success'/'failure'."""
    if "stretch_index" in returns.columns:
        n = returns.groupby("episode_index")["stretch_index"].max().clip(upper=2)
        return n.map({0: "0 takeovers", 1: "1 takeover", 2: "2+ takeovers"})
    return returns.groupby("episode_index")["success"].first().map({True: "success", False: "failure"})


def select_episodes(returns: pd.DataFrame, k: int, seed: int) -> dict[str, list[int]]:
    splits = json.loads(paths.SPLITS_FILE.read_text())
    groups = episode_groups(returns)
    rng = np.random.default_rng(seed)
    out = {}
    for split in ("heldout", "train"):
        chosen = []
        for g in sorted(groups.unique()):
            pool = sorted(e for e in splits[split] if groups.get(e) == g)
            chosen += sorted(int(e) for e in rng.choice(pool, size=min(k, len(pool)), replace=False))
        out[split] = chosen
    return out


class Panel:
    """Static V/g plot rendered once; `frame(k)` paints the cursor at frame k (fixed axes, so stacked cursors align)."""

    def __init__(self, v, g, human, penalty, width: int, title: str, color: str, xlabel: bool):
        self.n = len(v)
        t = np.arange(self.n) / FPS
        fig = plt.figure(figsize=(width / 100, PANEL_H / 100), dpi=100, facecolor="white")
        ax = fig.add_axes([0.075, 0.22 if xlabel else 0.14, 0.905, 0.62 if xlabel else 0.70])
        draw_series(ax, t, v, g, human, penalty, color)
        ax.set_title(title, loc="left", fontsize=9, pad=3, color=color, fontweight="bold")
        ax.legend(loc="upper left", fontsize=7, frameon=False, ncol=4, bbox_to_anchor=(0.0, 1.0))
        if xlabel:
            ax.set_xlabel("time [s]", fontsize=8)
        fig.canvas.draw()
        self.img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        H = self.img.shape[0]
        (self.x0, _), (self.x1, _) = ax.transData.transform([(0, 0), (t[-1], 0)])
        bb = ax.get_window_extent()
        self.ytop, self.ybot = int(H - bb.y1), int(H - bb.y0)
        self.yv = [int(round(H - ax.transData.transform((0, val))[1])) for val in v]
        self.color = np.array(matplotlib.colors.to_rgb(color)) * 255
        plt.close(fig)

    def frame(self, k: int) -> np.ndarray:
        img = self.img.copy()
        x = int(round(self.x0 + (self.x1 - self.x0) * k / max(self.n - 1, 1)))
        img[self.ytop : self.ybot, max(x - 1, 0) : x + 1] = 20
        y = self.yv[k]
        img[max(y - 4, 0) : y + 5, max(x - 4, 0) : x + 5] = self.color
        return img


def draw_series(ax, t, v, g, human, penalty, color, label="V(t)") -> None:
    for a, b in true_runs(human):
        ax.axvspan(t[a], t[b - 1] + 1 / FPS, color=HUMAN, alpha=0.25, lw=0)
    if human.any():
        ax.axvspan(0, 0, color=HUMAN, alpha=0.25, lw=0, label="human control")
    ax.plot(t, g, "--", color="0.35", lw=1.0, label="target g(t)")
    if v is not None:
        ax.plot(t, v, color=color, lw=1.3, label=label)
    if penalty.any():
        ax.plot(t[penalty], g[penalty], "v", color="#c0392b", ms=5, label="takeover penalty")
    ax.set_xlim(0, t[-1])
    ax.set_ylim(-1.02, 0.03)
    ax.set_yticks([-1, -0.75, -0.5, -0.25, 0])
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=7.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def main() -> None:
    args = parse_args()
    ret_path = paths.returns_paths(args.returns_tag)[0]
    returns = pd.read_parquet(ret_path)
    if args.select:
        sel = select_episodes(returns, args.select, args.seed)
        print(json.dumps(sel))
        print(",".join(str(e) for e in sel["heldout"] + sel["train"]))
        if args.print_episodes:
            return
    if args.print_episodes:
        return
    if not args.checkpoints or not (args.episodes or args.select) or args.out_dir is None:
        raise SystemExit("need label:run checkpoints, --episodes (or --select) and --out-dir")
    episodes = [int(e) for e in args.episodes.split(",")] if args.episodes else sel["heldout"] + sel["train"]
    ckpts = [(label, paths.run_dir(run)) for label, run in (a.split(":", 1) for a in args.checkpoints)]
    heldout = set(json.loads(paths.SPLITS_FILE.read_text())["heldout"])
    groups = episode_groups(returns)
    has_human = "intervention" in returns.columns
    penalty_value = -float(json.loads(paths.returns_paths(args.returns_tag)[1].read_text()).get("c_interv", np.inf))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    series = {}
    for e in episodes:
        ss = [load_episode_series(rd, e, advantages=None, returns_path=ret_path) for _, rd in ckpts]
        r = returns[returns["episode_index"] == e].sort_values("frame_index")
        human = r["intervention"].to_numpy(bool) if has_human else np.zeros(len(r), bool)
        penalty = r["reward"].to_numpy() <= penalty_value
        series[e] = (ss, ss[0]["return_norm"].to_numpy(float), human, penalty)

    # one PNG per split: all checkpoints overlaid
    for split in ("heldout", "train"):
        eps = [e for e in episodes if (e in heldout) == (split == "heldout")]
        if not eps:
            continue
        cols = 3
        rows = (len(eps) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 2.6 * rows), dpi=110, squeeze=False)
        for ax, e in zip(axes.ravel(), eps, strict=False):
            ss, g, human, penalty = series[e]
            t = np.arange(len(g)) / FPS
            draw_series(ax, t, None, g, human, penalty, "k")
            for i, ((label, _), s) in enumerate(zip(ckpts, ss, strict=True)):
                ax.plot(t, s["value"], color=COLORS[i % len(COLORS)], lw=1.0, label=f"V step {label}")
            ax.set_title(f"{split} ep {e} · {groups.get(e)}", fontsize=9)
        for ax in axes.ravel()[len(eps) :]:
            ax.axis("off")
        axes[0, 0].legend(fontsize=7, frameon=False, loc="lower right")
        fig.tight_layout()
        out = args.out_dir / f"v_curves_{split}_steps_{'_'.join(lb for lb, _ in ckpts)}.png"
        fig.savefig(out)
        plt.close(fig)
        print(f"wrote {out}")
    if args.no_video:
        return

    ds = FrameDataset(episodes)
    width = IMAGE_SIZE * len(CAMERAS)
    for e in episodes:
        ss, g, human, penalty = series[e]
        T = len(g)
        last = len(ckpts) - 1
        panels = [
            Panel(s["value"].to_numpy(float), g, human, penalty, width, f"checkpoint step {lb}", COLORS[i], i == last)
            for i, ((lb, _), s) in enumerate(zip(ckpts, ss, strict=True))
        ]
        split = "held-out" if e in heldout else "train"
        left = f"ep {e:04d} · {split} · {groups.get(e)}"
        sub = args.out_dir / ("heldout" if e in heldout else "train")
        sub.mkdir(exist_ok=True)
        out = sub / f"ep_{e:04d}.mp4"
        frames = np.arange(0, T, 2)  # every 2nd frame at 25 fps = real time
        height = HEADER_H + IMAGE_SIZE + PANEL_H * len(panels)
        with Mp4Writer(out, width, height, fps=25, crf=26) as w:
            for c0 in range(0, len(frames), 64):
                sel_frames = frames[c0 : c0 + 64]
                ep = ds.get_episode(e, frames=sel_frames)
                for j, t in enumerate(sel_frames):
                    cams = {c: ep["images"][c][j] for c in CAMERAS}
                    who = "HUMAN" if human[t] else "robot"
                    vals = "  ".join(f"V{lb} {s['value'].iloc[t]:+.2f}" for (lb, _), s in zip(ckpts, ss, strict=True))
                    right = f"t {t / FPS:5.2f} s  {who}  g {g[t]:+.2f}  {vals}"
                    rows = [header_bar(width, HEADER_H, left, right, size=12), compose_camera_strip(cams, CAMERAS)]
                    w.write(np.concatenate(rows + [p.frame(int(t)) for p in panels], axis=0))
        print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB, {len(frames)} frames)")


if __name__ == "__main__":
    main()
