"""Step 6: N-step advantages A_t and indicators I_t = 1[A_t >= eps] from a run's values.parquet.

    env -u PYTHONPATH uv run python scripts/06_compute_advantages.py --run vf_c2229_s0
    env -u PYTHONPATH uv run python scripts/06_compute_advantages.py --run vf_c2229_s0 \
        --segment-len 50 --smooth-values 5

Writes runs/<run>/advantages_{N}_{p}.parquet + .json for the default (N, p_pos) (and, with --write-grid, for every
setting of the sensitivity grid), runs/<run>/sensitivity.json (grid N x p_pos with label agreement vs the default),
optional segments_{N}_{p}.parquet, and diagnostics figures under reports/figures/step_06/<run>/.
With --smooth-values K the same outputs are produced a second time from median-smoothed V, with a `_smoothK` suffix.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from recap_vf import paths
from recap_vf.advantage import (
    ADVANTAGE_COLUMNS,
    VALUE_COLUMNS,
    compute_advantages,
    diagnostics,
    intervention_indices,
    label,
    per_episode_summary,
    segment_labels,
    smooth_values,
)
from recap_vf.constants import FPS
from recap_vf.returns import load_returns

logger = logging.getLogger("06_compute_advantages")

DEFAULT_CONFIG = paths.REPO_ROOT / "configs" / "advantage.yaml"
FIG_ROOT = paths.FIGURES_DIR / "step_06"
COLORS = {"success": "#2a78d6", "failure": "#eb6834", "positive": "#1baf7a", "ink": "#2b2b2b", "muted": "#8a8984"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="run directory name under runs/")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--values", type=Path, default=None, help="values parquet (default runs/<run>/values.parquet)")
    p.add_argument("--returns-tag", default=None)
    p.add_argument("--returns-dir", type=Path, default=None, help="directory of returns_{tag}.* (default data/cache)")
    p.add_argument(
        "--no-force-interventions",
        action="store_true",
        help="do not force human-intervention frames (returns column `intervention`) to I_t = True",
    )
    p.add_argument("--lookahead", type=int, default=None, help="N (frames)")
    p.add_argument("--p-pos", type=float, default=None)
    p.add_argument("--grid-lookaheads", type=int, nargs="*", default=None)
    p.add_argument("--grid-p-pos", type=float, nargs="*", default=None)
    p.add_argument("--no-grid", action="store_true", help="only the default setting")
    p.add_argument("--write-grid", action="store_true", help="write a parquet for every grid setting")
    p.add_argument("--segment-len", type=int, default=None, help="also write segments_{N}_{p}.parquet")
    p.add_argument("--smooth-values", type=int, default=None, help="flagged option: centred median over K frames")
    p.add_argument("--forced-positive", type=Path, default=None, help="JSON list of global indices forced True")
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--fig-dir", type=Path, default=None, help="default reports/figures/step_06/<run>")
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def resolve(args: argparse.Namespace) -> dict:
    cfg = yaml.safe_load(args.config.read_text()) if args.config.exists() else {}
    over = {
        "returns_tag": args.returns_tag,
        "lookahead": args.lookahead,
        "p_pos": args.p_pos,
        "grid_lookaheads": args.grid_lookaheads,
        "grid_p_pos": args.grid_p_pos,
        "segment_len": args.segment_len,
        "smooth_values": args.smooth_values,
        "forced_positive": str(args.forced_positive) if args.forced_positive else None,
        "returns_dir": str(args.returns_dir) if args.returns_dir else None,
        "seed": args.seed,
    }
    cfg.update({k: v for k, v in over.items() if v is not None})
    cfg.setdefault("returns_tag", "c2229")
    cfg.setdefault("lookahead", 50)
    cfg.setdefault("p_pos", 0.4)
    cfg.setdefault("tie_atol", 1e-6)
    cfg.setdefault("seed", 0)
    cfg.setdefault("force_interventions_positive", True)
    if args.no_force_interventions:
        cfg["force_interventions_positive"] = False
    cfg.setdefault("num_example_episodes", 2)
    if args.write_grid:
        cfg["write_grid"] = True
    if args.no_figures:
        cfg["figures"] = False
    if args.no_grid:
        cfg["grid_lookaheads"], cfg["grid_p_pos"] = [], []
    cfg["lookahead"], cfg["p_pos"] = int(cfg["lookahead"]), float(cfg["p_pos"])
    return cfg


def p_tag(p: float) -> str:
    return f"{p:g}"


# ---------------------------------------------------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------------------------------------------------
def _style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#e4e3df", lw=0.6)
    ax.set_axisbelow(True)


def plot_advantage_hist(labelled: pd.DataFrame, succ: np.ndarray, info: dict, title: str, out: Path) -> None:
    import matplotlib.pyplot as plt

    a = labelled["advantage"].to_numpy(np.float64)
    lo, hi = np.quantile(a, [0.001, 0.999])
    if hi - lo < 1e-5:  # degenerate (e.g. oracle values): show a tiny window around the data
        mid = 0.5 * (lo + hi)
        lo, hi = mid - 1e-5, mid + 1e-5
    edges = np.linspace(lo, hi, 121)
    fig, ax = plt.subplots(figsize=(7.5, 3.6), dpi=110)
    for name, mask in (("success", succ), ("failure", ~succ)):
        ax.hist(
            np.clip(a[mask], lo, hi),
            bins=edges,
            histtype="step",
            lw=2,
            color=COLORS[name],
            label=f"{name}-episode frames ({int(mask.sum()):,})",
        )
    eps = info["epsilon"]
    if np.isfinite(eps):
        ax.axvline(eps, color=COLORS["ink"], lw=1.2, ls="--")
        ax.annotate(
            f"ε = {eps:.4g}",
            (eps, 1),
            xycoords=("data", "axes fraction"),
            xytext=(4, -4),
            textcoords="offset points",
            va="top",
            fontsize=9,
        )
    ax.set_yscale("log")
    ax.set_xlabel("advantage A_t (normalised units; 1 frame of progress = 1/Z); clipped to 0.1–99.9 %")
    ax.set_ylabel("frames (log)")
    ax.set_title(title, fontsize=10)
    _style(ax)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_episode_pos_frac(per_ep: pd.DataFrame, p_pos: float, title: str, out: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 3.4), dpi=110)
    edges = np.linspace(0, 1, 21)
    for name, flag in (("success", True), ("failure", False)):
        pf = per_ep.loc[per_ep["success"] == flag, "positive_fraction"]
        ax.hist(
            pf,
            bins=edges,
            histtype="step",
            lw=2,
            color=COLORS[name],
            label=f"{name} episodes ({len(pf)}, mean {pf.mean():.0%})",
        )
    ax.axvline(p_pos, color=COLORS["muted"], lw=1, ls=":")
    ax.set_xlim(0, 1)
    ax.set_xlabel("fraction of the episode's frames with I_t = True")
    ax.set_ylabel("episodes")
    ax.set_title(title, fontsize=10)
    _style(ax)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def _shade_positive(ax, t: np.ndarray, ind: np.ndarray) -> None:
    if not ind.any():
        return
    edges = np.flatnonzero(np.diff(np.r_[0, ind.astype(np.int8), 0]))
    for s, e in zip(edges[::2], edges[1::2], strict=True):
        ax.axvspan(t[s], t[e - 1] + 1 / FPS, color=COLORS["positive"], alpha=0.18, lw=0)


def plot_examples(
    labelled: pd.DataFrame, returns: pd.DataFrame, episodes: list[int], info: dict, title: str, out: Path
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig = plt.figure(figsize=(9, 3.2 * len(episodes)), dpi=100)
    # rows per episode: V panel, A panel, spacer (keeps tick labels clear of the next episode's title)
    gs = fig.add_gridspec(3 * len(episodes), 1, height_ratios=[2, 1, 0.45] * len(episodes), hspace=0.15)
    g_by_index = returns.set_index("index")["return_norm"]
    for k, e in enumerate(episodes):
        sub = labelled[labelled["episode_index"] == e].sort_values("frame_index")
        t = sub["frame_index"].to_numpy() / FPS
        ind = sub["indicator"].to_numpy(bool)
        g = g_by_index.loc[sub["index"]].to_numpy()
        success = bool(returns.loc[returns["episode_index"] == e, "success"].iloc[0])
        ax_v = fig.add_subplot(gs[3 * k])
        ax_a = fig.add_subplot(gs[3 * k + 1], sharex=ax_v)
        for ax in (ax_v, ax_a):
            _shade_positive(ax, t, ind)
            _style(ax)
        ax_v.plot(t, sub["value"], color=COLORS["success" if success else "failure"], lw=1.6, label="V(t)")
        ax_v.plot(t, g, color=COLORS["muted"], lw=1.2, ls="--", label="g(t) true return")
        ax_v.set_ylabel("value")
        ax_v.set_title(
            f"episode {e} ({'success' if success else 'failure'}, {len(sub)} frames, {ind.mean():.0%} positive)",
            fontsize=9,
            loc="left",
        )
        ax_a.plot(t, sub["advantage"], color=COLORS["ink"], lw=1.0)
        if np.isfinite(info["epsilon"]):
            ax_a.axhline(info["epsilon"], color=COLORS["ink"], lw=0.8, ls=":")
        ax_a.set_ylabel("A(t)")
        plt.setp(ax_v.get_xticklabels(), visible=False)
        if k == 0:
            handles, labels = ax_v.get_legend_handles_labels()
            handles.append(Patch(color=COLORS["positive"], alpha=0.3))
            labels.append("I_t = True")
            ax_v.legend(handles, labels, frameon=False, fontsize=8, ncol=3, loc="lower right")
        ax_a.set_xlabel("time (s)", fontsize=8)
    fig.suptitle(title, fontsize=10, y=0.91)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def pick_examples(available: np.ndarray, success: pd.Series, n: int, seed: int) -> list[int]:
    """n success + n failure episodes, preferring the held-out split (fixed seed), else any available episode."""
    rng = np.random.default_rng(seed)
    heldout = np.array(json.loads(paths.SPLITS_FILE.read_text())["heldout"]) if paths.SPLITS_FILE.exists() else []
    chosen: list[int] = []
    for flag in (True, False):
        of_class = np.array([e for e in available if bool(success.loc[e]) == flag])
        cand = np.intersect1d(of_class, heldout)
        if len(cand) < n:
            cand = of_class
        if len(cand):
            chosen += sorted(rng.choice(cand, size=min(n, len(cand)), replace=False).tolist())
    return [int(e) for e in chosen]


# ---------------------------------------------------------------------------------------------------------------------
def write_labels(run_dir: Path, labelled: pd.DataFrame, meta: dict, n: int, p: float, suffix: str) -> tuple[Path, Path]:
    stem = f"advantages_{n}_{p_tag(p)}{suffix}"
    pq_path, json_path = run_dir / f"{stem}.parquet", run_dir / f"{stem}.json"
    labelled[ADVANTAGE_COLUMNS].to_parquet(pq_path, index=False)
    json_path.write_text(json.dumps(meta, indent=2, default=float) + "\n")
    return pq_path, json_path


def _portable(path: Path) -> str:
    """Path relative to the runs root / repo when possible, so the JSON is valid on both machines."""
    path = Path(path).resolve()
    for root, prefix in ((paths.RUNS_ROOT, "runs"), (paths.REPO_ROOT, "")):
        try:
            return str(Path(prefix) / path.relative_to(Path(root).resolve()))
        except ValueError:
            continue
    return str(path)


def markdown_table(rows: list[dict]) -> str:
    cols = [
        ("N", "{}"),
        ("p_pos", "{:g}"),
        ("epsilon", "{:.5f}"),
        ("positive_fraction", "{:.3f}"),
        ("pos_frac_success", "{:.3f}"),
        ("pos_frac_failure", "{:.3f}"),
        ("agreement_vs_default", "{:.3f}"),
        ("mean_run_length", "{:.1f}"),
        ("degenerate", "{}"),
    ]
    lines = ["| " + " | ".join(c for c, _ in cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(fmt.format(r[c]) for c, fmt in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    cfg = resolve(args)
    run_dir = paths.run_dir(args.run)
    values_path = args.values or run_dir / "values.parquet"
    run_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = args.fig_dir or FIG_ROOT / args.run

    values = pd.read_parquet(values_path)
    if missing := [c for c in VALUE_COLUMNS if c not in values.columns]:
        raise KeyError(f"{values_path} lacks contract columns {missing}")
    returns, rmeta = load_returns(cfg["returns_tag"], cache_dir=cfg.get("returns_dir"))
    z = float(rmeta["Z"])
    success = returns.groupby("episode_index")["success"].first()
    n_eps = values["episode_index"].nunique()
    if len(values) != len(returns):
        logger.warning(
            "values cover %d frames of %d episodes (returns table: %d frames)", len(values), n_eps, len(returns)
        )
    model_ids = sorted(values["model_id"].astype(str).unique().tolist())
    logger.info(
        "values %s: %d frames, %d episodes, model_ids %s; returns %s, Z = %g",
        values_path,
        len(values),
        n_eps,
        model_ids[:6],
        cfg["returns_tag"],
        z,
    )

    forced: set[int] = set()
    if cfg.get("forced_positive"):
        forced |= {int(i) for i in json.loads(Path(cfg["forced_positive"]).read_text())}
    if cfg["force_interventions_positive"] and (human := intervention_indices(returns)):
        forced |= human & set(values["index"].astype(np.int64).tolist())
        logger.info("human-intervention frames forced positive (RECAP): %d", len(human))
    forced = forced or None
    if forced:
        logger.info("%d forced-positive frames in total", len(forced))

    n0, p0 = cfg["lookahead"], cfg["p_pos"]
    grid_n = sorted({n0, *map(int, cfg.get("grid_lookaheads") or [])})
    grid_p = sorted({p0, *map(float, cfg.get("grid_p_pos") or [])})
    variants = [("", values)]
    if cfg.get("smooth_values"):
        k = int(cfg["smooth_values"])
        variants.append((f"_smooth{k}", smooth_values(values, k)))

    values_ref = _portable(values_path)
    summary: dict = {"run": args.run, "values_file": values_ref, "model_ids": model_ids, "config": cfg}
    default_ind: dict[str, np.ndarray] = {}
    for suffix, vals in variants:
        smooth = int(suffix.removeprefix("_smooth")) if suffix else None
        rows, default = [], None
        for n in grid_n:
            adv = compute_advantages(vals, returns, n, z=z)
            for p in grid_p:
                labelled, info = label(adv, p, forced, tie_atol=float(cfg["tie_atol"]))
                diag = diagnostics(labelled, success)
                ind = labelled["indicator"].to_numpy(bool)
                is_default = n == n0 and p == p0
                meta = {
                    "N": n,
                    "p_pos": p,
                    "epsilon": info["epsilon"],
                    "returns_tag": cfg["returns_tag"],
                    "Z": z,
                    "c_fail": rmeta.get("c_fail"),
                    "c_interv": rmeta.get("c_interv"),
                    "num_forced_positive": len(forced or ()),
                    "values_file": values_ref,
                    "model_ids": model_ids,
                    "num_frames": int(len(labelled)),
                    "num_episodes": int(n_eps),
                    "positive_fraction": info["positive_fraction"],
                    "positive_fraction_success_frames": diag["positive_fraction_success_frames"],
                    "positive_fraction_failure_frames": diag["positive_fraction_failure_frames"],
                    "smooth_values": smooth,
                    "forced_positive_file": cfg.get("forced_positive"),
                    "threshold": info,
                    "diagnostics": diag,
                    "seed": cfg["seed"],
                    "created_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
                }
                if is_default or cfg.get("write_grid"):
                    pq_path, _ = write_labels(run_dir, labelled, meta, n, p, suffix)
                    logger.info("wrote %s", pq_path)
                if is_default:
                    default = (labelled, info, diag, ind)
                    if cfg.get("segment_len"):
                        seg = segment_labels(labelled, int(cfg["segment_len"]))
                        seg_path = run_dir / f"segments_{n}_{p_tag(p)}{suffix}.parquet"
                        seg.to_parquet(seg_path, index=False)
                        logger.info(
                            "wrote %s (%d segments, %.1f%% positive)", seg_path, len(seg), 100 * seg["indicator"].mean()
                        )
                rows.append(
                    {
                        "N": n,
                        "p_pos": p,
                        "epsilon": info["epsilon"],
                        "positive_fraction": info["positive_fraction"],
                        "pos_frac_success": diag["positive_fraction_success_frames"],
                        "pos_frac_failure": diag["positive_fraction_failure_frames"],
                        "mean_run_length": diag["mean_run_length"],
                        "degenerate": info["degenerate"],
                        "_ind": ind,
                    }
                )
        assert default is not None
        labelled, info, diag, ind0 = default
        default_ind[suffix] = ind0
        for r in rows:
            r["agreement_vs_default"] = float((r.pop("_ind") == ind0).mean())
        table = markdown_table(rows)
        logger.info("sensitivity%s (default N=%d, p=%g):\n%s", suffix, n0, p0, table)
        for w in info["warnings"]:
            logger.warning("default labels%s: %s", suffix, w)
        logger.info("default diagnostics%s:\n%s", suffix, json.dumps(diag, indent=2))
        (run_dir / f"sensitivity{suffix}.json").write_text(json.dumps(rows, indent=2) + "\n")
        summary[f"default{suffix}"] = {"threshold": info, "diagnostics": diag}
        summary[f"sensitivity{suffix}"] = rows
        summary[f"sensitivity_table{suffix}"] = table

        if cfg.get("figures", True):
            fig_dir.mkdir(parents=True, exist_ok=True)
            succ_frame = success.reindex(labelled["episode_index"]).to_numpy(bool)
            name = f"{args.run}{suffix}: N = {n0}, p_pos = {p0:g}"
            plot_advantage_hist(
                labelled, succ_frame, info, f"{name}, ε = {info['epsilon']:.4g}", fig_dir / f"adv_hist{suffix}.png"
            )
            per_ep = per_episode_summary(labelled, success)
            plot_episode_pos_frac(
                per_ep, p0, f"{name}: per-episode positive fraction", fig_dir / f"episode_pos_frac{suffix}.png"
            )
            examples = pick_examples(
                np.unique(labelled["episode_index"]), success, int(cfg["num_example_episodes"]), int(cfg["seed"])
            )
            plot_examples(
                labelled, returns, examples, info, f"{name} (shaded: I_t = True)", fig_dir / f"examples{suffix}.png"
            )
            summary[f"example_episodes{suffix}"] = examples
            logger.info("figures written to %s (examples: %s)", fig_dir, examples)

    if len(default_ind) > 1:
        a, b = default_ind.values()
        summary["agreement_raw_vs_smoothed"] = float((a == b).mean())
        logger.info("default-label agreement raw vs smoothed: %.3f", summary["agreement_raw_vs_smoothed"])
    if cfg.get("figures", True):
        (fig_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")


if __name__ == "__main__":
    main()
