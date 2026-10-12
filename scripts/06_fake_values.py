"""Development only: write a fake runs/<run>/values.parquet (contract schema) from the returns table.

Lets the step-6 labelling and the step-7 tooling run end-to-end before a value model exists. model_id = fake:<mode>.

Modes (g = return_norm, T = episode length, Z / C_fail from the returns JSON):
  oracle            V = g exactly (A = 0 everywhere -> degenerate threshold, by design)
  noisy             V = g + noise (default sigma 0.01; --noise-corr tau for AR(1)-correlated noise)
  lagged            successes V = g; failures look like a success, V = -(T-1-t)/Z, until `--lag` frames before the
                    end, then drop linearly over `--ramp` frames onto the true g (a VF that notices failures late)
  smoothed-outcome  V = w_t g + (1 - w_t) prior_t with w_t = (t/(T-1))^gamma and prior_t the outcome-averaged return
                    (success rate p_s): uncertain at the start, certain at the end
Every mode accepts --sigma / --noise-corr to add noise on top. Values are clipped to [-1, 0] like a real model.

    env -u PYTHONPATH uv run python scripts/06_fake_values.py --mode noisy --sigma 0.01 --run dev_fake_noisy_s0.01
"""

from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from recap_vf import paths
from recap_vf.advantage import VALUE_COLUMNS
from recap_vf.returns import load_returns

logger = logging.getLogger("06_fake_values")

MODES = ("oracle", "noisy", "lagged", "smoothed-outcome")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=MODES, required=True)
    p.add_argument("--run", default=None, help="default dev_fake_<mode>[_s<sigma>]")
    p.add_argument("--returns-tag", default="c2229")
    p.add_argument("--sigma", type=float, default=None, help="noise std (normalised units); noisy default 0.01")
    p.add_argument("--noise-corr", type=float, default=0.0, help="AR(1) correlation time in frames (0 = iid)")
    p.add_argument("--lag", type=int, default=100, help="lagged: frames before the end at which V reacts")
    p.add_argument("--ramp", type=int, default=25, help="lagged: frames over which V drops")
    p.add_argument("--gamma", type=float, default=1.0, help="smoothed-outcome: w_t = (t/(T-1))^gamma")
    p.add_argument("--max-episodes", type=int, default=None, help="only the first N episodes (smoke scale)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def ar1_noise(ep_bounds: np.ndarray, n: int, tau: float, rng: np.random.Generator) -> np.ndarray:
    """Unit-variance noise, iid (tau = 0) or stationary AR(1) with correlation time tau, restarted per episode."""
    xi = rng.standard_normal(n)
    if tau <= 0:
        return xi
    rho = float(np.exp(-1.0 / tau))
    s = np.sqrt(1 - rho**2)
    out = np.empty(n)
    for a, b in zip(ep_bounds[:-1], ep_bounds[1:], strict=True):
        y, _ = lfilter([s], [1.0, -rho], xi[a:b], zi=[rho * rng.standard_normal()])
        out[a:b] = y
    return out


def fake_values(returns: pd.DataFrame, meta: dict, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    """(value, value_std) float64 arrays aligned with `returns` (sorted by index)."""
    rng = np.random.default_rng(args.seed)
    z, c_fail = float(meta["Z"]), float(meta["c_fail"])
    g = returns["return_raw"].to_numpy(np.float64) / z
    t = returns["frame_index"].to_numpy(np.float64)
    T = returns["episode_length"].to_numpy(np.float64)
    succ = returns["success"].to_numpy(bool)
    remaining = (T - 1 - t) / z  # -g of a success with the same end time
    std = np.zeros(len(g))

    if args.mode in ("oracle", "noisy"):
        v = g.copy()
    elif args.mode == "lagged":
        v = g.copy()
        t_react = np.maximum(T - args.lag, 0)
        w = np.clip((t - t_react) / max(args.ramp, 1), 0.0, 1.0)  # 0 before the reaction, 1 after the ramp
        fail = ~succ
        v[fail] = (1 - w[fail]) * (-remaining[fail]) + w[fail] * g[fail]
    elif args.mode == "smoothed-outcome":
        p_s = float(returns.groupby("episode_index")["success"].first().mean())
        prior = -remaining - (1 - p_s) * c_fail / z
        w = (t / np.maximum(T - 1, 1)) ** args.gamma
        v = w * g + (1 - w) * prior
        std = (1 - w) * np.sqrt(p_s * (1 - p_s)) * c_fail / z  # std of the outcome mixture
    else:  # pragma: no cover - argparse restricts choices
        raise ValueError(args.mode)

    sigma = args.sigma if args.sigma is not None else (0.01 if args.mode == "noisy" else 0.0)
    args.sigma = sigma
    if sigma > 0:
        ep = returns["episode_index"].to_numpy()
        bounds = np.r_[0, np.flatnonzero(np.diff(ep)) + 1, len(ep)]
        v = v + sigma * ar1_noise(bounds, len(v), args.noise_corr, rng)
        std = np.sqrt(std**2 + sigma**2)
    return np.clip(v, -1.0, 0.0), std


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    returns, meta = load_returns(args.returns_tag)
    returns = returns.sort_values("index").reset_index(drop=True)
    if args.max_episodes is not None:
        keep = np.sort(returns["episode_index"].unique())[: args.max_episodes]
        returns = returns[returns["episode_index"].isin(keep)].reset_index(drop=True)

    value, std = fake_values(returns, meta, args)
    run = args.run or f"dev_fake_{args.mode}" + (f"_s{args.sigma:g}" if args.sigma else "")
    if not run.startswith("dev_fake_"):
        logger.warning("fake values written to a run not named dev_fake_*: %s", run)
    out_dir = paths.run_dir(run)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "index": returns["index"].to_numpy(np.int64),
            "episode_index": returns["episode_index"].to_numpy(np.int64),
            "frame_index": returns["frame_index"].to_numpy(np.int64),
            "value": value.astype(np.float32),
            "value_std": std.astype(np.float32),
            "model_id": f"fake:{args.mode}",
        }
    )[VALUE_COLUMNS]
    assert df["index"].is_unique
    df.to_parquet(out_dir / "values.parquet", index=False)
    info = {
        "fake": True,
        "mode": args.mode,
        "returns_tag": args.returns_tag,
        "Z": meta["Z"],
        "sigma": args.sigma,
        "noise_corr": args.noise_corr,
        "lag": args.lag,
        "ramp": args.ramp,
        "gamma": args.gamma,
        "seed": args.seed,
        "num_frames": len(df),
        "num_episodes": int(df["episode_index"].nunique()),
        "rmse_vs_return": float(np.sqrt(np.mean((value - returns["return_norm"].to_numpy(np.float64)) ** 2))),
    }
    (out_dir / "values.json").write_text(json.dumps(info, indent=2) + "\n")
    logger.info("wrote %s (%d frames): %s", out_dir / "values.parquet", len(df), info)


if __name__ == "__main__":
    main()
