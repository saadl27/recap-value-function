"""Step 3 smoke test with the real backbones: parameter counts, attention/embedding checks, init loss, memory, overfit.

The overfit test memorises 16 fixed random pairs with full-batch AdamW (lr 1e-4, 200 steps); on small GPUs the
16-sample batch is split into micro-batches with gradient accumulation (same maths). Plain batch-4 SGD steps do NOT
reach < 0.1 in 200 steps (mini-batch noise; see reports/step_03_model.md).

Laptop (8 GB):  env -u PYTHONPATH uv run python scripts/03_model_smoke.py --freeze-vision --overfit-gc \
                    --overfit-batch 4 --overfit-accum 4
Workstation:    uv run python scripts/03_model_smoke.py --overfit-batch 16   # full fine-tune (+ --overfit-gc)
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import time
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers.models.gemma3.modeling_gemma3 import Gemma3TextModel

from recap_vf import bins, constants, paths
from recap_vf.model import ValueModel, ValueModelConfig, load_state_stats

log = logging.getLogger("03_model_smoke")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stats", type=Path, default=paths.RAW_DIR / "meta" / "stats.json")
    p.add_argument("--language-backbone", default=constants.LANGUAGE_BACKBONE)
    p.add_argument("--vision-backbone", default=constants.VISION_BACKBONE)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--freeze-vision", action="store_true", help="overfit test with a frozen vision tower (8 GB GPUs)")
    p.add_argument("--overfit-gc", action="store_true", help="gradient checkpointing during the overfit test")
    p.add_argument("--vision-bf16", action="store_true", help="keep a frozen vision tower in bf16 (saves memory)")
    p.add_argument("--overfit-steps", type=int, default=200)
    p.add_argument("--overfit-batch", type=int, default=16, help="micro-batch size")
    p.add_argument(
        "--overfit-accum",
        type=int,
        default=1,
        help="gradient accumulation (laptop: --overfit-batch 4 --overfit-accum 4)",
    )
    p.add_argument("--overfit-samples", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--memory-batch", type=int, default=8)
    p.add_argument(
        "--memory-variants",
        default="frozen_vision,frozen_vision+gc,full,full+gc",
        help="comma list of {frozen_vision,full}[+gc]; OOMs are recorded, not fatal",
    )
    p.add_argument("--skip-checks", action="store_true")
    p.add_argument("--skip-memory", action="store_true")
    p.add_argument("--skip-overfit", action="store_true")
    p.add_argument(
        "--out", type=Path, default=None, help="results JSON (default reports/figures/step_03/smoke_<gpu>.json)"
    )
    return p.parse_args()


def random_batch(n: int, device: str, seed: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    images = {
        c: torch.randint(0, 256, (n, 3, constants.IMAGE_SIZE, constants.IMAGE_SIZE), dtype=torch.uint8, generator=g)
        for c in constants.CAMERAS
    }
    state = torch.randn(n, constants.STATE_DIM, generator=g)
    return {c: v.to(device) for c, v in images.items()}, state.to(device)


def free_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def gib(nbytes: int) -> float:
    return round(nbytes / 2**30, 2)


def autocast(device: str):
    return torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16)


# ----------------------------------------------------------------------------------------------------------------------
def attention_checks(args: argparse.Namespace) -> dict:
    """Why the override enlarges `sliding_window` instead of rewriting `layer_types`, measured on the real LM."""
    lm = Gemma3TextModel.from_pretrained(args.language_backbone, attn_implementation="sdpa", dtype=torch.float32)
    lm = lm.to(args.device).eval()
    cfg = lm.config
    g = torch.Generator().manual_seed(args.seed)
    out: dict = {"orig_sliding_window": cfg.sliding_window, "layer_types": list(cfg.layer_types)}
    out["rope_parameters"] = cfg.rope_parameters

    def run(ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return lm(input_ids=ids, use_cache=False).last_hidden_state.float()

    # (1) short sequence (< window): enlarging the window must be a no-op, rewriting layer_types must not be.
    short = torch.randint(0, cfg.vocab_size, (1, 300), generator=g).to(args.device)
    ref = run(short)
    orig_sw, orig_lt = cfg.sliding_window, list(cfg.layer_types)
    cfg.sliding_window = 4096
    out["short_seq_maxdiff_window_enlarged"] = (run(short) - ref).abs().max().item()
    cfg.sliding_window = orig_sw
    cfg.layer_types = ["full_attention"] * len(orig_lt)
    out["short_seq_maxdiff_layer_types_rewritten"] = (run(short) - ref).abs().max().item()
    out["short_seq_ref_absmean"] = ref.abs().mean().item()
    cfg.layer_types = orig_lt

    # (2) long sequence (783 = our length): last token's dependence on the first 256 tokens per setting.
    long = torch.randint(0, cfg.vocab_size, (1, 783), generator=g).to(args.device)
    pert = long.clone()
    pert[:, 1:256] = torch.randint(0, cfg.vocab_size, (1, 255), generator=g).to(args.device)
    for name, sw in (("window_512", orig_sw), ("window_enlarged", 4096)):
        cfg.sliding_window = sw
        out[f"long_seq_last_token_change_{name}"] = (run(long)[:, -1] - run(pert)[:, -1]).norm().item()
    cfg.sliding_window = orig_sw

    # (3) embedding scaling: embed_tokens multiplies by sqrt(hidden_size) (cast to the weight dtype).
    ids = torch.arange(10, 20, device=args.device)
    emb = lm.embed_tokens
    with torch.no_grad():
        ratio = (emb(ids) / emb.weight[ids]).float()
        out["embed_scale_fp32"] = ratio.median().item()
        out["embed_scale_bf16"] = (
            torch.tensor(math.sqrt(cfg.hidden_size)).to(torch.bfloat16).float().item()
        )  # what the module uses when weights are bf16
    out["sqrt_hidden_size"] = math.sqrt(cfg.hidden_size)

    # (4) SDPA vs eager, fp32 and bf16 autocast, last token of the long sequence; relative L2 error w.r.t. fp32 eager.
    lm.set_attn_implementation("eager")
    ref = run(long)[0, -1]
    for impl in ("eager", "sdpa"):
        lm.set_attn_implementation(impl)
        for prec in ("fp32", "bf16"):
            with torch.autocast(device_type=args.device.split(":")[0], dtype=torch.bfloat16, enabled=prec == "bf16"):
                h = run(long)[0, -1]
            out[f"rel_err_{impl}_{prec}_vs_eager_fp32"] = ((h - ref).norm() / ref.norm()).item()
    return out  # lm is freed by the caller's free_memory() once this frame is gone


def build(base: ValueModelConfig, q01, q99, device: str, vision_bf16: bool = False, **kw) -> ValueModel:
    model = ValueModel(replace(base, **kw), q01, q99)
    if vision_bf16 and model.cfg.freeze_vision:  # a frozen tower needs no fp32 master weights (-0.8 GB)
        model.vision_tower.to(torch.bfloat16)
    return model.to(device)


def init_loss_and_counts(base: ValueModelConfig, q01, q99, args: argparse.Namespace) -> dict:
    model = build(base, q01, q99, args.device)
    counts = model.parameter_counts()
    for k, v in counts.items():
        log.info("params %-28s total %12s  trainable %12s", k, f"{v['total']:,}", f"{v['trainable']:,}")
    images, state = random_batch(args.memory_batch, args.device, args.seed)
    targets = torch.randint(0, bins.NUM_BINS, (args.memory_batch,), generator=torch.Generator().manual_seed(1))
    with torch.no_grad(), autocast(args.device):
        logits = model(images, state)
    loss = F.cross_entropy(logits, targets.to(args.device)).item()
    log.info(
        "seq_len %d, logits %s %s, init CE %.4f (ln 201 = %.4f)",
        model.seq_len,
        tuple(logits.shape),
        logits.dtype,
        loss,
        math.log(bins.NUM_BINS),
    )
    res = {
        "param_counts": counts,
        "seq_len": model.seq_len,
        "prompt_tokens": int(model.prompt_ids.numel()),
        "logits_dtype": str(logits.dtype),
        "init_loss": loss,
        "init_value_mean": bins.expected_value(logits).mean().item(),
        "lm_sliding_window_after_override": model.language_model.config.sliding_window,
    }
    del model
    free_memory()
    return res


def memory_variant(base: ValueModelConfig, q01, q99, args: argparse.Namespace, variant: str) -> dict:
    """Peak allocated memory for one fwd+bwd (and fwd+bwd+AdamW step) at --memory-batch, bf16 autocast."""
    kind, _, flag = variant.partition("+")
    kw = {"freeze_vision": kind == "frozen_vision", "gradient_checkpointing": flag == "gc"}
    res: dict = {"variant": variant, "batch": args.memory_batch, "vision_bf16": args.vision_bf16, **kw}
    model = opt = None
    try:
        free_memory()
        model = build(base, q01, q99, args.device, vision_bf16=args.vision_bf16, **kw).train()
        with torch.no_grad():
            model.head.weight.normal_(std=0.01)  # non-zero head so the backward reaches the backbones
        res["weights_gib"] = gib(torch.cuda.memory_allocated())
        images, state = random_batch(args.memory_batch, args.device, args.seed)
        target = torch.randint(0, bins.NUM_BINS, (args.memory_batch,), device=args.device)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        with autocast(args.device):
            loss = F.cross_entropy(model(images, state), target)
        loss.backward()
        torch.cuda.synchronize()
        res["fwd_bwd_s"] = round(time.perf_counter() - t0, 3)
        res["peak_fwd_bwd_gib"] = gib(torch.cuda.max_memory_allocated())
        opt = torch.optim.AdamW(model.param_groups(args.lr, args.lr, 0.01), fused=True)
        opt.step()
        opt.zero_grad(set_to_none=True)
        with autocast(args.device):
            loss = F.cross_entropy(model(images, state), target)
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        res["peak_with_adamw_gib"] = gib(torch.cuda.max_memory_allocated())
    except torch.OutOfMemoryError as e:
        res["oom"] = str(e).split("\n")[0][:200]
    finally:
        del model, opt
        free_memory()
    log.info("memory %s", res)
    return res


def overfit(base: ValueModelConfig, q01, q99, args: argparse.Namespace) -> dict:
    """Memorise N fixed random (observation, bin) pairs: loss must fall below 0.1."""
    torch.manual_seed(args.seed)
    model = build(
        base,
        q01,
        q99,
        args.device,
        vision_bf16=args.vision_bf16,
        freeze_vision=args.freeze_vision,
        gradient_checkpointing=args.overfit_gc,
    ).train()
    n = args.overfit_samples
    images, state = random_batch(n, args.device, args.seed + 1)
    targets = torch.randint(0, bins.NUM_BINS, (n,), generator=torch.Generator().manual_seed(args.seed + 2))
    targets = targets.to(args.device)
    opt = torch.optim.AdamW(model.param_groups(args.lr, args.lr, 0.0), lr=args.lr)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    curve, t0 = [], time.perf_counter()
    g = torch.Generator().manual_seed(args.seed + 3)
    perm = torch.randperm(n, generator=g)
    pos = 0
    for step in range(args.overfit_steps):
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(args.overfit_accum):  # effective batch = overfit_batch * overfit_accum
            if pos + args.overfit_batch > n:
                perm, pos = torch.randperm(n, generator=g), 0
            idx = perm[pos : pos + args.overfit_batch].to(args.device)
            pos += args.overfit_batch
            with autocast(args.device):
                logits = model({c: v[idx] for c, v in images.items()}, state[idx])
            loss = F.cross_entropy(logits, targets[idx]) / args.overfit_accum
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}")
            loss.backward()
            step_loss += loss.item()
        opt.step()
        curve.append(step_loss)
        if step % 20 == 0 or step == args.overfit_steps - 1:
            log.info("overfit step %3d  loss %.4f", step, step_loss)
    elapsed = time.perf_counter() - t0
    # final loss on all N samples, eval mode, in chunks
    model.eval()
    losses, correct = [], 0
    with torch.no_grad(), autocast(args.device):
        for s in range(0, n, args.overfit_batch):
            sl = slice(s, s + args.overfit_batch)
            logits = model({c: v[sl] for c, v in images.items()}, state[sl])
            losses.append(F.cross_entropy(logits, targets[sl], reduction="sum").item())
            correct += (logits.argmax(-1) == targets[sl]).sum().item()
    final = sum(losses) / n
    res = {
        "freeze_vision": args.freeze_vision,
        "gradient_checkpointing": args.overfit_gc,
        "vision_bf16": args.vision_bf16,
        "steps": args.overfit_steps,
        "batch": args.overfit_batch,
        "accum": args.overfit_accum,
        "samples": n,
        "lr": args.lr,
        "first_loss": curve[0],
        "loss_at": {str(s): round(curve[s], 4) for s in range(0, len(curve), 20)},
        "final_train_loss": curve[-1],
        "final_eval_loss_all": final,
        "final_top1_acc": correct / n,
        "passed": final < 0.1,
        "seconds": round(elapsed, 1),
        "peak_gib": gib(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
    }
    log.info("overfit result %s", {k: v for k, v in res.items() if k != "loss_at"})
    del model, opt
    free_memory()
    return res


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.manual_seed(args.seed)
    import transformers

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    results: dict = {
        "seed": args.seed,
        "gpu": gpu,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "vision_backbone": args.vision_backbone,
        "language_backbone": args.language_backbone,
    }
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        results["gpu_free_at_start_gib"], results["gpu_total_gib"] = gib(free), gib(total)
    q01, q99 = load_state_stats(args.stats)
    base = ValueModelConfig(vision_backbone=args.vision_backbone, language_backbone=args.language_backbone)

    out = args.out or paths.FIGURES_DIR / "step_03" / f"smoke_{gpu.replace(' ', '_')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:  # after every stage, so a late OOM keeps the earlier numbers
        out.write_text(json.dumps(results, indent=2))

    if not args.skip_checks:
        results["attention_checks"] = attention_checks(args)
        free_memory()
        log.info("attention checks %s", results["attention_checks"])
        save()
    results.update(init_loss_and_counts(base, q01, q99, args))
    save()
    if not args.skip_memory and torch.cuda.is_available():
        results["memory"] = [memory_variant(base, q01, q99, args, v) for v in args.memory_variants.split(",") if v]
        save()
    if not args.skip_overfit:
        results["overfit"] = overfit(base, q01, q99, args)
        save()
    log.info("wrote %s", out)
    if "overfit" in results and not results["overfit"]["passed"]:
        raise SystemExit("overfit test FAILED")


if __name__ == "__main__":
    main()
