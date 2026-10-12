# Step 03 — Value model
**Status:** done (laptop scope; full-fine-tune numbers on the 5090 pending)        **Agent/date:** Claude (step-3 agent), 2026-10-12

## Summary
- Built `src/recap_vf/model.py` (`ValueModelConfig`, `ValueModel`, `load_state_stats`, `tokenize_prompt`,
  `apply_attention_override`), `tests/test_model.py` (11 CPU tests, ~5 s) and `scripts/03_model_smoke.py`
  (real-backbone smoke: param counts, attention / embedding / SDPA checks, init loss, peak memory, overfit test).
- Real model: **681.4 M parameters** (SigLIP2 vision tower 412.4 M + Gemma-3-270M text model 268.1 M, of which the
  262k-token embedding is 167.8 M, + 0.88 M new). Trainable by default (token embedding frozen): **513.6 M**; with
  `freeze_vision=True`: 101.2 M. Sequence length **783** tokens (3 × 256 image + 13 prompt incl. `<bos>` + state + CLS).
- **Sliding window:** fixed by enlarging `sliding_window` to the sequence length, **not** by rewriting `layer_types` —
  in transformers 5.19 the layer type also selects the RoPE base (10 k local vs 1 M global), so rewriting it changes
  the pretrained network (max |Δh| = 271 on a 300-token input, vs exactly 0.0 for the window change).
- Init CE with the zero head = **5.3033 = ln 201** exactly, value = −0.5. Overfit test (frozen vision, 16 random
  pairs, AdamW 1e-4, 200 full-batch steps via 4×4 gradient accumulation): **PASSED, final loss 0.0004**, 16/16
  correct (loss < 0.1 from step ~75).
- Laptop (RTX 4060, only ~5 GiB actually free — the desktop holds ~2.5 GiB): batch 8 fwd+bwd peaks at
  **3.39 GiB** with frozen vision + gradient checkpointing (4.15 GiB incl. AdamW state); everything else at batch 8
  OOMs on this GPU. Full-fine-tune memory must be measured on the 5090 (pending).

## How to reproduce
```bash
# unit tests (CPU, tiny random backbones, no download)
env -u PYTHONPATH uv run pytest -q tests/test_model.py
env -u PYTHONPATH uv run ruff check src/recap_vf/model.py scripts/03_model_smoke.py tests/test_model.py

# real-backbone smoke on the laptop (downloads ~5.1 GB of backbone weights into the HF cache on first run:
# google/siglip2-so400m-patch14-224 model.safetensors is 4.5 GB because it contains the SigLIP text tower too)
env -u PYTHONPATH uv run python scripts/03_model_smoke.py --freeze-vision --overfit-gc --overfit-batch 4 --overfit-accum 4
#   -> reports/figures/step_03/smoke_NVIDIA_GeForce_RTX_4060_Laptop_GPU.json
# extra memory variants (batch 4; bf16 frozen vision tower at batch 8)
env -u PYTHONPATH uv run python scripts/03_model_smoke.py --skip-checks --skip-overfit --memory-batch 4 \
    --memory-variants frozen_vision,full,full+gc --out reports/figures/step_03/smoke_RTX4060_memory_b4.json
env -u PYTHONPATH uv run python scripts/03_model_smoke.py --skip-checks --skip-overfit --vision-bf16 --memory-batch 8 \
    --memory-variants frozen_vision,frozen_vision+gc --out reports/figures/step_03/smoke_RTX4060_memory_b8_vision_bf16.json

# workstation (pending): full fine-tune overfit + memory table
uv run python scripts/03_model_smoke.py --overfit-batch 16            # add --overfit-gc if it does not fit
```
Flags: `--freeze-vision`, `--overfit-gc`, `--overfit-batch` (micro-batch), `--overfit-accum`, `--overfit-steps`,
`--overfit-samples`, `--lr`, `--vision-bf16` (keep a *frozen* tower in bf16), `--memory-batch`,
`--memory-variants` (comma list of `frozen_vision|full` with optional `+gc`; OOMs are recorded, not fatal),
`--skip-checks/--skip-memory/--skip-overfit`, `--language-backbone`, `--vision-backbone`, `--stats`, `--seed`, `--out`.

## Results

### Usage
```python
from recap_vf.model import ValueModel, ValueModelConfig, load_state_stats
q01, q99 = load_state_stats(paths.RAW_DIR / "meta" / "stats.json")        # observation.state q01 / q99
model = ValueModel(ValueModelConfig(freeze_vision=False), q01, q99).cuda()
with torch.autocast("cuda", dtype=torch.bfloat16):
    logits = model(batch["images"], batch["state"])                         # float32 [B, 201]
loss = F.cross_entropy(logits, batch["target_bin"])
opt = torch.optim.AdamW(model.param_groups(lr_backbone=5e-5, lr_new=1e-4, weight_decay=0.01))
model.save_pretrained(ckpt_dir, extra={"train": train_cfg})                 # model.safetensors + config.yaml
model = ValueModel.from_pretrained(ckpt_dir)                                # no backbone download needed
out = model.predict(images, state)                                          # value, value_std, logits
```

### Parameter counts (real backbones)
| Component | Total | Trainable (default cfg) |
|---|---:|---:|
| SigLIP2-so400m vision tower (pooling head removed) | 412,442,352 | 412,442,352 |
| Gemma-3-270M text model (no `lm_head`) | 268,098,176 | 100,326,016 |
| ↳ token embedding (262 144 × 640, frozen by default) | 167,772,160 | 0 |
| New: projector RMSNorm+Linear, camera emb., state Linear, CLS, head | 880,073 | 880,073 |
| **All** | **681,420,601** | **513,648,441** |

With `freeze_vision=True` trainable = 101,206,089. ≈ 670 M as in the paper (681 M; the difference is SigLIP's
412 M vs the nominal "400 M").

### Checks on the real backbones (`attention_checks` in the smoke JSON)
| Check | Result |
|---|---|
| Gemma-3-270M `layer_types` | 5 × sliding + 1 × full, ×3 (layers 5, 11, 17 global); `sliding_window = 512` |
| RoPE per layer type (v5 `rope_parameters`) | sliding: θ = 10 000; full: θ = 1 000 000 |
| 300-token input, `sliding_window` 512 → 4096 | max \|Δ last_hidden_state\| = **0.0** (no-op below the window, as it must be) |
| 300-token input, all `layer_types` → `full_attention` | max \|Δ\| = **271.3** (mean \|h\| = 2.93) → breaks the pretrained model |
| 783-token input, ‖Δ h_last‖ when tokens 1–255 change | window 512: 40.7 (only via the 3 global layers) · enlarged: 68.3 |
| `embed_tokens(ids) / weight[ids]` | **25.2982 = √640** in fp32 (would be 25.25 with bf16 weights — the scale is cast to the weight dtype) |
| SDPA vs eager, last token, rel. L2 error vs fp32 eager | SDPA fp32 7e-7 · SDPA bf16 0.0073 · eager bf16 0.0132 |
| Backbone loading | 0 missing / 0 mismatched keys for both; SigLIP text tower + pooling head reported as *unexpected* (dropped) |

### Init loss
Random batch of 8 at bf16 autocast: logits float32 `[8, 201]`, all zero → CE = **5.3033** (ln 201 = 5.3033),
mean value **−0.5000**.

### Overfit test (16 fixed random (images, state, bin) pairs, frozen vision, laptop)
Micro-batch 4 × 4 accumulation (= full batch 16), AdamW lr 1e-4 on all trainable params (LM w/o token embedding
+ new modules), wd 0, bf16 autocast, gradient checkpointing, 200 steps, 709 s on the RTX 4060.

| step | 0 | 20 | 40 | 60 | 80 | 100 | 140 | 199 | final (eval mode, all 16) |
|---|---|---|---|---|---|---|---|---|---|
| loss | 5.303 | 2.532 | 1.832 | 0.272 | 0.0054 | 0.0015 | 0.0007 | 0.0004 | **0.00039**, top-1 16/16 |

**PASSED** (< 0.1). Peak memory 4.08 GiB. The first attempt (200 independent batch-4 steps) FAILED with 1.32 — see
"Overfit test and batch size" below.

### Peak GPU memory (`torch.cuda.max_memory_allocated`, bf16 autocast, fp32 master weights, RTX 4060 Laptop)
Only **~5 GiB** of the 8 GiB were free at start (desktop session holds ~2.5 GiB), so "OOM" below means "> ~5 GiB".
| Variant (vision tower dtype) | Batch | Weights | Peak fwd+bwd | Peak fwd+bwd+AdamW step |
|---|---:|---:|---:|---:|
| frozen vision, no checkpointing (fp32) | 8 | 2.55 GiB | OOM | — |
| frozen vision, no checkpointing (fp32) | 4 | 2.55 GiB | OOM | — |
| **frozen vision + grad. checkpointing (fp32)** | 8 | 2.55 GiB | **3.39 GiB** (1.8 s/step) | **4.15 GiB** |
| frozen vision, no checkpointing (bf16 tower) | 8 | 1.79 GiB | OOM | — |
| **frozen vision + grad. checkpointing (bf16 tower)** | 8 | 1.79 GiB | **2.63 GiB** | **3.40 GiB** |
| full fine-tune, no checkpointing | 8 | 2.56 GiB | OOM | — |
| full fine-tune, no checkpointing | 4 | 2.55 GiB | OOM | — |
| full fine-tune + grad. checkpointing | 8 | 2.56 GiB | OOM | — |
| full fine-tune + grad. checkpointing | 4 | 2.55 GiB | **4.64 GiB** (1.6 s/step) | OOM |

Without checkpointing the Gemma activations at 783 tokens (18 layers, MLP width 2048, fp32 RMSNorm upcasts) exceed
~2.5 GiB already at batch 4. Raw JSON: `reports/figures/step_03/smoke_*.json`.

5090 (32 GB): **pending** — run the workstation command above (`memory` + full fine-tune `overfit`).
Back-of-envelope for full fine-tune: 2.56 GiB fp32 weights + 513.6 M trainable × 12 B (grad + 2 Adam moments)
≈ 5.7 GiB → ~8.3 GiB before activations, so it will fit on the 5090 at a reasonable batch, with checkpointing for
large batches.

## Design notes / findings
- **transformers 5.19.0 / torch 2.11.0+cu128.** `SiglipVisionModel.from_pretrained` does not accept
  `vision_use_head=` as a kwarg in v5 (it is forwarded to `__init__` and crashes); we load
  `SiglipVisionConfig.from_pretrained(...)`, set `vision_use_head = False` on it and pass `config=`. The pooling head
  is then never built (no optimizer state). `SiglipVisionModel` picks the `vision_config` and `vision_model.*`
  weights out of the composite `siglip` checkpoint.
- **Language backbone:** default `ValueModelConfig.language_backbone = constants.LANGUAGE_BACKBONE_MIRROR`
  (`unsloth/gemma-3-270m`) because `google/gemma-3-270m` is gated for our HF login. The spec's interface lists
  `constants.LANGUAGE_BACKBONE`; switching back is a one-line config change once the licence is accepted.
- **Sliding window:** `apply_attention_override` sets `sliding_window = max(512, seq_len)` (= 783) when
  `full_attention=True`; `layer_types` are kept so every layer keeps the RoPE base it was trained with. With SDPA the
  sliding window only acts through the mask built by `create_sliding_window_causal_mask` (the `sliding_window` kwarg
  is ignored by `sdpa_attention_forward`), so the config change is all that is needed. Proven by
  `test_reach_first_camera` (tiny all-sliding LM, window 16: CLS changes when only the top camera changes with the
  override, bit-identical without it) and `test_override_keeps_layer_types`. Note that even without the override the
  real model is not completely blind to the top camera (3 global layers), but 15/18 layers would be.
- **Embedding scaling:** `Gemma3TextScaledWordEmbedding.forward` multiplies by `embed_scale = √hidden_size`;
  `Gemma3TextModel.forward(inputs_embeds=...)` applies no further scaling. So prompt embeddings are built with the
  LM's own `embed_tokens` (scaled), image / state / CLS embeddings are passed unscaled — identical to
  `Gemma3Model.forward`, which `masked_scatter`s unscaled projector outputs into scaled text embeddings.
- **Attention implementation:** SDPA for both towers. fp32 SDPA matches eager to 7e-7; bf16 SDPA is *closer* to the
  fp32 reference than bf16 eager. No NaNs in any run. No mask / position ids are passed (default causal 0…L−1).
- **Prompt:** tokenised once with the Gemma tokenizer (`<bos>` + 12 tokens, no chat template) and stored as a
  persistent buffer `prompt_ids`, together with `state_q01` / `state_q99`, so a checkpoint is self-contained
  (`from_pretrained` needs neither tokenizer nor backbone download).
- **Frozen vision** runs under `no_grad` (no activations kept). A frozen tower can additionally be cast to bf16
  (`model.vision_tower.to(torch.bfloat16)`, `--vision-bf16` in the smoke script); `encode_images` casts pixels to the
  tower's dtype.
- **Overfit test and batch size — important for step 4.** With per-step mini-batches of 4 at lr 1e-4 the loss
  stalls around ln 16 ≈ 2.7 (the model predicts the target marginal; first run: 1.32 after 200 steps, FAILED). A
  diagnostic on cached frozen-vision features (same maths) showed: batch 4 lr 1e-4 → 2.71; **full batch 16 lr 1e-4 →
  0.0005 (loss 0.009 by step 80)**; batch 4 with lr_backbone 3e-5 / lr_new 1e-3 → 0.26. I.e. the architecture
  memorises easily, but lr 1e-4 on the pretrained Gemma with tiny batches is too noisy. The smoke test therefore uses
  gradient accumulation to a full 16-sample batch (the spec's "batch ≤ 4" is respected per micro-batch). For step 4,
  use RLinf-like settings (backbone lr ≈ 5e-5, head lr 1e-4, large effective batch) rather than 1e-4 with small batches.
- **Checkpoint format:** `config.yaml` = `{model: ValueModelConfig, vision_config: …, text_config: <original, pre-
  override>, **extra}`; `save_pretrained(path, extra=...)` lets step 4 add the training config / returns tag / split
  required by the checkpoint contract. `from_pretrained` ignores unknown top-level keys.

## Deviations from the spec
1. `language_backbone` default is the ungated mirror `unsloth/gemma-3-270m` (identical weights), as instructed.
2. Sliding-window fix uses `sliding_window ≥ seq_len`, not `layer_types = full_attention` (the latter changes RoPE in
   transformers 5.x — see findings). The spec allowed either; only one is correct here.
3. Extra (backward-compatible) items: `ValueModelConfig.attn_implementation` (default `"sdpa"`) and `.prompt`;
   `ValueModel.__init__` has keyword-only `vision_config/text_config/prompt_ids` used by `from_backbone_configs` and
   `from_pretrained`; `save_pretrained(path, extra=None)`; `from_pretrained(path, **cfg_overrides)`;
   helpers `parameter_counts`, `build_inputs_embeds`, `encode_images`, `normalize_state`, `seq_len`.
4. `param_groups` returns up to four groups (`backbone_decay`, `backbone_no_decay`, `new_decay`, `new_no_decay`, each
   with a `name` key) — two lrs as specified, split by decay. No-decay = ndim < 2, names containing `norm` / `embed`,
   and `cls_token`.
5. Overfit test uses 4 × 4 gradient accumulation (full 16-sample batch) instead of 200 independent batch-4 steps,
   which does not reach < 0.1 at lr 1e-4 (documented above).
6. Full-fine-tune overfit and memory on the 5090 not run (no workstation access from this agent) — pending.

## Open issues / questions for the orchestrator
- **5090 run pending:** full fine-tune overfit + memory table (`uv run python scripts/03_model_smoke.py
  --overfit-batch 16`). Copy the JSON into `reports/figures/step_03/`.
- **Step 4 hyper-parameters:** lr 1e-4 on the pretrained LM with small batches did not memorise 16 samples; prefer
  backbone lr ≤ 5e-5 and a large effective batch (RLinf: 5e-5 / 1e-4, batch 256).
- **Laptop memory:** only frozen-vision + gradient checkpointing trains at batch 8 on the laptop (desktop takes
  ~2.5 GiB). Step-4 laptop smoke runs should use `freeze_vision=True, gradient_checkpointing=True`, batch ≤ 8.
- `docs/steps/03_model.md` interface shows `language_backbone = constants.LANGUAGE_BACKBONE`; consider updating it
  (and conventions) to say the mirror is the default until the licence is accepted.
- `docs/conventions.md` says backbones are ~2 GB on disk; the SigLIP2 checkpoint alone is 4.5 GB (fp32, includes the
  text tower) → ~5.1 GB total in the HF cache.

## Definition-of-done checklist
- [x] `ValueModel` implemented per interface; all tests pass on CPU in < 1 min (11 tests, ~5 s); ruff clean
- [x] Sliding-window override proven by the reach test
- [x] Smoke script: param counts, init loss ≈ 5.30, overfit test passes, memory numbers reported
      (laptop, frozen vision; full fine-tune on the 5090 **pending**)
- [x] Notes on transformers version, embedding scaling, attention implementation in the report
- [x] `reports/step_03_model.md` written
