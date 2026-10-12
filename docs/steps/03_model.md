# Step 3 — Value model

**Depends on:** step 0. **Parallel with:** steps 1 and 2 (develop on random tensors; no dataset needed except
`meta/stats.json` for state normalisation stats).
**Owns:** `src/recap_vf/model.py`, `scripts/03_model_smoke.py`, `tests/test_model.py`, `reports/step_03_model.md`.

## Goal
A PyTorch module that maps one timestep's observation (3 camera images + 14-D state + task prompt) to **logits over
201 value bins**, built like the paper's value function: a ~670M VLM initialised from **Gemma 3**.

## Architecture

```
 top (224²)      left_wrist (224²)   right_wrist (224²)
     │                 │                  │          uint8 → float, (x/255 − 0.5)/0.5
     ▼                 ▼                  ▼
 ┌──────────────── SigLIP2-so400m vision tower (shared weights, 27 layers, 1152-d) ───────────────┐
 └──── 256 patch tokens / camera (16×16 grid, patch 14) ─────────────────────────────────────────┘
     │ RMSNorm(1152) → Linear(1152→640)  (+ learned camera embedding, one per camera)
     ▼
 sequence = [img_top ×256][img_left ×256][img_right ×256][<bos> prompt tokens ×~15][state ×1][CLS ×1]  ≈ 785 tokens
     │
     ▼
 ┌──────────────── Gemma-3-270M decoder (18 layers, 640-d), causal, sliding window disabled ─────┐
 └────────────────────────────────────────────────────────────────────────────────────────────────┘
     │ final hidden state at CLS (after the model's final norm)
     ▼
 Dropout(p) → Linear(640 → 201) → logits         V = Σ softmax(logits)_b · v_b   (recap_vf.bins)
```

| Component | Source | Params |
|---|---|---|
| Vision tower | `google/siglip2-so400m-patch14-224` → `.vision_model` (config `model_type: siglip`, loads with `AutoModel`/`SiglipModel`) | ~400 M |
| LM | `google/gemma-3-270m` (`Gemma3ForCausalLM` → use the inner `Gemma3TextModel`; drop `lm_head`) | 268 M (of which ~168 M is the 262k-token embedding) |
| New | projector, camera embeddings, state projector, CLS, head | ~1 M |

≈ 670 M in total, matching the paper. RLinf uses the same pair of checkpoints.

### Details and why
- **Image normalisation:** SigLIP expects inputs in [−1, 1] (mean 0.5, std 0.5); confirm with the checkpoint's
  `preprocessor_config.json`. Do the uint8→float conversion **inside** the model on the GPU (dataloader ships uint8).
- **Use `last_hidden_state`** (256×1152) of the vision tower, not the pooled output. The SigLIP attention-pooling
  head is unused → load with `vision_use_head=False` if available, else delete it so it does not hold
  optimizer state.
- **Projector:** RMSNorm + Linear, mirroring Gemma-3's own multimodal projector. Init Linear with small std (0.02).
- **Camera embeddings:** a learned 640-d vector per camera added to that camera's tokens (cheap; helps the LM tell
  the two wrist views apart). Init to zeros.
- **Text:** tokenise `constants.PROMPT` once with the Gemma tokenizer (add `<bos>`, no chat template). Embed with the
  LM's own `embed_tokens`. ⚠ HF Gemma-3's `Gemma3TextScaledWordEmbedding` multiplies by `sqrt(hidden_size)` *inside*
  the embedding module; when you build `inputs_embeds` yourself, text embeddings come out scaled and image/state/CLS
  embeddings are passed **unscaled** — this matches what Gemma-3 multimodal does. Verify in your transformers version
  and note it in the report.
- **State token** (`state_mode="linear"`, default): normalise each dim to [−1, 1] with `q01`/`q99` from the dataset's
  `meta/stats.json` (`observation.state`), clip, then `Linear(14→640)`. Store the stats as **buffers** so checkpoints
  are self-contained. `state_mode="none"` drops the token (RLinf does not use state; the paper's observation does
  include it) — keep both for an ablation.
- **CLS token:** learned 640-d parameter appended last; its final hidden state is the value read-out. With causal
  attention the last token attends to everything.
- **Sliding window — important.** Gemma-3-270M has `sliding_window=512` on 15 of its 18 layers
  (`layer_types`: 5 × sliding, 1 × full, repeated). Our sequence is ~785 tokens, so in those layers the CLS token
  could **not** see the top camera. Fix at load time: set every entry of `config.layer_types` to `"full_attention"`
  (or `config.sliding_window ≥ 1024`) — whichever your transformers version honours; prove it with the test below.
  Cost is negligible at this length.
- **Attention implementation:** `attn_implementation="sdpa"` for both models (verify Gemma-3 trains fine with SDPA in
  bf16; fall back to `"eager"` if you see NaNs, and report).
- **Head init:** zero weights and bias → uniform distribution at init → initial loss = ln 201 ≈ 5.30.
- **Freezing knobs:** `freeze_vision`, `freeze_language`, `freeze_token_embeddings` (default **True**: the prompt is
  constant and the 168 M-param embedding would otherwise cost ~2 GB of Adam state for nothing).
- **Gradient checkpointing** knob for both towers (`gradient_checkpointing_enable()`).

## Interface (`src/recap_vf/model.py`)
```python
@dataclass
class ValueModelConfig:
    vision_backbone: str = constants.VISION_BACKBONE
    language_backbone: str = constants.LANGUAGE_BACKBONE_MIRROR   # identical weights, ungated
    cameras: tuple[str, ...] = constants.CAMERAS
    state_mode: Literal["linear", "none"] = "linear"
    full_attention: bool = True
    freeze_vision: bool = False
    freeze_language: bool = False
    freeze_token_embeddings: bool = True
    head_dropout: float = 0.0
    gradient_checkpointing: bool = False

class ValueModel(nn.Module):
    def __init__(self, cfg: ValueModelConfig, state_q01: Tensor, state_q99: Tensor): ...
    @classmethod
    def from_backbone_configs(cls, cfg, vision_config, text_config, ...) -> "ValueModel":  # random init, for tests
    def forward(self, images: dict[str, Tensor], state: Tensor) -> Tensor:        # logits [B, 201], float32
    @torch.no_grad()
    def predict(self, images, state) -> dict[str, Tensor]:                       # value, value_std, logits
    def param_groups(self, lr_backbone: float, lr_new: float, weight_decay: float) -> list[dict]
    def save_pretrained(self, path: Path) -> None                                # model.safetensors + config.yaml
    @classmethod
    def from_pretrained(cls, path: Path) -> "ValueModel"
```
- `forward` accepts `uint8 [B,3,224,224]` per camera (batch contract in [conventions](../conventions.md)).
- `param_groups`: pretrained weights in one group (lr_backbone), new modules (projector, camera/state embeddings,
  CLS, head) in another (lr_new); no weight decay on norms/biases/embeddings.
- Logits returned in **float32** even under bf16 autocast (cast before returning) so the loss is stable.

## Tasks
1. Implement `model.py` per the interface.
2. `tests/test_model.py` (CPU, tiny random configs via `from_backbone_configs` — e.g. SigLIP `hidden_size=32,
   num_hidden_layers=2, image_size=224, patch_size=14`; Gemma-3 text `hidden_size=32, num_hidden_layers=6,
   layer_types` with sliding layers and `sliding_window=16`, small vocab):
   - output shape `[B, 201]`, dtype float32; initial logits all zero → `expected_value == −0.5`;
   - **reach test:** with `full_attention=True`, perturbing only the first camera's pixels changes the CLS output of
     a model whose layers are all "sliding" in the original config; with `full_attention=False` and a window smaller
     than the sequence, it does not (proves the override works);
   - gradient reaches every trainable parameter; frozen parameters get no grad;
   - `save_pretrained` → `from_pretrained` round-trip gives identical outputs;
   - `state_mode="none"` works and has one fewer token.
3. `scripts/03_model_smoke.py` with the **real** backbones:
   - print parameter counts (total / trainable / per component) — expect ≈ 670 M total;
   - forward a random batch in bf16 autocast; CE loss vs random targets ≈ 5.30;
   - **overfit test** on 16 fixed random (image, target) pairs: 200 steps AdamW lr 1e-4 → loss < 0.1. On the 8 GB
     laptop, use `freeze_vision=True` and batch ≤ 4; on the 5090, run it full fine-tune too;
   - report peak GPU memory (`torch.cuda.max_memory_allocated`) for batch 8 forward+backward, with and without
     gradient checkpointing, on whatever GPU is available.

## Pitfalls
- Gemma-3 is gated; fall back to `constants.LANGUAGE_BACKBONE_MIRROR` and record it.
- Do not use the Gemma-3 *4B* multimodal class — 270M is text-only; we attach SigLIP ourselves.
- Keep the camera order of `constants.CAMERAS` everywhere.
- `position_ids`: let the LM build default positions 0…L−1 (all tokens are a single sequence, no padding since every
  sample has the same length). No attention mask needed — every sample has identical length.

## Definition of done
- [ ] `ValueModel` implemented per interface; all tests pass on CPU in < 1 min; ruff clean
- [ ] Sliding-window override proven by the reach test
- [ ] Smoke script: param counts, init loss ≈ 5.30, overfit test passes, memory numbers reported
- [ ] Notes on transformers version, embedding scaling, attention implementation in the report
- [ ] `reports/step_03_model.md` written
