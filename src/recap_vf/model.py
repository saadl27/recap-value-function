"""Distributional value model: SigLIP2-so400m vision tower + Gemma-3-270M decoder -> 201 value-bin logits.

One observation = 3 camera images + 14-D joint state + the task prompt. Each camera is encoded by the shared SigLIP
tower into 256 patch tokens, projected to the LM width, and concatenated with the prompt, a state token and a learned
CLS token:

    [top x256][left_wrist x256][right_wrist x256][<bos> prompt][state][CLS]

The Gemma decoder runs causally over this sequence; the final (normed) hidden state at CLS feeds a linear head with
`bins.NUM_BINS` outputs. See docs/steps/03_model.md for the design rationale.
"""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Literal

import torch
import yaml
from safetensors.torch import load_file, save_file
from torch import Tensor, nn
from transformers import Gemma3TextConfig, SiglipVisionConfig
from transformers.models.gemma3.modeling_gemma3 import Gemma3RMSNorm, Gemma3TextModel
from transformers.models.siglip.modeling_siglip import SiglipVisionModel

from recap_vf import bins, constants

log = logging.getLogger(__name__)

# SigLIP preprocessing (preprocessor_config.json of google/siglip2-so400m-patch14-224): rescale 1/255, then
# normalise with mean = std = 0.5, i.e. pixels in [-1, 1].
IMAGE_MEAN = 0.5
IMAGE_STD = 0.5

WEIGHTS_NAME = "model.safetensors"
CONFIG_NAME = "config.yaml"

# Names of the modules created by this class (everything else is a pretrained backbone).
NEW_MODULES = ("projector_norm", "projector", "camera_embed", "state_proj", "cls_token", "head")


@dataclass
class ValueModelConfig:
    vision_backbone: str = constants.VISION_BACKBONE
    # Gated: needs the Gemma licence accepted + `hf auth login`. Without access, pass
    # constants.LANGUAGE_BACKBONE_MIRROR (unsloth/gemma-3-270m, byte-identical model.safetensors).
    language_backbone: str = constants.LANGUAGE_BACKBONE
    cameras: tuple[str, ...] = constants.CAMERAS
    state_mode: Literal["linear", "none"] = "linear"
    full_attention: bool = True
    freeze_vision: bool = False
    freeze_language: bool = False
    freeze_token_embeddings: bool = True
    head_dropout: float = 0.0
    gradient_checkpointing: bool = False
    attn_implementation: str = "sdpa"
    prompt: str = constants.PROMPT

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ValueModelConfig:
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in d.items() if k in known}
        if "cameras" in kwargs:
            kwargs["cameras"] = tuple(kwargs["cameras"])
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["cameras"] = list(self.cameras)
        return d


def load_state_stats(stats_path: Path, key: str = "observation.state") -> tuple[Tensor, Tensor]:
    """(q01, q99) float32 tensors from a LeRobot `meta/stats.json`, used to normalise the state token."""
    stats = json.loads(Path(stats_path).read_text())[key]
    return torch.tensor(stats["q01"], dtype=torch.float32), torch.tensor(stats["q99"], dtype=torch.float32)


def tokenize_prompt(language_backbone: str, prompt: str = constants.PROMPT) -> Tensor:
    """Prompt token ids (int64 [L]) with a leading <bos>, no chat template."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(language_backbone)
    ids = tok(prompt, add_special_tokens=False)["input_ids"]
    return torch.tensor([tok.bos_token_id, *ids], dtype=torch.long)


def apply_attention_override(text_config: Gemma3TextConfig, full_attention: bool, min_window: int) -> None:
    """Make every Gemma layer see the whole sequence, in place.

    Gemma-3-270M has `sliding_window=512` on 15/18 layers; our ~785-token sequence would hide the top camera from CLS
    in those layers. We do NOT rewrite `layer_types` to "full_attention": in transformers 5.x the layer type also
    selects the RoPE base (10k local vs 1M global), so that would silently change the pretrained positional encoding
    of 15 layers. Enlarging `sliding_window` keeps the RoPE bases and removes the window (mask == plain causal).
    """
    if full_attention and text_config.sliding_window is not None:
        text_config.sliding_window = max(int(text_config.sliding_window), int(min_window))


def _check_loading(repo: str, info: dict[str, Any]) -> None:
    """Fail loudly if a backbone checkpoint did not cover every weight of our module."""
    bad = {k: info.get(k) for k in ("missing_keys", "mismatched_keys", "error_msgs") if info.get(k)}
    if bad:
        raise RuntimeError(f"incomplete weight loading from {repo}: {bad}")


def _jsonable(d: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(d, default=str))


class ValueModel(nn.Module):
    """observation (images, state) -> logits [B, NUM_BINS] (float32)."""

    def __init__(
        self,
        cfg: ValueModelConfig,
        state_q01: Tensor,
        state_q99: Tensor,
        *,
        vision_config: SiglipVisionConfig | None = None,
        text_config: Gemma3TextConfig | None = None,
        prompt_ids: Tensor | None = None,
    ):
        """Load pretrained backbones from `cfg` (default), or build randomly-initialised ones from the given configs.

        Prefer `from_backbone_configs` (random init) and `from_pretrained` (checkpoint) over the keyword arguments.
        """
        super().__init__()
        self.cfg = cfg
        random_init = vision_config is not None or text_config is not None
        if random_init and (vision_config is None or text_config is None or prompt_ids is None):
            raise ValueError("random init needs vision_config, text_config and prompt_ids")

        # --- vision tower -------------------------------------------------------------------------------------
        if random_init:
            vision_config = copy.deepcopy(vision_config)
            vision_config.vision_use_head = False
            self.vision_tower = SiglipVisionModel._from_config(
                vision_config, attn_implementation=cfg.attn_implementation
            )
        else:
            # The checkpoint is a full SiglipModel (config model_type "siglip"); SiglipVisionModel picks its
            # vision_config and vision_model.* weights; the text tower and the pooling head are dropped (unexpected).
            vision_config = SiglipVisionConfig.from_pretrained(cfg.vision_backbone)
            vision_config.vision_use_head = False
            self.vision_tower, info = SiglipVisionModel.from_pretrained(
                cfg.vision_backbone,
                config=vision_config,
                attn_implementation=cfg.attn_implementation,
                dtype=torch.float32,
                output_loading_info=True,
            )
            _check_loading(cfg.vision_backbone, info)
        vcfg = self.vision_tower.config
        if getattr(self.vision_tower, "head", None) is not None:  # older transformers ignore vision_use_head
            del self.vision_tower.head
            self.vision_tower.use_head = False
        self.tokens_per_camera = (vcfg.image_size // vcfg.patch_size) ** 2
        self.image_size = vcfg.image_size

        # --- prompt -------------------------------------------------------------------------------------------
        if prompt_ids is None:
            prompt_ids = tokenize_prompt(cfg.language_backbone, cfg.prompt)
        self.register_buffer("prompt_ids", prompt_ids.long().clone(), persistent=True)

        n_state = 1 if cfg.state_mode == "linear" else 0
        self.seq_len = len(cfg.cameras) * self.tokens_per_camera + len(prompt_ids) + n_state + 1

        # --- language model -----------------------------------------------------------------------------------
        if random_init:
            text_config = copy.deepcopy(text_config)
            self._text_config_src = copy.deepcopy(text_config)
            apply_attention_override(text_config, cfg.full_attention, self.seq_len)
            self.language_model = Gemma3TextModel._from_config(text_config, attn_implementation=cfg.attn_implementation)
        else:
            text_config = Gemma3TextConfig.from_pretrained(cfg.language_backbone)
            self._text_config_src = copy.deepcopy(text_config)
            apply_attention_override(text_config, cfg.full_attention, self.seq_len)
            self.language_model, info = Gemma3TextModel.from_pretrained(
                cfg.language_backbone,
                config=text_config,
                attn_implementation=cfg.attn_implementation,
                dtype=torch.float32,
                output_loading_info=True,
            )
            _check_loading(cfg.language_backbone, info)
        d_model = self.language_model.config.hidden_size

        # --- new modules --------------------------------------------------------------------------------------
        self.projector_norm = Gemma3RMSNorm(vcfg.hidden_size, eps=vcfg.layer_norm_eps)  # (1 + w), w init 0
        self.projector = nn.Linear(vcfg.hidden_size, d_model)
        self.camera_embed = nn.Parameter(torch.zeros(len(cfg.cameras), d_model))
        if cfg.state_mode == "linear":
            self.state_proj = nn.Linear(constants.STATE_DIM, d_model)
        elif cfg.state_mode == "none":
            self.state_proj = None
        else:
            raise ValueError(f"unknown state_mode {cfg.state_mode!r}")
        self.cls_token = nn.Parameter(torch.zeros(d_model))
        self.head_dropout = nn.Dropout(cfg.head_dropout)
        self.head = nn.Linear(d_model, bins.NUM_BINS)
        self._init_new_modules()

        self.register_buffer("state_q01", state_q01.float().clone(), persistent=True)
        self.register_buffer("state_q99", state_q99.float().clone(), persistent=True)

        self._apply_freezing()
        if cfg.gradient_checkpointing:
            kwargs = {"use_reentrant": False}
            self.vision_tower.gradient_checkpointing_enable(gradient_checkpointing_kwargs=kwargs)
            self.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=kwargs)

    # ------------------------------------------------------------------------------------------------------------
    def _init_new_modules(self) -> None:
        nn.init.normal_(self.projector.weight, std=0.02)
        nn.init.zeros_(self.projector.bias)
        if self.state_proj is not None:
            nn.init.normal_(self.state_proj.weight, std=0.02)
            nn.init.zeros_(self.state_proj.bias)
        nn.init.normal_(self.cls_token, std=0.02)
        # zero head -> uniform distribution -> initial CE = ln(201), initial value = -0.5
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def _apply_freezing(self) -> None:
        cfg = self.cfg
        self.vision_tower.requires_grad_(not cfg.freeze_vision)
        self.language_model.requires_grad_(not cfg.freeze_language)
        if cfg.freeze_token_embeddings:
            self.language_model.embed_tokens.weight.requires_grad_(False)

    @classmethod
    def from_backbone_configs(
        cls,
        cfg: ValueModelConfig,
        vision_config: SiglipVisionConfig,
        text_config: Gemma3TextConfig,
        state_q01: Tensor | None = None,
        state_q99: Tensor | None = None,
        prompt_ids: Tensor | None = None,
    ) -> ValueModel:
        """Randomly-initialised model from backbone configs (tests, checkpoint loading). No download."""
        if state_q01 is None:
            state_q01 = -torch.ones(constants.STATE_DIM)
        if state_q99 is None:
            state_q99 = torch.ones(constants.STATE_DIM)
        if prompt_ids is None:
            n = min(text_config.vocab_size, 16)
            prompt_ids = torch.arange(n, dtype=torch.long) % text_config.vocab_size
        return cls(
            cfg, state_q01, state_q99, vision_config=vision_config, text_config=text_config, prompt_ids=prompt_ids
        )

    # ------------------------------------------------------------------------------------------------------------
    @property
    def num_state_tokens(self) -> int:
        return 0 if self.state_proj is None else 1

    def _vision_trainable(self) -> bool:
        return any(p.requires_grad for p in self.vision_tower.parameters())

    def normalize_state(self, state: Tensor) -> Tensor:
        """Map raw joint positions to [-1, 1] with the dataset's q01/q99, clipped."""
        span = (self.state_q99 - self.state_q01).clamp_min(1e-6)
        return (2 * (state.float() - self.state_q01) / span - 1).clamp(-1, 1)

    def encode_images(self, images: dict[str, Tensor]) -> Tensor:
        """uint8 [B,3,H,W] per camera -> projected tokens [B, n_cams * 256, d_model] (camera embedding added)."""
        pix = torch.stack([images[c] for c in self.cfg.cameras], dim=1)  # [B, C, 3, H, W]
        b, n_cam = pix.shape[:2]
        dtype = self.vision_tower.embeddings.patch_embedding.weight.dtype
        pix = pix.flatten(0, 1).to(dtype=torch.float32)
        pix = ((pix / 255.0 - IMAGE_MEAN) / IMAGE_STD).to(dtype)
        with torch.set_grad_enabled(torch.is_grad_enabled() and self._vision_trainable()):
            feats = self.vision_tower(pixel_values=pix).last_hidden_state  # [B*C, 256, 1152]
        tokens = self.projector(self.projector_norm(feats))  # [B*C, 256, d]
        tokens = tokens.unflatten(0, (b, n_cam)) + self.camera_embed[None, :, None, :].to(tokens.dtype)
        return tokens.flatten(1, 2)

    def build_inputs_embeds(self, images: dict[str, Tensor], state: Tensor) -> Tensor:
        """Sequence [images][<bos> prompt][state][CLS] -> [B, seq_len, d_model].

        Text goes through Gemma's own `embed_tokens`, which multiplies by sqrt(hidden_size) internally; image, state
        and CLS embeddings are passed unscaled (same as Gemma-3 multimodal, which scatters projector outputs into
        already-scaled text embeddings).
        """
        img = self.encode_images(images)
        b = img.shape[0]
        dtype = img.dtype
        text = self.language_model.embed_tokens(self.prompt_ids[None]).to(dtype).expand(b, -1, -1)
        parts = [img, text]
        if self.state_proj is not None:
            parts.append(self.state_proj(self.normalize_state(state))[:, None, :].to(dtype))
        parts.append(self.cls_token.to(dtype)[None, None, :].expand(b, 1, -1))
        return torch.cat(parts, dim=1)

    def forward(self, images: dict[str, Tensor], state: Tensor) -> Tensor:
        """images: {camera: uint8 [B,3,224,224]}, state: float [B,14] (raw) -> logits float32 [B, NUM_BINS]."""
        embeds = self.build_inputs_embeds(images, state)
        hidden = self.language_model(inputs_embeds=embeds, use_cache=False).last_hidden_state  # final norm applied
        cls_out = hidden[:, -1]
        return self.head(self.head_dropout(cls_out)).float()

    @torch.no_grad()
    def predict(self, images: dict[str, Tensor], state: Tensor) -> dict[str, Tensor]:
        """Eval-mode forward -> {"value": [B], "value_std": [B], "logits": [B, NUM_BINS]} (float32)."""
        was_training = self.training
        self.eval()
        try:
            logits = self.forward(images, state)
        finally:
            self.train(was_training)
        return {"value": bins.expected_value(logits), "value_std": bins.value_std(logits), "logits": logits}

    # ------------------------------------------------------------------------------------------------------------
    def param_groups(self, lr_backbone: float, lr_new: float, weight_decay: float) -> list[dict]:
        """AdamW groups: pretrained vs new modules, each split into decay / no-decay (norms, biases, embeddings)."""
        groups: dict[str, dict] = {}
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            is_new = name.split(".")[0] in NEW_MODULES
            no_decay = p.ndim < 2 or "norm" in name or "embed" in name or name == "cls_token"
            key = ("new" if is_new else "backbone") + ("_no_decay" if no_decay else "_decay")
            if key not in groups:
                groups[key] = {
                    "name": key,
                    "params": [],
                    "lr": lr_new if is_new else lr_backbone,
                    "weight_decay": 0.0 if no_decay else weight_decay,
                }
            groups[key]["params"].append(p)
        return list(groups.values())

    def parameter_counts(self) -> dict[str, dict[str, int]]:
        """{component: {"total", "trainable"}} for vision / language (+ token embedding) / new / all."""

        def count(params) -> dict[str, int]:
            params = list(params)
            return {
                "total": sum(p.numel() for p in params),
                "trainable": sum(p.numel() for p in params if p.requires_grad),
            }

        new = [p for n, p in self.named_parameters() if n.split(".")[0] in NEW_MODULES]
        return {
            "vision_tower": count(self.vision_tower.parameters()),
            "language_model": count(self.language_model.parameters()),
            "  of which token_embedding": count(self.language_model.embed_tokens.parameters()),
            "new_modules": count(new),
            "all": count(self.parameters()),
        }

    # ------------------------------------------------------------------------------------------------------------
    def save_pretrained(self, path: Path, extra: dict[str, Any] | None = None) -> None:
        """Write `model.safetensors` + `config.yaml` (model cfg + backbone configs; `extra` merged at top level)."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        state = {k: v.detach().contiguous().cpu() for k, v in self.state_dict().items()}
        save_file(state, str(path / WEIGHTS_NAME))
        doc = {
            "model": self.cfg.to_dict(),
            "vision_config": _jsonable(self.vision_tower.config.to_dict()),
            "text_config": _jsonable(self._text_config_src.to_dict()),
        }
        if extra:
            doc.update(extra)
        (path / CONFIG_NAME).write_text(yaml.safe_dump(doc, sort_keys=False))

    @classmethod
    def from_pretrained(cls, path: Path, **overrides: Any) -> ValueModel:
        """Rebuild from a `save_pretrained` directory (no backbone download). `overrides` patch the model cfg."""
        path = Path(path)
        doc = yaml.safe_load((path / CONFIG_NAME).read_text())
        cfg = ValueModelConfig.from_dict({**doc["model"], **overrides})
        vision_config = SiglipVisionConfig.from_dict(doc["vision_config"])
        text_config = Gemma3TextConfig.from_dict(doc["text_config"])
        state = load_file(str(path / WEIGHTS_NAME))
        model = cls(
            cfg,
            state["state_q01"],
            state["state_q99"],
            vision_config=vision_config,
            text_config=text_config,
            prompt_ids=state["prompt_ids"],
        )
        model.load_state_dict(state, strict=True)
        return model
