"""CPU tests for the value model on tiny random backbones (no download)."""

import math

import pytest
import torch
from transformers import Gemma3TextConfig, SiglipVisionConfig

from recap_vf import bins, constants
from recap_vf.model import ValueModel, ValueModelConfig

B = 2
WINDOW = 16


def vision_config() -> SiglipVisionConfig:
    return SiglipVisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        image_size=224,
        patch_size=14,
    )


def text_config(all_sliding: bool = False) -> Gemma3TextConfig:
    n_layers = 6
    if all_sliding:
        layer_types = ["sliding_attention"] * n_layers
    else:
        layer_types = ["sliding_attention"] * 5 + ["full_attention"]
    return Gemma3TextConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=n_layers,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=8,
        query_pre_attn_scalar=8,
        sliding_window=WINDOW,
        layer_types=layer_types,
        max_position_embeddings=2048,
        pad_token_id=0,
    )


def make_model(all_sliding: bool = False, **cfg_kwargs) -> ValueModel:
    torch.manual_seed(0)
    cfg = ValueModelConfig(**cfg_kwargs)
    model = ValueModel.from_backbone_configs(cfg, vision_config(), text_config(all_sliding))
    return model


def make_batch(seed: int = 0) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    images = {c: torch.randint(0, 256, (B, 3, 224, 224), dtype=torch.uint8, generator=g) for c in constants.CAMERAS}
    state = torch.randn(B, constants.STATE_DIM, generator=g)
    return images, state


def randomize_head(model: ValueModel) -> None:
    with torch.no_grad():
        model.head.weight.normal_(std=1.0)
        model.head.bias.normal_(std=1.0)


def cls_hidden(model: ValueModel, images, state) -> torch.Tensor:
    embeds = model.build_inputs_embeds(images, state)
    return model.language_model(inputs_embeds=embeds, use_cache=False).last_hidden_state[:, -1]


def test_output_shape_dtype_and_init():
    model = make_model()
    images, state = make_batch()
    logits = model(images, state)
    assert logits.shape == (B, bins.NUM_BINS)
    assert logits.dtype == torch.float32
    assert torch.all(logits == 0)
    assert torch.allclose(bins.expected_value(logits), torch.full((B,), -0.5))
    loss = torch.nn.functional.cross_entropy(logits, torch.randint(0, bins.NUM_BINS, (B,)))
    assert math.isclose(loss.item(), math.log(bins.NUM_BINS), rel_tol=1e-6)
    out = model.predict(images, state)
    assert set(out) == {"value", "value_std", "logits"}
    assert out["value"].shape == (B,)


def test_bf16_autocast_returns_float32():
    model = make_model()
    images, state = make_batch()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits = model(images, state)
    assert logits.dtype == torch.float32


@pytest.mark.parametrize("full_attention", [True, False])
def test_reach_first_camera(full_attention: bool):
    """CLS must see the first camera's tokens only when the sliding-window override is on."""
    model = make_model(all_sliding=True, full_attention=full_attention).eval()
    assert model.seq_len > WINDOW * model.language_model.config.num_hidden_layers  # out of reach without override
    images, state = make_batch()
    perturbed = dict(images)
    perturbed[constants.CAMERAS[0]] = 255 - images[constants.CAMERAS[0]]
    with torch.no_grad():
        h0 = cls_hidden(model, images, state)
        h1 = cls_hidden(model, perturbed, state)
        # sanity: the perturbation does change the first camera's tokens
        t0 = model.encode_images(images)[:, : model.tokens_per_camera]
        t1 = model.encode_images(perturbed)[:, : model.tokens_per_camera]
    assert not torch.allclose(t0, t1)
    if full_attention:
        assert (h0 - h1).abs().max() > 1e-4
        assert model.language_model.config.sliding_window >= model.seq_len
    else:
        torch.testing.assert_close(h0, h1, rtol=0, atol=1e-6)


def test_override_keeps_layer_types():
    """The override must not rewrite layer_types (they select the per-layer RoPE base)."""
    model = make_model(all_sliding=False, full_attention=True)
    assert model.language_model.config.layer_types == text_config().layer_types


def test_gradients_trainable_vs_frozen():
    model = make_model(freeze_vision=True)
    randomize_head(model)  # zero head would block gradients to everything upstream
    images, state = make_batch()
    loss = torch.nn.functional.cross_entropy(model(images, state), torch.tensor([3, 150]))
    loss.backward()
    n_trainable = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            n_trainable += 1
            assert p.grad is not None and p.grad.abs().sum() > 0, f"no grad for trainable {name}"
        else:
            assert p.grad is None, f"frozen {name} received a grad"
    assert n_trainable > 0
    assert not model.language_model.embed_tokens.weight.requires_grad  # freeze_token_embeddings default
    assert not any(p.requires_grad for p in model.vision_tower.parameters())


def test_gradients_full_finetune():
    model = make_model(freeze_token_embeddings=False, gradient_checkpointing=True).train()
    randomize_head(model)
    images, state = make_batch()
    loss = torch.nn.functional.cross_entropy(model(images, state), torch.tensor([0, 200]))
    loss.backward()
    for name, p in model.named_parameters():
        assert p.requires_grad
        if name == "language_model.embed_tokens.weight":
            # only the prompt rows get gradient
            assert p.grad[model.prompt_ids].abs().sum() > 0
            continue
        assert p.grad is not None and p.grad.abs().sum() > 0, f"no grad for {name}"


def test_param_groups():
    model = make_model(freeze_vision=True)
    groups = model.param_groups(lr_backbone=1e-5, lr_new=1e-4, weight_decay=0.01)
    names = {g["name"]: g for g in groups}
    assert set(names) == {"backbone_decay", "backbone_no_decay", "new_decay", "new_no_decay"}
    grouped = [p for g in groups for p in g["params"]]
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert len(grouped) == len(trainable) and {id(p) for p in grouped} == {id(p) for p in trainable}
    assert names["new_decay"]["lr"] == 1e-4 and names["backbone_decay"]["lr"] == 1e-5
    assert names["new_no_decay"]["weight_decay"] == 0.0
    new_no_decay = {id(p) for p in names["new_no_decay"]["params"]}
    for p in (model.cls_token, model.camera_embed, model.head.bias, model.projector_norm.weight):
        assert id(p) in new_no_decay
    assert id(model.head.weight) in {id(p) for p in names["new_decay"]["params"]}
    torch.optim.AdamW(groups)  # accepted by the optimizer


def test_save_load_roundtrip(tmp_path):
    model = make_model().eval()
    randomize_head(model)
    images, state = make_batch()
    with torch.no_grad():
        ref = model(images, state)
    model.save_pretrained(tmp_path / "ckpt", extra={"returns_tag": "c2229"})
    assert (tmp_path / "ckpt" / "model.safetensors").exists()
    loaded = ValueModel.from_pretrained(tmp_path / "ckpt").eval()
    assert loaded.cfg == model.cfg
    with torch.no_grad():
        out = loaded(images, state)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    torch.testing.assert_close(loaded.state_q01, model.state_q01)
    assert torch.equal(loaded.prompt_ids, model.prompt_ids)


def test_state_mode_none():
    with_state = make_model(state_mode="linear")
    without = make_model(state_mode="none")
    assert without.seq_len == with_state.seq_len - 1
    images, state = make_batch()
    assert without.build_inputs_embeds(images, state).shape[1] == without.seq_len
    assert with_state.build_inputs_embeds(images, state).shape[1] == with_state.seq_len
    assert without(images, state).shape == (B, bins.NUM_BINS)
    assert without.state_proj is None


def test_state_normalisation():
    q01 = torch.zeros(constants.STATE_DIM)
    q99 = torch.full((constants.STATE_DIM,), 2.0)
    model = ValueModel.from_backbone_configs(ValueModelConfig(), vision_config(), text_config(), q01, q99)
    s = torch.stack([q01, q99, torch.ones(constants.STATE_DIM), torch.full((constants.STATE_DIM,), 10.0)])
    n = model.normalize_state(s)
    assert torch.allclose(n[0], -torch.ones(constants.STATE_DIM))
    assert torch.allclose(n[1], torch.ones(constants.STATE_DIM))
    assert torch.allclose(n[2], torch.zeros(constants.STATE_DIM))
    assert torch.allclose(n[3], torch.ones(constants.STATE_DIM))  # clipped
