# Step 0 — Environment setup

**Owner:** orchestrator. Run once on the **laptop** before launching agents, and once on the **workstation** after
its first `git clone`. **Time:** ~15 min each.

## Goal
A working `.venv` with CUDA PyTorch, access to both backbones on the Hugging Face Hub, and enough disk.

## Instructions

1. **Install uv** in user space if missing (no sudo): `curl -LsSf https://astral.sh/uv/install.sh | sh`
   (installs to `~/.local/bin`). Check `nvidia-smi` (driver version) and `df -h ~`.
2. **Create the environment** from the repo root:
   ```bash
   uv sync
   uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   uv run python -c "import torch; print(torch.cuda.get_device_capability(0))"   # 5090 → (12, 0)
   ```
   On the RTX 5090 the wheel **must** be a CUDA 12.8 build (`+cu128`); older builds fail with
   "no kernel image is available for execution on the device". The NVIDIA driver must be ≥ 570.
3. **Hugging Face access** — agents never handle tokens or passwords.
   - Saad may log in himself on the workstation (`uv run huggingface-cli login`); otherwise no token is needed:
     the dataset and SigLIP2 are public and Gemma has an ungated mirror (below).
   - `google/gemma-3-270m` is **gated**: accept the Gemma licence on its model page with the same account.
     If access is not possible, `unsloth/gemma-3-270m` is an ungated mirror with identical weights
     (`constants.LANGUAGE_BACKBONE_MIRROR`); record which one was used.
   - `google/siglip2-so400m-patch14-224` is not gated.
   - Check: `uv run python -c "from huggingface_hub import model_info as m; print(m('google/gemma-3-270m').sha)"`.
4. **Disk budget** (check with `df -h $RECAP_DATA_ROOT`):

   | Item | Size |
   |---|---|
   | `.venv` (torch cu128 + deps) | ~6 GB |
   | HF cache: SigLIP2 (1.1 GB fp32) + Gemma-3-270M (0.5 GB) | ~2 GB |
   | Raw dataset | 5.3 GB |
   | JPEG frame store (step 1) | ~20–30 GB (step 1 measures it) |
   | Checkpoints (≈2.8 GB each in fp32 / 1.4 GB bf16), keep ≤ 3 per run | ~5–10 GB per run |

   The dev laptop has ~15 GB free → only run steps 1/4 on a **subset** of episodes there. Point `RECAP_DATA_ROOT`
   / `RECAP_RUNS_ROOT` at a large disk on the office machine if the repo disk is small.
5. **Sanity:** `uv run pytest` passes (only `tests/test_bins.py` exists at this point).

## Definition of done
- [ ] `torch.cuda.is_available()` is True and the device is listed correctly
- [ ] Gemma-3-270M and SigLIP2 configs download (`AutoConfig.from_pretrained(...)`)
- [ ] ≥ 45 GB free on the volume behind `RECAP_DATA_ROOT` (office machine)
- [ ] `uv run pytest` green
