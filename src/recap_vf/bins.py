"""Discretised support of the distributional value function.

The paper discretises normalised returns in [-1, 0] into B = 201 bins and trains with cross-entropy;
the scalar value is the expectation of the bin centres under the predicted distribution.
Bin b has centre v_b = V_MIN + b * (V_MAX - V_MIN) / (NUM_BINS - 1), so bin 0 is -1 and bin 200 is 0.
"""

import math

import numpy as np
import torch

NUM_BINS = 201
V_MIN = -1.0
V_MAX = 0.0
BIN_WIDTH = (V_MAX - V_MIN) / (NUM_BINS - 1)


def bin_centers(device: torch.device | str | None = None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.linspace(V_MIN, V_MAX, NUM_BINS, device=device, dtype=dtype)


def value_to_bin(values: torch.Tensor) -> torch.Tensor:
    """Index of the nearest bin centre (values outside [V_MIN, V_MAX] are clipped)."""
    scaled = (values.clamp(V_MIN, V_MAX) - V_MIN) / BIN_WIDTH
    return scaled.round().long()


def value_to_bin_np(values: np.ndarray) -> np.ndarray:
    """NumPy twin of `value_to_bin`, used when writing the returns table."""
    scaled = (np.clip(values, V_MIN, V_MAX) - V_MIN) / BIN_WIDTH
    return np.rint(scaled).astype(np.int64)


def hl_gauss_targets(values: torch.Tensor, sigma_bins: float = 0.75) -> torch.Tensor:
    """Soft targets (HL-Gauss, Imani & White 2018; Farebrother et al. 2024): mass of N(value, sigma) in each bin.

    Optional alternative to the paper's one-hot targets. sigma is expressed in bin widths.
    Returns [B, NUM_BINS] rows that sum to 1.
    """
    values = values.float().clamp(V_MIN, V_MAX)
    edges = torch.linspace(
        V_MIN - BIN_WIDTH / 2, V_MAX + BIN_WIDTH / 2, NUM_BINS + 1, device=values.device, dtype=torch.float32
    )
    sigma = sigma_bins * BIN_WIDTH
    cdf = 0.5 * (1 + torch.erf((edges[None, :] - values[:, None]) / (sigma * math.sqrt(2))))
    probs = cdf[:, 1:] - cdf[:, :-1]
    return probs / probs.sum(dim=-1, keepdim=True)


def expected_value(logits: torch.Tensor) -> torch.Tensor:
    """E[v] under softmax(logits). logits: [..., NUM_BINS] -> [...]."""
    probs = logits.float().softmax(dim=-1)
    return probs @ bin_centers(logits.device)


def value_std(logits: torch.Tensor) -> torch.Tensor:
    """Standard deviation of the predicted value distribution (uncertainty diagnostic)."""
    probs = logits.float().softmax(dim=-1)
    centers = bin_centers(logits.device)
    mean = probs @ centers
    var = probs @ centers.square() - mean.square()
    return var.clamp_min(0).sqrt()
