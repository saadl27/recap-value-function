import numpy as np
import torch

from recap_vf import bins


def test_support_endpoints():
    c = bins.bin_centers()
    assert c.shape == (bins.NUM_BINS,)
    assert c[0].item() == bins.V_MIN
    assert c[-1].item() == bins.V_MAX
    assert np.isclose(c[1].item() - c[0].item(), bins.BIN_WIDTH)


def test_value_to_bin_roundtrip_and_clipping():
    c = bins.bin_centers()
    assert torch.equal(bins.value_to_bin(c), torch.arange(bins.NUM_BINS))
    assert bins.value_to_bin(torch.tensor([-5.0, 3.0])).tolist() == [0, bins.NUM_BINS - 1]
    v = np.random.default_rng(0).uniform(-1, 0, 1000)
    assert np.array_equal(bins.value_to_bin_np(v), bins.value_to_bin(torch.from_numpy(v)).numpy())


def test_hl_gauss_is_a_distribution_centred_on_value():
    v = torch.tensor([-0.73, -0.5, 0.0, -1.0])
    p = bins.hl_gauss_targets(v)
    assert torch.allclose(p.sum(-1), torch.ones(4))
    # mean of the soft target stays close to the value (edges are truncated, hence the tolerance)
    assert torch.allclose(p @ bins.bin_centers(), v, atol=bins.BIN_WIDTH)


def test_expected_value_and_std_of_one_hot():
    logits = torch.full((2, bins.NUM_BINS), -1e4)
    logits[0, 0] = 0.0
    logits[1, 150] = 0.0
    assert torch.allclose(bins.expected_value(logits), torch.tensor([-1.0, -0.25]))
    assert torch.allclose(bins.value_std(logits), torch.zeros(2), atol=1e-3)
