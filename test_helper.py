import torch
import pytest
from helper import LLAMAConfig, LLAMAModel, RMSNorm, MultiheadAttention, precompute_freqs_cis


@pytest.fixture
def small_config():
    return LLAMAConfig(vocab_size=256, dim=64, n_layers=2, n_heads=4, n_kv_heads=4, max_seq_len=32)

@pytest.fixture
def freqs_cis(small_config):
    return precompute_freqs_cis(small_config.head_dim, small_config.max_seq_len, small_config.rope_theta)


# --- RMSNorm ---

def test_rmsnorm_output_shape(small_config):
    norm = RMSNorm(small_config.dim)
    x = torch.randn(2, 10, small_config.dim)
    assert norm(x).shape == x.shape

def test_rmsnorm_unit_weight_approx_unit_norm(small_config):
    norm = RMSNorm(small_config.dim)
    x = torch.randn(2, 10, small_config.dim)
    out = norm(x)
    rms = out.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)

def test_rmsnorm_weight_scales_output(small_config):
    norm = RMSNorm(small_config.dim)
    x = torch.randn(1, 4, small_config.dim)
    norm.weight.data.fill_(2.0)
    out = norm(x)
    rms = out.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.full_like(rms, 2.0), atol=1e-4)


# --- MultiheadAttention ---

def test_mha_output_shape(small_config, freqs_cis):
    attn = MultiheadAttention(small_config)
    x = torch.randn(2, 10, small_config.dim)
    assert attn(x, freqs_cis).shape == x.shape

def test_mha_causal_mask(small_config, freqs_cis):
    attn = MultiheadAttention(small_config)
    attn.eval()
    x = torch.randn(1, 8, small_config.dim)
    x_perturbed = x.clone()
    x_perturbed[0, 7] += 100.0
    out1 = attn(x, freqs_cis)
    out2 = attn(x_perturbed, freqs_cis)
    assert torch.allclose(out1[0, 0], out2[0, 0], atol=1e-5)

def test_mha_single_token(small_config, freqs_cis):
    attn = MultiheadAttention(small_config)
    x = torch.randn(1, 1, small_config.dim)
    assert attn(x, freqs_cis).shape == (1, 1, small_config.dim)


# --- LLAMAModel ---

def test_model_output_shape(small_config):
    model = LLAMAModel(small_config)
    input_ids = torch.randint(0, small_config.vocab_size, (2, 10))
    logits = model(input_ids)
    assert logits.shape == (2, 10, small_config.vocab_size)

def test_model_forward_no_crash(small_config):
    model = LLAMAModel(small_config)
    input_ids = torch.randint(0, small_config.vocab_size, (1, 16))
    model(input_ids)

def test_model_single_token(small_config):
    model = LLAMAModel(small_config)
    input_ids = torch.randint(0, small_config.vocab_size, (1, 1))
    logits = model(input_ids)
    assert logits.shape == (1, 1, small_config.vocab_size)
