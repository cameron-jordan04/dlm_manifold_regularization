import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from manifold_regularization import (
    ArithmeticDataset,
    compute_edit_distance,
    batch_compute_edit_distance,
    sample_sequence_pairs,
    MicroMDLM,
    MaskedDiffusionProcess,
    compute_mdlm_loss,
    compute_isometric_loss,
    compute_total_loss,
    evaluate_latent_interpolation,
    ValueTwistMLP,
    measure_lipschitz_continuity
)

# =============================================================================
# Fixtures for Reusable Test Components
# =============================================================================

@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def tiny_dataset():
    # Use a very small number of samples for fast testing
    return ArithmeticDataset(num_samples=32, seq_len=16)

@pytest.fixture
def config():
    return {
        "batch_size": 4,
        "seq_len": 16,
        "vocab_size": 19,
        "d_model": 32, # Scaled down for speed
        "n_layers": 2,
        "n_heads": 2,
        "num_timesteps": 50,
        "pad_id": 0,
        "mask_id": 1
    }

@pytest.fixture
def model(config, device):
    return MicroMDLM(
        vocab_size=config["vocab_size"],
        num_timesteps=config["num_timesteps"],
        d_model=config["d_model"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"]
    ).to(device)

@pytest.fixture
def diffusion(config):
    return MaskedDiffusionProcess(
        num_timesteps=config["num_timesteps"],
        vocab_size=config["vocab_size"],
        mask_token_id=config["mask_id"]
    )

# =============================================================================
# Part 1: Dataset and Distance Metric Tests
# =============================================================================

def test_arithmetic_dataset(tiny_dataset):
    """
    Test arithmetic_dataset]
    """
    assert len(tiny_dataset) == 32
    sample = tiny_dataset[0]
    assert sample.shape == (16,)
    assert sample.dtype == torch.long

    # Test decoding
    decoded = tiny_dataset.decode(sample)
    assert isinstance(decoded, str)
    assert "[PAD]" not in decoded # Decode should filter PADs

def test_compute_edit_distance():
    """
    Test compute_edit_distance
    """
    # Identical sequences -> 0.0
    seq_a = torch.tensor([2, 3, 4, 12, 5])
    seq_b = torch.tensor([2, 3, 4, 12, 5])
    assert compute_edit_distance(seq_a, seq_b) == 0.0

    # One substitution -> 1 / 5 = 0.2
    seq_c = torch.tensor([2, 3, 9, 12, 5])
    assert compute_edit_distance(seq_a, seq_c) == 0.2

    # Completely different lengths and content
    seq_empty = torch.tensor([])
    assert compute_edit_distance(seq_a, seq_empty) == 1.0

def test_batch_compute_edit_distance():
    """
    Test batch_compute_edit_distance
    """
    batch_a = torch.tensor([[2, 3, 4], [5, 6, 7]])
    batch_b = torch.tensor([[2, 3, 4], [5, 9, 7]]) # 0 mutations in first, 1 in second

    distances = batch_compute_edit_distance(batch_a, batch_b)
    assert distances.shape == (2,)
    assert torch.isclose(distances[0], torch.tensor(0.0))
    assert torch.isclose(distances[1], torch.tensor(1.0 / 3.0))

def test_sample_sequence_pairs(device):
    """
    Test sample_sequence_pairs
    """
    batch_size = 8
    seq_len = 16
    vocab_size = 19
    pad_id = 0

    # Create a dummy batch filled with 2s, but with padding at the end
    batch = torch.full((batch_size, seq_len), 2, device=device)
    batch[:, 12:] = pad_id 

    min_mut, max_mut = 1, 3
    x_a, x_b, distances = sample_sequence_pairs(
        batch, vocab_size, min_mutations=min_mut, max_mutations=max_mut, pad_id=pad_id
    )

    assert x_b.shape == x_a.shape
    assert distances.shape == (batch_size,)

    # Ensure edit distances are bounded properly and non-zero
    assert torch.all(distances >= 0.0)
    assert torch.all(distances <= 1.0)

    # Verify padding tokens were strictly protected
    pad_mask = (x_a == pad_id)
    assert torch.all(x_b[pad_mask] == pad_id)

# =============================================================================
# Part 2: Architecture Tests
# =============================================================================

def test_micro_mdlm_forward(model, config, device):
    """
    Test micro_mdlm forward
    """
    x = torch.randint(0, config["vocab_size"], (config["batch_size"], config["seq_len"])).to(device)
    t = torch.randint(0, config["num_timesteps"], (config["batch_size"],)).to(device)

    logits, latents = model(x, t)

    assert logits.shape == (config["batch_size"], config["seq_len"], config["vocab_size"])
    assert latents.shape == (config["batch_size"], config["seq_len"], config["d_model"])

    # Ensure gradients can flow through latents
    assert latents.requires_grad

def test_masked_diffusion_process(diffusion, config, device):
    """
    Test masked_diffusion_process
    """
    x_0 = torch.full((config["batch_size"], config["seq_len"]), 2).to(device)
    # Add padding to ensure it doesn't get masked
    x_0[:, -4:] = config["pad_id"]

    t = torch.randint(0, config["num_timesteps"], (config["batch_size"],)).to(device)
    x_t = diffusion.q_sample(x_0, t)

    assert x_t.shape == x_0.shape

    # Padding tokens should NEVER be masked
    pad_mask = x_0 == config["pad_id"]
    assert torch.all(x_t[pad_mask] == config["pad_id"])

# =============================================================================
# Part 3: Loss Function Tests
# =============================================================================

def test_compute_mdlm_loss():
    """
    Test compute_mdlm_loss
    """
    vocab_size = 19
    # ADDED requires_grad=True here
    logits = torch.randn(4, 16, vocab_size, requires_grad=True) 
    targets = torch.randint(0, vocab_size, (4, 16))
    mask = torch.rand(4, 16) > 0.5 # Random boolean mask

    loss = compute_mdlm_loss(logits, targets, mask)
    assert loss.ndim == 0
    assert loss.requires_grad

    # Test Edge Case: Empty Mask (no tokens masked)
    empty_mask = torch.zeros(4, 16, dtype=torch.bool)
    loss_empty = compute_mdlm_loss(logits, targets, empty_mask)
    assert loss_empty.item() == 0.0

def test_compute_isometric_loss():
    """
    Test compute_isometric_loss
    """
    batch_size, seq_len, d_model = 4, 16, 32
    z_a = torch.randn(batch_size, seq_len, d_model, requires_grad=True)

    # Test identical latents (distance should be 0)
    d_edit = torch.zeros(batch_size)
    loss_identical = compute_isometric_loss(z_a, z_a, d_edit, d_max=10.0)
    assert torch.isclose(loss_identical, torch.tensor(0.0))

    # Test separate latents
    z_b = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
    d_edit_random = torch.rand(batch_size)
    loss_diff = compute_isometric_loss(z_a, z_b, d_edit_random, d_max=10.0)
    assert loss_diff.item() > 0.0
    assert loss_diff.requires_grad

def test_compute_total_loss(model, diffusion, config, device):
    """
    Test compute_total_loss
    """
    x_a = torch.randint(2, config["vocab_size"], 
                        (config["batch_size"], config["seq_len"])).to(device)
    x_b = torch.randint(2, config["vocab_size"], 
                        (config["batch_size"], config["seq_len"])).to(device)
    d_edit = torch.rand(config["batch_size"]).to(device)

    loss, metrics = compute_total_loss(
        model, diffusion, x_a, x_b, d_edit, lambda_iso=0.1, d_max=10.0, pad_id=config["pad_id"]
    )

    assert loss.requires_grad
    assert "loss_mdlm" in metrics
    assert "loss_iso" in metrics
    assert "total_loss" in metrics

# =============================================================================
# Part 4: Evaluation Metrics Tests
# =============================================================================

def test_evaluate_latent_interpolation(model, config, device):
    """
    Test evaluate_latent_interpolation
    """
    x_a = torch.randint(0, config["vocab_size"], 
                        (config["batch_size"], config["seq_len"])).to(device)
    x_b = torch.randint(0, config["vocab_size"], 
                        (config["batch_size"], config["seq_len"])).to(device)

    alphas = [0.0, 0.5, 1.0]
    decoded_seqs = evaluate_latent_interpolation(model, x_a, x_b, alphas)

    assert len(decoded_seqs) == 3
    for seq in decoded_seqs:
        assert seq.shape == (config["batch_size"], config["seq_len"])

def test_value_twist_mlp(config):
    """
    Test value_twist_mlp
    """
    mlp = ValueTwistMLP(d_model=config["d_model"])
    z = torch.randn(config["batch_size"], config["seq_len"], config["d_model"])

    output = mlp(z)

    assert output.shape == (config["batch_size"], 1)
    # Check if Sigmoid boundary holds
    assert torch.all(output >= 0.0)
    assert torch.all(output <= 1.0)

def test_measure_lipschitz_continuity(model, config, device):
    """
    Test measure_lipschitz_continuity
    """
    value_model = ValueTwistMLP(d_model=config["d_model"]).to(device)

    # Create a dummy dataloader
    data = torch.randint(0, config["vocab_size"], (16, config["seq_len"]))
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"])

    variance = measure_lipschitz_continuity(model, value_model, dataloader)

    assert isinstance(variance, float)
    assert variance >= 0.0 # Variance cannot be negative

# =============================================================================
# Execution Block
# =============================================================================

if __name__ == "__main__":
    import sys
    
    # Run pytest on the current file with verbose output
    exit_code = pytest.main(["-v", __file__])
    
    # Exit with the code returned by pytest
    sys.exit(exit_code)