import pytest
import torch
from minimal_fno.generate import generate_pde_data


def test_generate_pde_data_shape():
    """Test the shape of the generated dataset."""
    grid_size = 64
    num_samples = 1000
    dataset = generate_pde_data(grid_size=grid_size, num_samples=num_samples, noise=0.0)
    inputs, outputs = dataset.tensors
    assert inputs.shape == (num_samples, 1, grid_size, grid_size)
    assert outputs.shape == (num_samples, 1, grid_size, grid_size)


def test_generate_pde_data_noise():
    """Test the noise level in the generated data."""
    grid_size = 64
    num_samples = 100
    noise = 0.1
    dataset = generate_pde_data(
        grid_size=grid_size, num_samples=num_samples, noise=noise
    )
    inputs, outputs = dataset.tensors
    noise_level = torch.abs(outputs - inputs)
    assert torch.mean(noise_level) > 0.0
    assert torch.mean(noise_level) < 5 * noise


def test_generate_pde_data_dtype():
    """Test the data type of the generated dataset."""
    dataset = generate_pde_data(grid_size=64, num_samples=1000, noise=0.0)
    inputs, outputs = dataset.tensors
    assert inputs.dtype == torch.float32
    assert outputs.dtype == torch.float32
