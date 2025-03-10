import torch
import pytest
import sys
import os
from minimal_fno.model import SpectralConv2d, FNO2d
from lightning.pytorch import Trainer
from torch.utils.data import DataLoader

@pytest.fixture
def sample_input():
    return torch.randn(1, 1, 64, 64)

def test_spectral_conv2d_forward(sample_input):
    conv = SpectralConv2d(in_channels=1, out_channels=1, modes1=16, modes2=16)
    output = conv(sample_input)
    assert output.shape == sample_input.shape

def test_fno2d_forward(sample_input):
    model = FNO2d(modes1=16, modes2=16, width=32)
    output = model(sample_input)
    assert output.shape == (1, 64, 64, 1)

def test_training_step():
    model = FNO2d(modes1=16, modes2=16, width=32)
    optimizer = model.configure_optimizers()
    sample_input = torch.randn(1, 1, 64, 64)
    sample_target = torch.randn(1, 64, 64, 1)
    batch = (sample_input, sample_target)
    loss = model.training_step(batch, 0)
    assert loss.item() > 0

def test_validation_step():
    model = FNO2d(modes1=16, modes2=16, width=32)
    sample_input = torch.randn(1, 1, 64, 64)
    sample_target = torch.randn(1, 64, 64, 1)
    batch = (sample_input, sample_target)
    model.validation_step(batch, 0)