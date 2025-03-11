import torch
import pytest
import multiprocessing
from lightning.pytorch import Trainer
from torch.utils.data import DataLoader, TensorDataset
from minimal_fourier_neural_operator.model import SpectralConv2d, FNO2d

# Ensure proper multiprocessing setup
multiprocessing.set_start_method("spawn", force=True)


@pytest.fixture
def sample_input():
    """Fixture for generating a random input tensor with shape (10, 1, 64, 64)."""
    return torch.randn(10, 1, 64, 64)


@pytest.fixture
def sample_target():
    """Fixture for generating a random target tensor with shape (10, 64, 64, 1)."""
    return torch.randn(10, 64, 64, 1)


@pytest.fixture
def spectral_conv_layer():
    """Fixture for creating a SpectralConv2d layer."""
    return SpectralConv2d(in_channels=1, out_channels=1, modes1=16, modes2=16)


@pytest.fixture
def fno2d_model():
    """Fixture for creating an FNO2d model."""
    return FNO2d(modes1=16, modes2=16, width=32)


@pytest.fixture
def trainer():
    """Fixture for creating a Trainer instance."""
    return Trainer(max_epochs=1, log_every_n_steps=1)


@pytest.fixture
def dataloaders(sample_input, sample_target):
    """Fixture for creating training and validation dataloaders."""
    dataset = TensorDataset(sample_input, sample_target)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset, batch_size=1, num_workers=11, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, num_workers=11, persistent_workers=True
    )

    return train_loader, val_loader


def test_spectral_conv2d_forward(spectral_conv_layer, sample_input):
    """Test the forward pass of the SpectralConv2d layer."""
    output = spectral_conv_layer(sample_input)
    assert output.shape == sample_input.shape


def test_fno2d_forward(fno2d_model, sample_input):
    """Test the forward pass of the FNO2d model."""
    output = fno2d_model(sample_input)
    assert output.shape == (10, 64, 64, 1)


def test_training_step(fno2d_model, trainer, dataloaders):
    """Test the training step of the FNO2d model."""
    train_loader, val_loader = dataloaders
    trainer.fit(fno2d_model, train_loader, val_loader)
    assert trainer.logged_metrics is not None


def test_validation_step(fno2d_model, trainer, dataloaders):
    """Test the validation step of the FNO2d model."""
    _, val_loader = dataloaders
    trainer.validate(fno2d_model, val_loader)
    assert trainer.logged_metrics is not None


def test_test_step(fno2d_model, trainer, dataloaders):
    """Test the test step of the FNO2d model."""
    _, test_loader = dataloaders
    trainer.test(fno2d_model, test_loader)
    assert trainer.logged_metrics is not None
