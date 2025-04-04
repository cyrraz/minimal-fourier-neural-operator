"""
Implements a minimal Fourier Neural Operator (FNO) using Lightning for learning mappings between function spaces.
"""

import lightning as L
import torch
import torch.nn.functional as F
from torch import nn


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        """
        Initialize spectral convolution with learnable complex weights.

        in_channels: Number of input feature channels.
        out_channels: Number of output feature channels.
        modes1, modes2: Number of retained Fourier modes for computation.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.weight1 = nn.Parameter(
            torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weight2 = nn.Parameter(
            torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def forward(self, x):
        batchsize, height, width = x.shape[0], x.shape[2], x.shape[3]

        # Compute 2D FFT on input
        x_ft = torch.fft.rfft2(x)

        # Ensure output tensor has correct shape
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            height,
            width // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        # Apply spectral convolution with correct einsum dimensions
        out_ft[:, :, : self.modes1, : self.modes2] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, : self.modes1, : self.modes2], self.weight1
        )
        out_ft[:, :, -self.modes1 :, : self.modes2] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, -self.modes1 :, : self.modes2], self.weight2
        )

        # Convert back to spatial domain
        return torch.fft.irfft2(out_ft, s=(height, width))


class MLP(nn.Module):
    """A simple Multi-Layer Perceptron (MLP) using 1x1 convolutions."""

    def __init__(self, in_channels, out_channels, mid_channels):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1),
        )

    def forward(self, x):
        return self.mlp(x)


class FNO2d(L.LightningModule):
    def __init__(self, modes1, modes2, width):
        """
        Initialize the FNO model with spectral and pointwise convolutions.

        modes1, modes2: Number of Fourier modes for convolution.
        width: Feature dimension after the initial lifting.
        """
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        # Lift input from 1 channel to a higher-dimensional feature space.
        self.fc0 = nn.Linear(1, self.width)
        # Spectral convolution operating in the Fourier domain.
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        # Pointwise convolution (1x1) for feature mixing in the spatial domain.
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        # Fully connected layers for final feature projection.
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # Lift input features.
        x = self.fc0(
            x.permute(0, 2, 3, 1)
        )  # Reshape from (batch, 1, H, W) to (batch, H, W, 1)
        # Rearrange to (batch, channels, height, width) for convolution.
        x = x.permute(0, 3, 1, 2)
        # Apply spectral convolution and pointwise convolution.
        x1 = self.conv0(x)
        x2 = self.w0(x)
        # Combine the two convolution outputs.
        x = x1 + x2
        # Apply non-linear activation.
        x = F.gelu(x)
        # Rearrange back to (batch, height, width, channels) for the fully connected layers.
        x = x.permute(0, 2, 3, 1)
        # Further process features with fully connected layers.
        x = F.gelu(self.fc1(x))

        return self.fc2(x)

    def training_step(self, batch, _batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = F.mse_loss(y_pred, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, _batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = F.mse_loss(y_pred, y)
        self.log("val_loss", loss)

    def test_step(self, batch, _batch_idx):  # noqa: PT019
        x, y = batch
        y_pred = self.forward(x)
        loss = F.mse_loss(y_pred, y)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
