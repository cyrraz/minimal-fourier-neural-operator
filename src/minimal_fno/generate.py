import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def generate_pde_data(grid_size=64, num_samples=1000, noise=0.0):
    """
    Generate synthetic data for training and testing the Fourier Neural Operator (FNO) model.
    The data is based on solving a 2D heat equation with a sinusoidal initial condition.

    Parameters:
        grid_size (int): Resolution of the spatial grid.
        num_samples (int): Number of samples to generate.
        noise (float): Noise level added to the generated data.

    Returns:
        dataset: TensorDataset object containing the generated data.
    """
    # Define spatial grid
    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    X, Y = np.meshgrid(x, y, indexing="ij")

    # Function to generate initial conditions and PDE solutions
    def pde_solution(x, y, time):
        return np.sin(np.pi * x) * np.sin(np.pi * y) * np.exp(-time)

    # Generate dataset
    inputs = []  # Input conditions (e.g., initial conditions or forcing terms)
    outputs = []  # Corresponding PDE solutions

    for _ in range(num_samples):
        time = np.random.uniform(0, 1)  # Random time parameter
        initial_condition = np.sin(np.pi * X) * np.sin(
            np.pi * Y
        )  # Example initial condition
        solution = pde_solution(X, Y, time) + noise * np.random.randn(
            grid_size, grid_size
        )

        inputs.append(initial_condition)
        outputs.append(solution)

    inputs = np.array(inputs, dtype=np.float32)
    outputs = np.array(outputs, dtype=np.float32)

    # Convert to torch tensors and reshape
    inputs = torch.tensor(inputs).unsqueeze(
        1
    )  # Shape: (num_samples, 1, grid_size, grid_size)
    outputs = torch.tensor(outputs).unsqueeze(
        1
    )  # Shape: (num_samples, 1, grid_size, grid_size)

    # Create TensorDataset object
    dataset = TensorDataset(inputs, outputs)

    return dataset
