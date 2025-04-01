"""
This script downloads the Darcy Flow dataset if not already downloaded and prints the number of samples in the training and testing sets.
"""

import logging

from neuralop.data.datasets import DarcyDataset

# Set up basic parameters
data_dir = "./data"
n_train = 100
n_tests = [50, 50]
batch_size = 4
test_batch_sizes = [16, 16]
train_resolution = 16
test_resolutions = [16, 32]

# Create the dataset, which will download the data if needed
dataset = DarcyDataset(
    root_dir=data_dir,
    n_train=n_train,
    n_tests=n_tests,
    batch_size=batch_size,
    test_batch_sizes=test_batch_sizes,
    train_resolution=train_resolution,
    test_resolutions=test_resolutions,
    encode_input=False,
    encode_output=True,
    encoding="channel-wise",
    channel_dim=1,
    subsampling_rate=None,
    download=True,
)

# Set up logging
logging.basicConfig(format="%(message)s: %(data)s", level=logging.INFO)

logging.info(
    "Darcy Flow dataset downloaded and loaded successfully to/from %s", data_dir
)
logging.info("Training samples: %i", len(dataset.train_db))
for res, test_db in dataset.test_dbs.items():
    logging.info("Testing samples at resolution %i: %i", res, len(test_db))
