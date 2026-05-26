"""PyTorch Dataset for loading encoded chess positions from HDF5.

Supports:
  - Random sampling of N positions per epoch
  - Multi-worker DataLoader loading
"""
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class ChessPositionDataset(Dataset):
    """Dataset of chess positions for supervised learning."""

    def __init__(self, h5_path: str, max_samples: int = None):
        self.h5_path = h5_path
        with h5py.File(h5_path, "r") as f:
            self.num_positions = f.attrs["num_positions"]
        self.max_samples = min(max_samples, self.num_positions) if max_samples else self.num_positions

    def __len__(self):
        return self.max_samples

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, "r") as f:
            X = torch.from_numpy(f["X"][idx].astype(np.float32))
            y_policy = torch.tensor(f["y_policy"][idx], dtype=torch.long)
            y_value = torch.tensor(f["y_value"][idx], dtype=torch.float32)

        legal_mask = torch.ones(4096, dtype=torch.float32)

        return {
            "X": X,
            "y_policy": y_policy,
            "y_value": y_value,
            "legal_mask": legal_mask,
        }


class RandomSliceDataset(Dataset):
    """Wraps ChessPositionDataset and yields a random subset each epoch."""

    def __init__(self, base_dataset: ChessPositionDataset, samples_per_epoch: int):
        self.base = base_dataset
        self.samples_per_epoch = samples_per_epoch

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        random_idx = np.random.randint(0, self.base.num_positions)
        return self.base[random_idx]
