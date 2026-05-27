"""PyTorch Dataset for loading encoded chess positions from HDF5.

Supports:
  - Pre-loads entire dataset into RAM for zero I/O during training
  - Falls back to lazy load for datasets too large for memory
  - Random sampling of N positions per epoch
  - Multi-worker DataLoader loading
"""
import os
import logging
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Threshold: datasets smaller than this (in GB) are fully loaded into RAM
_IN_MEMORY_THRESHOLD_GB = 8.0


class ChessPositionDataset(Dataset):
    """Dataset of chess positions for supervised learning."""

    def __init__(self, h5_path: str, max_samples: int = None):
        self.h5_path = h5_path
        self._X = None
        self._y_policy = None
        self._y_value = None

        # Get metadata first
        with h5py.File(h5_path, "r") as f:
            self.num_positions = f.attrs["num_positions"]
            # Estimate size: X is (N, 119, 8, 8) float32
            estimated_gb = self.num_positions * 119 * 8 * 8 * 4 / 1e9

        self.max_samples = min(max_samples, self.num_positions) if max_samples else self.num_positions

        if estimated_gb <= _IN_MEMORY_THRESHOLD_GB:
            logger.info(
                f"Pre-loading {self.num_positions:,} positions into RAM "
                f"(~{estimated_gb:.1f} GB)"
            )
            with h5py.File(h5_path, "r") as f:
                self._X = f["X"][:].astype(np.float32)
                self._y_policy = f["y_policy"][:].astype(np.int64)
                self._y_value = f["y_value"][:].astype(np.float32)
            logger.info("Pre-load complete.")
        else:
            logger.info(
                f"Dataset is large (~{estimated_gb:.1f} GB, threshold={_IN_MEMORY_THRESHOLD_GB} GB) — "
                "using lazy HDF5 access"
            )

    def __len__(self):
        return self.max_samples

    def __getitem__(self, idx):
        if self._X is not None:
            # Fast path: from RAM
            X = torch.from_numpy(self._X[idx])
            y_policy = torch.tensor(self._y_policy[idx], dtype=torch.long)
            y_value = torch.tensor(self._y_value[idx], dtype=torch.float32)
        else:
            # Slow path: lazy HDF5 (large datasets)
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
