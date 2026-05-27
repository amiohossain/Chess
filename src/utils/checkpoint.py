"""Checkpoint save/load/resume utilities.

Supports:
  - Saving model, optimizer, scheduler state
  - Resuming from the latest checkpoint
  - Cloud upload to Google Drive / Kaggle Dataset
"""
import os
import glob
import time
import torch
from src.config import PathConfig


def save_latest_weights(
    model: torch.nn.Module,
    step: int,
    loss: float,
    save_dir: str = "./checkpoints",
) -> str:
    """Quick save of model weights + step number only (no optimizer/scheduler).

    Lightweight per-step checkpoint so crashes lose at most 1 step.
    Overwrites ``checkpoint_latest.pt`` each call — only one file on disk.
    """
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "checkpoint_latest.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "step": step,
        "loss": loss,
        "tag": "latest",
        "saved_at": time.time(),
    }, path)
    return path


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler = None,
    step: int = 0,
    epoch: int = 0,
    loss: float = None,
    config: PathConfig = None,
    tag: str = "latest",
    extra: dict = None,
) -> str:
    """Save a training checkpoint."""
    save_dir = config.checkpoint_dir if config else "./checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "epoch": epoch,
        "loss": loss,
        "tag": tag,
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    if extra:
        checkpoint.update(extra)

    path = os.path.join(save_dir, f"checkpoint_{tag}.pt")
    torch.save(checkpoint, path)

    if tag != "latest":
        latest_path = os.path.join(save_dir, "checkpoint_latest.pt")
        torch.save(checkpoint, latest_path)

    return path


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler = None,
    device: torch.device = None,
):
    """Load a checkpoint and return saved state."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint


def find_latest_checkpoint(checkpoint_dir: str) -> str:
    """Find the latest checkpoint in a directory."""
    pattern = os.path.join(checkpoint_dir, "checkpoint_*.pt")
    files = glob.glob(pattern)
    if not files:
        return ""
    latest = os.path.join(checkpoint_dir, "checkpoint_latest.pt")
    if os.path.exists(latest):
        return latest
    return max(files, key=os.path.getmtime)


def save_for_kaggle(checkpoint_path: str, kaggle_dataset_name: str = "chess-training-data-2013") -> None:
    """Print instructions for uploading checkpoint to Kaggle Dataset."""
    print(f"Checkpoint saved at {checkpoint_path}")
    print(f"To upload to Kaggle Dataset '{kaggle_dataset_name}':")
    print(f"  kaggle datasets version -p {os.path.dirname(checkpoint_path)} -m 'checkpoint update'")
