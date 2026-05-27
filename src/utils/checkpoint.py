"""Checkpoint save/load/resume utilities.

Supports:
  - Saving model, optimizer, scheduler state
  - Resuming from the latest checkpoint
  - Best-loss checkpoint (only overwritten on improvement)
"""
import os
import glob
import time
import torch
from src.config import PathConfig


def _best_loss_path(save_dir: str) -> str:
    return os.path.join(save_dir, "checkpoint_best.pt")


def save_best_weights(
    model: torch.nn.Module,
    step: int,
    loss: float,
    save_dir: str = "./checkpoints",
) -> bool:
    """Save model weights only if *loss* is the lowest seen so far.

    Returns True if a save was performed (i.e. loss is a new best), False otherwise.
    Previous best checkpoint is silently overwritten — only one best file exists.
    """
    best_path = _best_loss_path(save_dir)
    prev_best = float("inf")

    if os.path.exists(best_path):
        prev = torch.load(best_path, map_location="cpu", weights_only=True)
        prev_best = prev.get("loss", float("inf"))

    if loss >= prev_best:
        return False

    os.makedirs(save_dir, exist_ok=True)
    state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    tmp = best_path + ".tmp"
    torch.save({
        "model_state_dict": state,
        "step": step,
        "loss": loss,
        "tag": "best",
        "saved_at": time.time(),
    }, tmp)
    os.replace(tmp, best_path)
    return True


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
