"""Checkpoint save/load/resume utilities.

Supports:
  - Saving model, optimizer, scheduler state
  - Resuming from the latest checkpoint
  - Best-loss checkpoint (only overwritten on improvement)
  - Git LFS sync for checkpoint persistence across Kaggle sessions
"""
import os
import glob
import time
import logging
import threading
import subprocess
import torch
from src.config import PathConfig

logger = logging.getLogger(__name__)

# -- git sync state --
_git_sync_lock = threading.Lock()
_git_token = None  # stored by init_lfs_and_auth, reused by sync_checkpoint_to_git


def init_lfs_and_auth(token: str = None) -> None:
    """Initialize Git LFS and configure auth for checkpoint persistence.

    Call once at startup (e.g. in Kaggle notebook Cell 1) to restore
    previously-pushed checkpoints via ``git lfs checkout``.

    Stores the token globally so sync_checkpoint_to_git can re-inject
    auth into the remote URL if needed.
    """
    global _git_token
    if token:
        _git_token = token
    elif not _git_token:
        _git_token = os.environ.get("GITHUB_TOKEN")

    # Derive repo root: assume CWD is inside the repo (Cell 1 does %cd ...)
    cwd = os.getcwd()

    # Try to install LFS, but don't abort on failure — auth setup matters more
    try:
        subprocess.run(["git", "lfs", "install"], capture_output=True, check=True)
        logger.info("Git LFS installed")
    except Exception as e:
        logger.warning(f"git lfs install failed (non-fatal): {e}")

    # Configure git user (needed for commits)
    for cfg, val in [("user.email", "kaggle@chess.ai"), ("user.name", "Kaggle Trainer")]:
        subprocess.run(["git", "config", cfg, val], capture_output=True)

    # Inject token into remote URL for authenticated pushes
    _ensure_remote_auth(cwd)

    # Restore previously-pushed LFS objects
    try:
        subprocess.run(["git", "lfs", "fetch"], capture_output=True, check=True, cwd=cwd)
        subprocess.run(["git", "lfs", "checkout"], capture_output=True, check=True, cwd=cwd)
        logger.info("LFS objects restored from remote")
    except Exception as e:
        logger.warning(f"LFS restore failed (expected on first run): {e}")


def _ensure_remote_auth(repo_root: str) -> None:
    """Inject the stored token into the 'origin' remote URL if needed."""
    global _git_token
    if not _git_token:
        return
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True, text=True, check=True, cwd=repo_root,
        )
        url = result.stdout.strip()
        if url.startswith("https://") and "@" not in url:
            authed_url = url.replace("https://", f"https://{_git_token}@")
            subprocess.run(
                ["git", "remote", "set-url", "origin", authed_url],
                capture_output=True, check=True, cwd=repo_root,
            )
            logger.info("Git remote configured with token auth")
    except Exception as e:
        logger.warning(f"Git remote auth setup failed: {e}")


def sync_checkpoint_to_git(save_dir: str, step: int) -> None:
    """Push checkpoint files to Git LFS in a background thread.

    If a previous sync is still in progress, this call is dropped.
    """
    if not _git_sync_lock.acquire(blocking=False):
        return

    def _sync():
        try:
            # The git repo root is one level above save_dir (e.g. /kaggle/working/Chess)
            repo_root = os.path.dirname(os.path.normpath(save_dir))

            # Only sync checkpoint_best.pt — skip periodic full checkpoints to save LFS storage
            best_path = os.path.join(save_dir, "checkpoint_best.pt")
            if not os.path.exists(best_path):
                logger.warning(f"Git sync: {best_path} not found — nothing to sync")
                return

            add = subprocess.run(
                ["git", "add", best_path], capture_output=True, text=True, cwd=repo_root,
            )
            if add.returncode != 0:
                logger.warning(f"Git add failed: {add.stderr.strip()}")
                return

            subprocess.run(
                ["git", "commit", "-m", f"checkpoint step {step}"],
                capture_output=True, text=True, cwd=repo_root,
            )

            # Ensure remote URL has auth token before pushing
            _ensure_remote_auth(repo_root)

            push = subprocess.run(
                ["git", "push", "origin", "HEAD:checkpoints"],
                capture_output=True, text=True, cwd=repo_root,
            )
            if push.returncode != 0:
                logger.warning(f"Git push failed: {push.stderr.strip()}")
                return

            logger.info(f"Checkpoint synced to git (branch: checkpoints, step {step})")
        except Exception as e:
            logger.warning(f"Git sync failed at step {step}: {e}")
        finally:
            _git_sync_lock.release()

    t = threading.Thread(target=_sync, daemon=True)
    t.start()


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
    logger.info(f"Saved checkpoint: {path}")

    if tag != "latest":
        latest_path = os.path.join(save_dir, "checkpoint_latest.pt")
        torch.save(checkpoint, latest_path)
        logger.info(f"Saved latest mirror: {latest_path}")

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
