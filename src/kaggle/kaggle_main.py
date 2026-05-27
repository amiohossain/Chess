"""Kaggle notebook entry point.

Handles:
  - Determining which phase to run based on checkpoint availability
  - Setting device and paths for Kaggle environment
  - Running the appropriate training phase
  - Saving checkpoints to Kaggle Dataset output

Usage on Kaggle:
  import sys
  sys.path.append("/kaggle/working")
  from src.kaggle.kaggle_main import run_session
  run_session()
"""
import os
import logging
import torch

from src.config import ChessConfig, PathConfig
from src.training.supervised_train import train_supervised
from src.training.trap_finetune import train_trap_specialization
from src.training.self_play_loop import run_self_play_session
from src.utils.checkpoint import find_latest_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def detect_phase(config: ChessConfig) -> str:
    """Determine which phase to run based on available checkpoints."""
    latest = find_latest_checkpoint(config.paths.checkpoint_dir)
    if not latest:
        logger.info("No checkpoints found — starting Phase 1 (supervised)")
        return "supervised"
    checkpoint = torch.load(latest, map_location="cpu", weights_only=True)
    tag = checkpoint.get("tag", "")
    step = checkpoint.get("step", 0)
    epoch = checkpoint.get("epoch", 0)
    loss = checkpoint.get("loss", "?")
    logger.info(f"Latest checkpoint: {latest} (tag={tag}, step={step}, epoch={epoch}, loss={loss})")
    if "trap" in tag:
        logger.info("Trap checkpoint found → Phase 3 (self-play)")
        return "self_play"
    elif "self_play" in tag:
        logger.info("Self-play checkpoint found → Phase 3 (self-play, continued)")
        return "self_play"
    else:
        logger.info("Supervised checkpoint found → Phase 2 (trap finetune)")
        return "trap"


def setup_kaggle_config() -> ChessConfig:
    """Create config with Kaggle-appropriate paths."""
    config = ChessConfig()
    config.paths = PathConfig(
        checkpoint_dir="/kaggle/working/checkpoints",
        data_dir="/kaggle/input/chess-training-data-2013",
        trap_data_path="/kaggle/input/chess-data/trap_positions.h5",
        supervised_data_path="/kaggle/working/supervised_positions.h5",
        log_dir="/kaggle/working/logs",
    )
    config.training.num_workers = 2
    return config


def run_session(phase: str = None):
    """Main entry point for a Kaggle training session."""
    config = setup_kaggle_config()

    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(config.paths.checkpoint_dir, exist_ok=True)

    if phase is None:
        phase = detect_phase(config)
        logger.info(f"Auto-detected phase: {phase}")
    else:
        logger.info(f"Starting phase: {phase}")

    if phase == "supervised":
        train_supervised(config, resume=True)
    elif phase == "trap":
        train_trap_specialization(config, resume=True)
    elif phase == "self_play":
        run_self_play_session(config)
    else:
        raise ValueError(f"Unknown phase: {phase}")

    logger.info("Session complete!")
