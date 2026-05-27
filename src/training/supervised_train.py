"""Phase 1: Supervised pretraining.

Trains ChessNet on a dataset of 30M+ positions from master-level games.

Supports:
  - Mixed-precision (FP16) training
  - Checkpoint every N steps
  - Resume from latest checkpoint
  - Cosine LR decay
"""
import time
import logging
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from src.config import ChessConfig
from src.model.chess_net import ChessNet
from src.model.losses import combined_loss
from src.data.chess_dataset import ChessPositionDataset, RandomSliceDataset
from src.utils.checkpoint import save_checkpoint, load_checkpoint, find_latest_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_supervised(config: ChessConfig, resume: bool = True):
    """Run supervised training."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model = ChessNet(config.model).to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        eps=config.training.adam_epsilon,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.training.cosine_decay_steps,
        eta_min=config.training.min_learning_rate,
    )

    scaler = GradScaler(enabled=(config.training.mixed_precision == "fp16"))

    dataset = ChessPositionDataset(config.paths.supervised_data_path)
    epoch_dataset = RandomSliceDataset(dataset, config.training.max_position_per_epoch)
    dataloader = DataLoader(
        epoch_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    start_step = 0
    start_epoch = 0
    best_loss = float("inf")

    if resume:
        latest_path = find_latest_checkpoint(config.paths.checkpoint_dir)
        if latest_path:
            state = load_checkpoint(latest_path, model, optimizer, scheduler, device)
            start_step = state.get("step", 0)
            start_epoch = state.get("epoch", 0)
            best_loss = state.get("loss", float("inf"))
            logger.info(f"Resumed from {latest_path} (step={start_step}, epoch={start_epoch})")

    model.train()
    global_step = start_step

    for epoch in range(start_epoch, 1000):
        epoch_loss = 0.0
        epoch_policy_acc = 0.0
        batch_count = 0
        epoch_start = time.time()

        for batch in dataloader:
            X = batch["X"].to(device, non_blocking=True)
            y_policy = batch["y_policy"].to(device, non_blocking=True)
            y_value = batch["y_value"].to(device, non_blocking=True)
            legal_masks = batch["legal_mask"].to(device, non_blocking=True)

            with autocast(enabled=(config.training.mixed_precision == "fp16")):
                policy_logits, value_pred = model(X)
                loss = combined_loss(
                    policy_logits, y_policy, value_pred, y_value,
                    legal_masks, config.training,
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            scheduler.step()
            global_step += 1

            epoch_loss += loss.item()
            with torch.no_grad():
                pred_moves = policy_logits.argmax(dim=-1)
                epoch_policy_acc += (pred_moves == y_policy).float().mean().item()
            batch_count += 1

            if global_step % config.training.checkpoint_every_n_steps == 0:
                avg_loss = epoch_loss / max(batch_count, 1)
                save_checkpoint(
                    model, optimizer, scheduler,
                    step=global_step, epoch=epoch, loss=avg_loss,
                    tag=f"step_{global_step}",
                )
                logger.info(f"Checkpoint saved at step {global_step}, loss={avg_loss:.4f}")

        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / max(batch_count, 1)
        avg_acc = epoch_policy_acc / max(batch_count, 1)
        logger.info(
            f"Epoch {epoch} | step {global_step} | "
            f"loss={avg_loss:.4f} | policy_acc={avg_acc:.4f} | "
            f"time={epoch_time:.1f}s"
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(model, optimizer, scheduler, step=global_step, epoch=epoch, loss=avg_loss, tag="best")

    logger.info("Training complete.")
