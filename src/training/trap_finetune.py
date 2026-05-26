"""Phase 2: Trap specialization fine-tuning.

Trains ChessNet on a 1:1 mix of trap positions and general chess positions.
Trap positions are priority-weighted for sampling and have 2x loss weight.
"""
import logging
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler

from src.config import ChessConfig
from src.model.chess_net import ChessNet
from src.model.losses import combined_loss
from src.data.chess_dataset import ChessPositionDataset
from src.data.trap_dataset import TrapDataset
from src.utils.checkpoint import save_checkpoint, load_checkpoint, find_latest_checkpoint

logger = logging.getLogger(__name__)


def train_trap_specialization(config: ChessConfig, resume: bool = True):
    """Fine-tune the model on trap positions."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model = ChessNet(config.model).to(device)

    latest_path = find_latest_checkpoint(config.paths.checkpoint_dir)
    if latest_path:
        state = load_checkpoint(latest_path, model, device=device)
        logger.info(f"Loaded base model from {latest_path}")
    else:
        logger.warning("No checkpoint found — training from scratch!")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate * 0.1,
        weight_decay=config.training.weight_decay,
    )
    scaler = GradScaler(enabled=(config.training.mixed_precision == "fp16"))

    general_dataset = ChessPositionDataset(config.paths.supervised_data_path, max_samples=500_000)
    trap_dataset = TrapDataset(config.paths.trap_data_path)

    trap_sampler = WeightedRandomSampler(
        weights=trap_dataset.sampling_weights,
        num_samples=min(len(trap_dataset), 200_000),
        replacement=True,
    )

    general_loader = DataLoader(
        general_dataset, batch_size=config.training.batch_size // 2,
        shuffle=True, num_workers=config.training.num_workers, pin_memory=True,
    )
    trap_loader = DataLoader(
        trap_dataset, batch_size=config.training.batch_size // 2,
        sampler=trap_sampler, num_workers=config.training.num_workers, pin_memory=True,
    )

    model.train()
    global_step = 0

    for epoch in range(10):
        epoch_loss = 0.0
        trap_acc = 0.0
        batch_count = 0

        for gen_batch, trap_batch in zip(general_loader, trap_loader):
            X = torch.cat([gen_batch["X"], trap_batch["X"]]).to(device, non_blocking=True)
            y_policy = torch.cat([gen_batch["y_policy"], trap_batch["y_policy"]]).to(device, non_blocking=True)
            y_value = torch.cat([gen_batch["y_value"], trap_batch["y_value"]]).to(device, non_blocking=True)
            legal_masks = torch.cat([gen_batch["legal_mask"], trap_batch["legal_mask"]]).to(device, non_blocking=True)

            trap_weights = torch.cat([
                torch.ones(len(gen_batch["X"])),
                torch.ones(len(trap_batch["X"])) * config.training.trap_loss_weight,
            ]).to(device, non_blocking=True)

            with autocast(device_type="cuda", enabled=(config.training.mixed_precision == "fp16")):
                policy_logits, value_pred = model(X)
                loss = combined_loss(
                    policy_logits, y_policy, value_pred, y_value,
                    legal_masks, config.training, trap_weights=trap_weights,
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1

            epoch_loss += loss.item()
            with torch.no_grad():
                pred = policy_logits.argmax(dim=-1)
                trap_pred = pred[len(gen_batch["X"]):]
                trap_target = y_policy[len(gen_batch["X"]):]
                trap_acc += (trap_pred == trap_target).float().mean().item()
            batch_count += 1

            if global_step % 1000 == 0:
                logger.info(
                    f"Trap step {global_step} | loss={epoch_loss/max(batch_count,1):.4f} | "
                    f"trap_acc={trap_acc/max(batch_count,1):.4f}"
                )

        avg_loss = epoch_loss / max(batch_count, 1)
        avg_trap_acc = trap_acc / max(batch_count, 1)
        logger.info(f"Trap Epoch {epoch} | loss={avg_loss:.4f} | trap_acc={avg_trap_acc:.4f}")
        save_checkpoint(model, optimizer, step=global_step, epoch=epoch, loss=avg_loss, tag="trap_phase")
