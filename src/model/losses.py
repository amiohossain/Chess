"""Loss functions for chess model training.

Combined loss = 0.5 * policy_ce + 0.3 * value_mse + 0.2 * top10_margin

Trap training additionally applies trap_loss_weight on trap position batches.

top10_margin is a margin-based penalty (bounded >= 0): it penalizes top-10
distractor moves whose logit exceeds the target move's logit.
"""
import torch
import torch.nn.functional as F
from src.config import TrainingConfig


def combined_loss(
    policy_logits: torch.Tensor,
    policy_targets: torch.Tensor,
    value_pred: torch.Tensor,
    value_targets: torch.Tensor,
    legal_masks: torch.Tensor,
    config: TrainingConfig,
    trap_weights: torch.Tensor = None,
) -> torch.Tensor:
    """Compute the combined loss for chess model training.

    Args:
        policy_logits: (batch, 4096) raw logits from model
        policy_targets: (batch,) — index of the target move
        value_pred: (batch, 1) value prediction in [-1, 1]
        value_targets: (batch, 1) ground truth in {-1, 0, 1}
        legal_masks: (batch, 4096) binary masks for legal moves
        config: TrainingConfig with loss weights
        trap_weights: (batch,) optional per-sample trap importance weight

    Returns:
        Scalar loss tensor
    """
    # Policy loss: cross-entropy on legal moves only
    masked_logits = policy_logits + (1.0 - legal_masks) * -1e9
    policy_loss = F.cross_entropy(masked_logits, policy_targets, reduction='none')

    # Value loss: MSE
    value_loss = F.mse_loss(value_pred.squeeze(-1), value_targets.squeeze(-1), reduction='none')

    # Top-10 accuracy regularizer: penalize distractors that beat the target
    # Bounded below by 0 — no more diverging loss
    top10_mask = _top10_mask(masked_logits, policy_targets)
    target_logit = masked_logits.gather(1, policy_targets.unsqueeze(1))
    top10_reg = F.relu(masked_logits - target_logit).mul(top10_mask).sum(dim=1).mean()

    # Combine
    loss = (
        config.policy_weight * policy_loss.mean()
        + config.value_weight * value_loss.mean()
        + config.top10_reg_weight * top10_reg
    )

    # Apply trap weights if provided (during Phase 2 training)
    if trap_weights is not None and trap_weights.sum() > 0:
        trap_policy_loss = (policy_loss * trap_weights).mean()
        loss = loss + trap_weights.mean() * trap_policy_loss

    return loss


def _top10_mask(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Create a mask selecting non-target top-10 moves for margin penalty.

    Vectorized — no Python loops. Gathers top-10 indices for all batch
    elements at once via scatter_.
    """
    top10 = logits.topk(10, dim=1).indices  # (batch, 10)
    mask = torch.zeros_like(logits)
    # Mark all top-10 positions as 1
    mask.scatter_(1, top10, 1.0)
    # Zero out the target position so we only penalize distractors
    mask.scatter_(1, targets.unsqueeze(1), 0.0)
    return mask
