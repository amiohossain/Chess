"""Loss functions for chess model training.

Combined loss = 0.5 * policy_ce + 0.3 * value_mse + 0.2 * top10_accuracy_reg

Trap training additionally applies trap_loss_weight on trap position batches.
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

    # Top-10 accuracy regularizer: encourages high probability on the correct top-10
    top10_mask = _top10_mask(masked_logits, policy_targets)
    top10_reg = (masked_logits * top10_mask).sum(dim=1) - masked_logits.gather(1, policy_targets.unsqueeze(1)).squeeze(1)
    top10_reg = top10_reg.mean()

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
    """Create a mask encouraging correct move to be in the top 10 logits."""
    batch_size = logits.size(0)
    mask = torch.zeros_like(logits)

    for i in range(batch_size):
        top10_indices = logits[i].topk(10).indices
        for idx in top10_indices:
            if idx != targets[i]:
                mask[i, idx] = -1.0
        mask[i, targets[i]] = 1.0

    return mask
