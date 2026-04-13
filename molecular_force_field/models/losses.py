"""Loss functions and weighted sum modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSELoss(nn.Module):
    """Root Mean Square Error loss function."""
    
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, y_pred, y_true):
        return torch.sqrt(self.mse(y_pred, y_true))


def weighted_mse_loss_stats(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return weighted MSE mean, summed loss, and summed normalizer."""
    loss = F.mse_loss(pred, target, reduction="none")
    if weights is None:
        loss_sum = loss.sum()
        normalizer = torch.tensor(float(loss.numel()), device=loss.device, dtype=loss.dtype)
        return loss_sum / normalizer.clamp_min(1e-12), loss_sum, normalizer
    weight_tensor = weights.to(device=loss.device, dtype=loss.dtype)
    while weight_tensor.dim() < loss.dim():
        weight_tensor = weight_tensor.unsqueeze(-1)
    expanded_weights = torch.ones_like(loss) * weight_tensor
    loss_sum = (loss * expanded_weights).sum()
    normalizer = expanded_weights.sum()
    return loss_sum / normalizer.clamp_min(1e-12), loss_sum, normalizer
