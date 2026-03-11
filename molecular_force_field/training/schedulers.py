"""Learning rate schedulers for training."""

from torch.optim.lr_scheduler import SequentialLR, LambdaLR, StepLR


def create_scheduler(optimizer, warmup_batches=1000, patience_opim=1000,
                     gamma_value=0.98, initial_learning_rate_for_weight=0.1):
    """
    Create learning rate scheduler with warmup and step decay.
    
    Args:
        optimizer: Optimizer
        warmup_batches: Number of warmup batches
        patience_opim: Patience for step scheduler
        gamma_value: Learning rate decay factor
        initial_learning_rate_for_weight: Initial learning rate ratio
        
    Returns:
        SequentialLR scheduler
    """
    milestones = [warmup_batches]
    
    def warmup_lambda(current_step):
        progress = min(current_step / warmup_batches, 1.0)
        initial_ratio = initial_learning_rate_for_weight
        return initial_ratio + (1 - initial_ratio) * progress
    
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
    step_scheduler = StepLR(
        optimizer,
        step_size=patience_opim,
        gamma=gamma_value
    )
    
    return SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, step_scheduler],
        milestones=milestones
    )