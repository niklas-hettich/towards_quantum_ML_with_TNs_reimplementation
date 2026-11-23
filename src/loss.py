import torch


def compute_probabilities(rho: torch.Tensor) -> torch.Tensor:
    """
    Extracts the measurement probabilities for class 0 and class 1 
    from the final density matrix.
    
    Args:
        rho: Density matrix of shape (batch_size, 2, 2)
        
    Returns:
        probs: Real tensor of shape (batch_size, 2) containing probabilities.
    """
    p0 = rho[:, 0, 0].real
    p1 = rho[:, 1, 1].real
    
    return torch.stack([p0, p1], dim=-1)


def ttn_loss(probs: torch.Tensor, y_true: torch.Tensor, margin: float = 0.234, eta: float = 5.59) -> torch.Tensor:
    """
    Computes the margin-based loss from Huggins et al. (2019).
    
    Args:
        probs: Probabilities of shape (batch_size, 2)
        y_true: True class labels of shape (batch_size,), containing 0 or 1
        margin: Hyperparameter lambda
        eta: Hyperparameter eta
        
    Returns:
        loss: Scalar tensor representing the mean loss over the batch.
    """

    batch_idx = torch.arange(probs.size(0))
    
    p_true = probs[batch_idx, y_true]
    p_false = probs[batch_idx, (y_true + 1) % 2]
    
    loss = torch.relu(p_false - p_true + margin) ** eta
    
    return loss.mean()