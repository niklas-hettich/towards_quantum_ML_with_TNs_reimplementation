import math
import torch


def feature_map(x: torch.Tensor) -> torch.Tensor:
    """
    Maps classical normalized input features into single-qubit quantum states.

    x: Tensor of shape (batch_size, N) with values in the range [0, 1], where N is the number of features (e.g., 64 pixels for 8x8 images).

    Returns: Tensor of shape (batch_size, N, 2), where the last dimension holds the amplitudes [cos(theta), sin(theta)] for state |0> and |1>.
    """
    
    theta = 0.5 * math.pi * x

    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    qubit_states = torch.stack([cos_theta, sin_theta], dim=-1)

    return qubit_states
