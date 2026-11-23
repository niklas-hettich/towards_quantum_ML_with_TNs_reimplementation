import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters

def get_flat_params(model: nn.Module) -> torch.Tensor:
    return parameters_to_vector(model.parameters())

def set_flat_params(model: nn.Module, flat_params: torch.Tensor):
    vector_to_parameters(flat_params, model.parameters())


class SPSAOptimizer:
    """
    SPSA optimizer with momentum following Huggins et al. (2019), Section 3.2.
    """
    def __init__(self, model: nn.Module, a: float = 28.0, b: float = 33.0, A: float = 74.1, s: float = 4.13, t: float = 0.658, gamma: float = 0.882):
        self.model = model
        self.a = a
        self.b = b
        self.A = A
        self.s = s
        self.t = t
        self.gamma = gamma

        num_params = get_flat_params(model).numel()
        self.v = torch.zeros(num_params, device=get_flat_params(model).device)
        self.epoch = 0


    def step(self, loss_fn, x_batch: torch.Tensor, y_batch: torch.Tensor, epoch: int) -> float:
        """
        Performs a single SPSA optimization step on a mini-batch.
        
        Args:
            loss_fn: A callable function that takes (model, x_batch, y_batch) and returns a scalar loss.
            x_batch: Batch of inputs (batch_size, 64)
            y_batch: Batch of labels (batch_size,)
            
        Returns:
            avg_loss: The estimated loss (average of L+ and L-).
        """

        alpha_k = self.a / ((epoch + 1 + self.A) ** self.s)
        beta_k = self.b / ((epoch + 1) ** self.t)

        lambda_old = get_flat_params(self.model)

        delta = (torch.randint(0, 2, size=lambda_old.shape, device=lambda_old.device) * 2 - 1).to(lambda_old.dtype)
        lambda_plus = lambda_old + alpha_k * delta
        set_flat_params(self.model, lambda_plus)
        with torch.no_grad():
            loss_plus = loss_fn(self.model, x_batch, y_batch).item()
        lambda_minus = lambda_old - alpha_k * delta
        set_flat_params(self.model, lambda_minus)
        with torch.no_grad():
            loss_minus = loss_fn(self.model, x_batch, y_batch).item()

        g = (loss_plus - loss_minus) / (2 * alpha_k)
        self.v = self.gamma * self.v - g * beta_k * delta
        lambda_new = lambda_old + self.v
        set_flat_params(self.model, lambda_new)

        return (loss_plus + loss_minus) / 2