import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from collections.abc import Callable

def get_flat_params(model: nn.Module) -> torch.Tensor:
    """
    Flattens all trainable parameters of a model into a single 1D tensor.

    Args:
        model: PyTorch model whose parameters are extracted.

    Returns:
        flat_params: 1D tensor containing all concatenated model parameters.
    """
    
    return parameters_to_vector(model.parameters())

def set_flat_params(model: nn.Module, flat_params: torch.Tensor) -> None:
    """
    Writes values from a 1D parameter vector back into the model's parameters.

    Args:
        model: PyTorch model to update in-place.
        flat_params: 1D tensor with the same total number of elements as
            the model's parameters.
    """

    vector_to_parameters(flat_params, model.parameters())


class SPSAOptimizer:
    """
    SPSA optimizer with momentum following Huggins et al. (2019), Section 3.2.

    Attributes:
        model (nn.Module): The model being optimized.
        a (float): Numerator scaling constant for the perturbation schedule alpha_k.
        b (float): Numerator scaling constant for the step-size schedule beta_k.
        A (float): Stability offset constant for the alpha_k schedule.
        s (float): Decay exponent for the alpha_k schedule.
        t (float): Decay exponent for the beta_k schedule.
        gamma (float): Momentum decay factor.
        v (torch.Tensor): 1D momentum velocity vector matching the flat parameter size.
    """
    
    def __init__(self, model: nn.Module, a: float = 28.0, b: float = 33.0, A: float = 74.1, s: float = 4.13, t: float = 0.658, gamma: float = 0.882) -> None:
        """
        Initializes the SPSA optimizer with the hyperparameters from Huggins et al. (2019).

        Args:
            model: Model whose parameters will be optimized.
            a: Scaling constant for alpha_k. Defaults to 28.0.
            b: Scaling constant for beta_k. Defaults to 33.0.
            A: Offset constant for alpha_k. Defaults to 74.1.
            s: Decay exponent for alpha_k. Defaults to 4.13.
            t: Decay exponent for beta_k. Defaults to 0.658.
            gamma: Momentum factor. Defaults to 0.882.
        """

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


    def step(self,
        loss_fn: Callable[[nn.Module, torch.Tensor, torch.Tensor], torch.Tensor],
        x_batch: torch.Tensor,
        y_batch: torch.Tensor,
        epoch: int
        ) -> float:
        """
        Performs a single SPSA optimization step on a mini-batch.

        Args:
            loss_fn: Callable taking (model, x_batch, y_batch) and returning a scalar loss.
            x_batch: Batch of inputs of shape (batch_size, 64).
            y_batch: Batch of binary labels of shape (batch_size,).
            epoch: Zero-indexed current epoch k used to compute alpha_k and beta_k.

        Returns:
            avg_loss: Average of the two perturbed loss evaluations (loss_plus + loss_minus) / 2.
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