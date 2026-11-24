import torch
import math

def get_amplitude_damping_kraus(Tg: float, T1: float, device: torch.device) -> list[torch.Tensor]:
    """
    Generates Kraus operators for amplitude damping (energy relaxation to |0>).
    
    Args:
        Tg: Gate execution time (e.g., in microseconds).
        T1: Relaxation time of the qubit.
        device: Torch device (cpu/cuda).
        
    Returns:
        A list containing the two Kraus operators [M0, M1] as complex128 tensors.
    """
    p_a = 1 - math.exp(-Tg / T1)
    
    M_0 = torch.tensor([[1, 0], [0, math.sqrt(1 - p_a)]], dtype=torch.complex128, device=device)
    M_1 = torch.tensor([[0, math.sqrt(p_a)], [0, 0]], dtype=torch.complex128, device=device)
    
    return [M_0, M_1]


def get_dephasing_kraus(Tg: float, T2: float, device: torch.device) -> list[torch.Tensor]:
    """
    Generates Kraus operators for pure dephasing (loss of phase coherence).
    
    Args:
        Tg: Gate execution time.
        T2: Dephasing time of the qubit.
        device: Torch device.
        
    Returns:
        A list containing the three Kraus operators [M0, M1, M2] as complex128 tensors.
    """
    p_d = 1 - math.exp(-Tg / T2)
    
    M_0 = torch.tensor([[math.sqrt(1 - p_d), 0], [0, math.sqrt(1 - p_d)]], dtype=torch.complex128, device=device)
    M_1 = torch.tensor([[math.sqrt(p_d), 0], [0, 0]], dtype=torch.complex128, device=device)
    M_2 = torch.tensor([[0, 0], [0, math.sqrt(p_d)]], dtype=torch.complex128, device=device)
    
    return [M_0, M_1, M_2]


def apply_cptp_channel(rho: torch.Tensor, kraus_ops: list[torch.Tensor]) -> torch.Tensor:
    """
    Applies a quantum channel defined by a set of Kraus operators to a batch of density matrices.
    
    Args:
        rho: Batched density matrices of shape (batch_size, 2, 2)
        kraus_ops: List of Kraus operators [E_0, E_1, ...] of shape (2, 2)
        
    Returns:
        rho_noisy: The degraded density matrices of shape (batch_size, 2, 2)
    """
    rho_noisy = torch.zeros_like(rho)

    for op in kraus_ops:
        op_dagger = torch.conj(op.T)
        rho_noisy += torch.einsum('ij,bjk,kl->bil', op, rho, op_dagger)

    return rho_noisy