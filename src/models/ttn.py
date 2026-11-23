import torch
import torch.nn as nn
from src.models.unitaries import UnitaryBlock
from src.feature_map import feature_map
from src.noise import apply_cptp_channel



class TreeTensorNetwork(nn.Module):
    """
    Tree Tensor Network for 64 inputs -> 1 output.
    Uses 63 unitary blocks arranged in a binary tree structure.
    """
    def __init__(self):
        super().__init__()
        self.gates = nn.ModuleList([UnitaryBlock() for _ in range(63)])

    def forward(self, x: torch.Tensor, kraus_ops: list[list[torch.Tensor]] = None) -> torch.Tensor:
        """
        Processes the input through the TTN.
        
        Args:
            x: Input tensor of shape (batch_size, 64)
            
        Returns:
            rho: The final density matrix of the single remaining qubit of shape (batch_size, 2, 2)
        """
        states = feature_map(x)
        
        states_ket = states.unsqueeze(-1)          # (batch size, 64, 2, 1)
        states_bra = torch.conj(states.unsqueeze(-2)) # (batch size, 64, 1, 2)
        rhos = (states_ket @ states_bra).to(torch.complex128)    # (batch size, 64, 2, 2)
        
        nodes = [rhos[:, i, :, :] for i in range(64)]
        
        gate_idx = 0
        
        assert((len(nodes)%2)==0)
        
        while len(nodes) > 1:

            temp_res_list = []

            for j in range(0,len(nodes),2):

                U = self.gates[gate_idx]()
                temp_res = apply_node(nodes[j], nodes[j+1], U, kraus_ops=kraus_ops)
                gate_idx += 1

                temp_res_list.append(temp_res)

            nodes = temp_res_list


        return nodes[0]




def apply_node(rho_left: torch.Tensor, rho_right: torch.Tensor, U: torch.Tensor, kraus_ops: list[torch.Tensor] = None) -> torch.Tensor:
    """
    Applies gate to 2 unentangled qubits (represented as density matrices) and traces out the right qubit.
    """

    if kraus_ops is not None:
        for kraus_op in kraus_ops:
            rho_left = apply_cptp_channel(rho_left, kraus_op)
            rho_right = apply_cptp_channel(rho_right, kraus_op)

    rho_in = torch.einsum('bij,bkl->bikjl', rho_left, rho_right).reshape(-1, 4, 4)

    U_dagger = torch.conj(U.T) 
    rho_out = torch.einsum('ij,bjk,kl->bil', U, rho_in, U_dagger)


    rho_reduced = torch.zeros((rho_out.shape[0], 2, 2), dtype=torch.complex128, device=rho_out.device)
    rho_reduced[:, 0, 0] = rho_out[:, 0, 0] + rho_out[:, 1, 1]
    rho_reduced[:, 0, 1] = rho_out[:, 0, 2] + rho_out[:, 1, 3]
    rho_reduced[:, 1, 0] = rho_out[:, 2, 0] + rho_out[:, 3, 1]
    rho_reduced[:, 1, 1] = rho_out[:, 2, 2] + rho_out[:, 3, 3]

    return rho_reduced