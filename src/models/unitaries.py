import torch
import torch.nn as nn

class UnitaryBlock(nn.Module):
    """
    Parametrized 2-qubit unitary gate U = exp(iH).
    Follows the setup from Huggins et al. (2019) where H is a 4x4 Hermitian matrix.
    A 4x4 Hermitian matrix has exactly 16 free real parameters.
    """
    def __init__(self):
        super().__init__()
        self.params = nn.Parameter(torch.randn(16) * 0.01)
        
        self.row_idx = [0, 0, 0, 1, 1, 2]
        self.col_idx = [1, 2, 3, 2, 3, 3]

    def forward(self) -> torch.Tensor:
        """
        Constructs the 4x4 unitary matrix U = exp(iH) from the current parameters.
        
        Returns:
            U: A complex tensor of shape (4, 4).
        """
        diagonal = self.params[:4]
        re_upper_triangular = self.params[4:10]
        im_upper_triangular = self.params[10:]

        H = torch.zeros((4, 4), dtype=torch.complex128, device=self.params.device)

        complex_diagonal = torch.complex(diagonal, torch.zeros_like(diagonal))
        complex_upper_triangular = torch.complex(re_upper_triangular, im_upper_triangular)

        for i in range (0,4):
            H[i, i] = complex_diagonal[i]

        for i in range (6):
            H[self.row_idx[i], self.col_idx[i]] = complex_upper_triangular[i]
            H[self.col_idx[i], self.row_idx[i]] = torch.conj(complex_upper_triangular[i])

        
        U = torch.linalg.matrix_exp(1j*H)
        
        return U