from typing import Optional

import torch

def bin_linear(
    x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]
) -> torch.Tensor: ...


def bin_matmul(
    mat1: torch.Tensor, mat2: torch.Tensor
) -> torch.Tensor: ...