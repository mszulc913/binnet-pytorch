from typing import Optional

import torch

def forward(
    x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]
) -> torch.Tensor: ...
