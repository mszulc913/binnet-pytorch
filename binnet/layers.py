from typing import Type

import torch
import torch.nn.functional as F
from torch.autograd import Function

from binnet.quantize import QuantizeSignSTE


class BinaryLinear(torch.nn.Linear):
    def __init__(self, *args, quantize_fn: Type[Function] = QuantizeSignSTE, **kwargs):
        super().__init__(*args, **kwargs)
        self.quantize_fn = quantize_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight_bin, self.bias_bin)

    @property
    def weight_bin(self) -> torch.Tensor:
        return self.quantize_fn.apply(self.weight)

    @property
    def bias_bin(self) -> torch.Tensor:
        return self.quantize_fn.apply(self.bias)
