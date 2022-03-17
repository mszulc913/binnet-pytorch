from typing import List, Optional, Tuple, cast

import bin_cuda
import torch
from torch.autograd import Function
from torch.autograd.function import FunctionCtx


class BinLinearXNOR(Function):
    """Optimized binary linear transformation for CUDA supporting devices."""

    @staticmethod
    def forward(  # type: ignore
        ctx: FunctionCtx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if bias is not None:
            ctx.save_for_backward(x, weight, bias)
        else:
            ctx.save_for_backward(x, weight)
        return bin_cuda.bin_linear(x, weight, bias)

    @staticmethod
    def backward(  # type: ignore
        ctx: FunctionCtx, grad_output: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        saved_tensors = cast(List[torch.Tensor], ctx.saved_tensors)  # type: ignore

        x = saved_tensors[0]
        weight = saved_tensors[1]
        if len(saved_tensors) == 3:
            bias = saved_tensors[2]
            d_bias = grad_output * torch.ones_like(bias)
        else:
            bias, d_bias = None, None

        d_x = grad_output @ weight
        d_weight = (x.transpose(-2, -1) @ grad_output).transpose(-2, -1)

        return d_x, d_weight, d_bias


class QuantizeSignSTE(Function):
    """Sign quantization operator.

    Transforms input float value to 1.0 if `x >= 0`, and to -1.0 otherwise.
    """

    @staticmethod
    def forward(ctx: FunctionCtx, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        ctx.save_for_backward(x)
        return torch.sign(x)

    @staticmethod
    def backward(  # type: ignore
        ctx: FunctionCtx, grad_output: torch.Tensor
    ) -> torch.Tensor:
        x = cast(torch.Tensor, ctx.saved_tensors[0])  # type: ignore
        out = torch.ones_like(x)
        out[torch.abs(x) > 1] = 0
        return grad_output * out


def bin_matmul(mat1: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
    """Binarized matrix product of two tensors.
    .
        Currently, only two-dimensional inputs are supported.

        :param mat1: The first tensor to be multiplied.
        :param mat2: The second tensor to be multiplied.
        :return: Tensor with the result of the operation.
    """
    return bin_cuda.bin_matmul(mat1, mat2)
