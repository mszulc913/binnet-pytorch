import torch
from torch.autograd import Function
from torch.autograd.function import FunctionCtx


class QuantizeSignSTE(Function):
    @staticmethod
    def forward(ctx: FunctionCtx, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        ctx.save_for_backward(x)
        return torch.sign(x)

    @staticmethod
    def backward(  # type: ignore
        ctx: FunctionCtx, grad_output: torch.Tensor
    ) -> torch.Tensor:
        x = ctx.saved_tensors[0]  # type: ignore
        out = torch.ones_like(x)
        out[torch.abs(x) > 1] = 0
        return grad_output * out
