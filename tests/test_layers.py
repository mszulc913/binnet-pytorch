from typing import Type

import pytest
import torch
from torch.autograd import Function
from torch.autograd.function import FunctionCtx

from binnet.layers import BinaryLinear
from binnet.quantize import QuantizeSignSTE


class MockQuantize(Function):
    @staticmethod
    def forward(ctx: FunctionCtx, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        ctx.save_for_backward(x)
        return torch.ones_like(x) * 10

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: torch.Tensor) -> torch.Tensor:  # type: ignore
        return torch.ones_like(ctx.saved_tensors[0]) * 5  # type: ignore


@pytest.mark.parametrize(
    "quantize_fn, expected_forward, expected_weight_grad, expected_bias_grad",
    [
        (
            QuantizeSignSTE,
            torch.tensor([[-2]]),
            torch.tensor([[0.5, -0.5]]),
            torch.tensor([[1]]),
        ),
        (
            MockQuantize,
            torch.tensor([[10]]),
            torch.tensor([[5, 5]]),
            torch.tensor([[5]]),
        ),
    ],
)
def test_binary_linear(
    quantize_fn: Type[Function],
    expected_forward: torch.Tensor,
    expected_weight_grad: torch.Tensor,
    expected_bias_grad: torch.Tensor,
):
    x = torch.tensor([[0.5, -0.5]], requires_grad=True)
    layer = BinaryLinear(2, 1, quantize_fn=quantize_fn)
    layer.weight = torch.nn.Parameter(torch.tensor([[-0.5, 0.5]]))
    layer.bias = torch.nn.Parameter(torch.tensor([[-0.5]]))

    forward = layer.forward(x)
    forward.backward()

    assert (forward == expected_forward).all()
    assert (layer.weight.grad == expected_weight_grad).all()
    assert (layer.bias.grad == expected_bias_grad).all()
