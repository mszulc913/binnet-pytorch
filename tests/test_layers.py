from typing import Type

import pytest
import torch
from torch.autograd import Function
from torch.autograd.function import FunctionCtx

from binnet.layers import BinaryLinear
from binnet.functional import QuantizeSignSTE


class MockQuantize(Function):
    @staticmethod
    def forward(ctx: FunctionCtx, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        ctx.save_for_backward(x)
        return torch.ones_like(x) * 10

    @staticmethod
    def backward(  # type: ignore
        ctx: FunctionCtx, grad_output: torch.Tensor
    ) -> torch.Tensor:
        return torch.ones_like(ctx.saved_tensors[0]) * 5  # type: ignore


@pytest.mark.parametrize(
    "quantize_fn, expected_forward, expected_weight_grad, expected_bias_grad",
    [
        (
            QuantizeSignSTE,
            torch.tensor([[-3]]),
            torch.tensor([[1.0, -1.0]]),
            torch.tensor([1]),
        ),
        (
            MockQuantize,
            torch.tensor([[10]]),
            torch.tensor([[5, 5]]),
            torch.tensor([5]),
        ),
    ],
)
def test_binary_linear(
    quantize_fn: Type[Function],
    expected_forward: torch.Tensor,
    expected_weight_grad: torch.Tensor,
    expected_bias_grad: torch.Tensor,
):
    x = torch.tensor([[1.0, -1.0]], requires_grad=True)
    layer = BinaryLinear(2, 1, quantize_fn=quantize_fn)
    layer.weight = torch.nn.Parameter(torch.tensor([[-0.5, 0.5]]))
    layer.bias = torch.nn.Parameter(torch.tensor([-0.5]))

    forward = layer.forward(x)
    forward.backward()

    assert forward.shape == expected_forward.shape
    assert (forward == expected_forward).all()
    assert layer.weight.grad.shape == expected_weight_grad.shape
    assert (layer.weight.grad == expected_weight_grad).all()
    assert layer.bias.grad.shape == expected_bias_grad.shape
    assert (layer.bias.grad == expected_bias_grad).all()


@pytest.mark.cuda
def test_binary_linear_xor():
    x = torch.tensor([[1.0, -1.0]], requires_grad=True).cuda()
    layer = BinaryLinear(2, 1, use_xor_kernel=True)
    layer.weight = torch.nn.Parameter(torch.tensor([[-0.5, 0.5]]).cuda())
    layer.bias = torch.nn.Parameter(torch.tensor([-0.5]).cuda())

    forward = layer.forward(x).cpu()
    forward.backward()

    assert forward.shape == (1, 1)
    assert (forward == torch.tensor([[-3]])).all()
    assert layer.weight.grad.shape == (1, 2)
    assert (layer.weight.grad.cpu() == torch.tensor([[1.0, -1.0]])).all()
    assert layer.bias.shape == (1,)
    assert (layer.bias.grad.cpu() == torch.tensor([[1]])).all()
