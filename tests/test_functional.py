from typing import Optional

import pytest
import torch
import torch.nn.functional as F

from binnet.functional import QuantizeSignSTE, BinLinearXOR


def test_quantize_ste_forward():
    x = torch.tensor([0.5, -0.5])

    y = QuantizeSignSTE.apply(x)

    assert (y == torch.tensor([1, -1])).all()


@pytest.mark.parametrize(
    "x, expected_grad",
    [
        (torch.tensor([1.25], requires_grad=True), torch.tensor([0.0])),
        (torch.tensor([1.0], requires_grad=True), torch.tensor([1])),
        (torch.tensor([-1.0], requires_grad=True), torch.tensor([1])),
        (torch.tensor([-1.25], requires_grad=True), torch.tensor([0])),
        (torch.tensor([0.5], requires_grad=True), torch.tensor([1])),
        (torch.tensor([0.0], requires_grad=True), torch.tensor([1])),
    ],
)
def test_quantize_ste_backward(x: torch.Tensor, expected_grad: torch.Tensor):
    y = QuantizeSignSTE.apply(x)
    y.backward()

    assert (x.grad == expected_grad).all()


@pytest.mark.cuda
@pytest.mark.parametrize(
    "x, weight, bias",
    [
        (torch.ones(2, 3), torch.ones(4, 3), None),
        (torch.ones(5, 2, 3), torch.ones(4, 3), None),
        (torch.ones(5, 2, 3), torch.ones(4, 3), torch.ones(4)),
        (torch.ones(10, 100, 200), torch.ones(300, 200), torch.ones(300)),
        (
            torch.sign(torch.randn(10, 100, 200)),
            torch.sign(torch.randn(300, 200)),
            torch.sign(torch.randn(300)),
        ),
    ],
)
def test_bin_linear_xor_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
):
    x = x.cuda()
    weight = weight.cuda()
    if bias is not None:
        bias = bias.cuda()
    expected_result = F.linear(x, weight, bias)

    result = BinLinearXOR.apply(x, weight, bias)

    assert result.shape == expected_result.shape
    assert (result == expected_result).all()


@pytest.mark.cuda
@pytest.mark.parametrize(
    "x, weight, bias",
    [
        (
            torch.sign(torch.randn(10, 100, 200, requires_grad=True)),
            torch.sign(torch.randn(300, 200, requires_grad=True)),
            None,
        ),
        (
            torch.sign(torch.randn(100, 200, requires_grad=True)),
            torch.sign(torch.randn(300, 200, requires_grad=True)),
            torch.sign(torch.randn(300, requires_grad=True)),
        ),
        (
            torch.sign(torch.randn(10, 100, 200, requires_grad=True)),
            torch.sign(torch.randn(300, 200, requires_grad=True)),
            torch.sign(torch.randn(300, requires_grad=True)),
        ),
    ],
)
def test_bin_linear_xor_backward(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
):
    x, weight = x.cuda(), weight.cuda()
    x.retain_grad()
    weight.retain_grad()
    if bias is not None:
        bias = bias.cuda()
        bias.retain_grad()
    expected_result = F.linear(x, weight, bias)
    expected_result.sum().backward()
    expected_d_x, expected_d_weight = x.grad, weight.grad
    x.grad, weight.grad = None, None
    if bias is not None:
        expected_d_bias = bias.grad
        bias.grad = None
    else:
        expected_d_bias = None

    result = BinLinearXOR.apply(x, weight, bias)
    result.sum().backward()

    assert x.grad.shape == expected_d_x.shape  # type: ignore
    assert (x.grad == expected_d_x).all()
    assert weight.grad.shape == expected_d_weight.shape  # type: ignore
    assert (weight.grad == expected_d_weight).all()
    if bias is not None:
        assert bias.grad.shape == expected_d_bias.shape
        assert (bias.grad == expected_d_bias).all()
