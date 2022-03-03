import pytest
import torch

from binnet.quantize import QuantizeSignSTE


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
