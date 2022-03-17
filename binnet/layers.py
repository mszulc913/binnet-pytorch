from typing import Type, Optional

import torch
import torch.nn.functional as F
from torch.autograd import Function
from binnet.functional import BinLinearXNOR, QuantizeSignSTE


class BinaryLinear(torch.nn.Linear):
    def __init__(
        self,
        *args,
        use_xnor_kernel: bool = False,
        quantize_fn: Type[Function] = QuantizeSignSTE,
        **kwargs
    ):
        """A binary linear transformation.

        This class extends `torch.nn.Linear` to the binary fashion, i.e. it
        quantizes (binarizes) the weights before matrix multiplication.

        Currently, only two-dimensional input matrices are supported.

        Warning: This layer does not check if inputs are already binarized.
        If they are not, the results will be incorrect when `use_xnor_kernel`
        is set to `False`.

        :param args: Positional arguments to the `torch.nn.Linear`.
        :param use_xnor_kernel: True if optimized CUDA kernel should be used.
        This only works on CUDA tensors and CUDA has to be available.
        :param quantize_fn: Quantization function to use to transform the weights.
        :param kwargs: Keyword arguments to the `torch.nn.Linear`.
        :raises: RuntimeError if `use_xnor_kernel` is set to `True`
        and CUDA isn't available.
        """
        super().__init__(*args, **kwargs)
        if use_xnor_kernel and not torch.cuda.is_available():
            raise RuntimeError(
                "`use_xnor_kernel` is set to True, but CUDA isn't available."
            )

        self.use_xnor_kernel = use_xnor_kernel
        self.quantize_fn = quantize_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_xnor_kernel:
            return BinLinearXNOR.apply(x, self.weight_bin, self.bias_bin)
        else:
            return F.linear(x, self.weight_bin, self.bias_bin)

    @property
    def weight_bin(self) -> torch.Tensor:
        """
        :return: Binarized `weights` matrix.
        """
        return self.quantize_fn.apply(self.weight)

    @property
    def bias_bin(self) -> Optional[torch.Tensor]:
        """
        :return: Binarized `bias` matrix (of the float type) if
        `bias` is `True`, `None` otherwise.
        """
        if self.bias is not None:
            return self.quantize_fn.apply(self.bias)
        else:
            return None
