# Binnet-CUDA
Experiments with _Binarized Neural Networks_ in _Pytorch_.

The code provides basic tools for building binarized neural network as well
as custom CUDA kernel for faster inference. It contains most of the
ideas introduced in
[Binarized Neural Networks paper](https://papers.nips.cc/paper/2016/file/d8330f857a17c53d217014ee776bfd50-Paper.pdf).

The only layer available right now is `BinaryLinear` which performs
binarized version of `torch.nn.Linear`. The optimized forward pass kernel
is available via `use_xor_kernel` argument.
The kernel implementation is quite naive and will be optimized in the future.


## Install
The code requires CUDA 10.2+.

1. Install Python dependencies:
```shell
pip install -r requirements.txt
```
2. Install optimized forward pass CUDA kernel for `BinaryLinear`:
```
cd cuda && pip install .
```
If this fails you can try to explicitly specify the compiler you want to use via environment
variable `CXX`, e.g.:
```shell
CXX=g++ pip install .
```

## Examples
The `experiments` directory contains simple usage examples.

## Benchmarks
TBA




