# Binnet-CUDA
Experiments with _Binarized Neural Networks_ in _Pytorch_.

The code provides easy and clean implementation of Binarized Neural Netwoks with custom CUDA kernel for forward pass.
It contains the main ideas introduced in
[Binarized Neural Networks paper](https://papers.nips.cc/paper/2016/file/d8330f857a17c53d217014ee776bfd50-Paper.pdf).

The only layer available right now is `BinaryLinear` which performs
binarized version of `torch.nn.Linear`. The optimized forward pass kernel
is available via `use_xnor_kernel` argument.


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
[experiments/mnist_mlp.py](experiments/mnist_mlp.py) contains example experiment with MLP on MNIST.

## Benchmarks
Benchmarks were conducted on `Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz` / `GeForce GTX 1650 Mobile`.
Custom CUDA XNOR kernel was compared to cuBLAS kernel on the following problems:
- (8196, 8916) x (8196, 8916) matrix multiplication
- MLP ((4096, 4096, 4096) hidden units) inference on the MNIST test set (batch size = 100); first
layer and softmax projection layers were not binarized

Each experiment was repeated 100 times with `torch.utils.benchmark`.

| Problem               | cuBLAS    | XNOR      |
|-----------------------|-----------|-----------|
| Matrix multiplication | 425.21 ms | 155.33 ms |
| MLP on MNIST test     | 772.96 ms | 690.84 ms |

Full report is available in the `experiments` folder.


Benchmarks were produced with [experiments/benchmark.py](experiments/benchmark.py).
