import logging
from typing import Tuple

import torch
from torch.utils import benchmark
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
import pytorch_lightning as pl

from experiments.mnist_mlp import BinMLPClassifier


def benchmark_matrix_multiply(matrix_size: int, n: int):
    num_threads = torch.get_num_threads()
    matrix1 = torch.randn((matrix_size, matrix_size), device="cuda")
    matrix2 = torch.randn((matrix_size, matrix_size), device="cuda")

    print(
        f"Measuring matrix multiplication"
        f" ({(matrix_size, matrix_size)} x {matrix_size, matrix_size})\n"
    )

    t_cublas = benchmark.Timer(
        stmt="matrix1 @ matrix2",
        num_threads=num_threads,
        label="cuBLAS matrix multiplication",
        globals={"matrix1": matrix1, "matrix2": matrix2},
    )

    print(t_cublas.timeit(n))

    print("-" * 30)

    t_xnor = benchmark.Timer(
        stmt="bin_matmul(matrix1, matrix2)",
        num_threads=num_threads,
        label="Binary XNOR matrix multiplication",
        setup="from binnet.functional import bin_matmul",
        globals={"matrix1": matrix1, "matrix2": matrix2},
    )

    print(t_xnor.timeit(n))


def benchmark_mnist_test_set(hidden_sizes: Tuple[int, ...], n: int):
    print(
        f"Measuring MLP ({hidden_sizes}) on the MNIST test set.\n",
    )
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    batch_size = 100
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            torch.nn.Flatten(0, -1),
        ]
    )
    model_cublas = BinMLPClassifier(28 * 28, hidden_sizes, 10, 1e-4, False)
    model_xnor = BinMLPClassifier(28 * 28, hidden_sizes, 10, 1e-4, True)
    mnist_test = MNIST("", train=False, download=True, transform=transform)
    test_loader = DataLoader(mnist_test, batch_size=batch_size, num_workers=4)
    trainer = pl.Trainer(
        max_epochs=1, gpus=1, enable_progress_bar=False, enable_model_summary=False
    )

    t_cublas = benchmark.Timer(
        stmt="trainer.test(model, dataloaders=test_loader, verbose=False)",
        num_threads=torch.get_num_threads(),
        label="MLP with cuBLAS kernel",
        globals={"trainer": trainer, "test_loader": test_loader, "model": model_cublas},
    )

    print(t_cublas.timeit(n))

    print("-" * 30)

    t_xnor = benchmark.Timer(
        stmt="trainer.test(model, dataloaders=test_loader, verbose=False)",
        num_threads=torch.get_num_threads(),
        label="MLP with XNOR kernel",
        globals={"trainer": trainer, "test_loader": test_loader, "model": model_xnor},
    )

    print(t_xnor.timeit(n))


def run_benchmark():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")
    benchmark_matrix_multiply(8192, 100)
    print("\n" + "=" * 30 + "\n")
    benchmark_mnist_test_set((4096, 4096, 4096), 100)


run_benchmark()
