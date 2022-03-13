from setuptools import setup  # type: ignore
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="bin_linear_cuda",
    ext_modules=[
        CUDAExtension(
            "bin_linear_cuda",
            [
                "bin_linear_cuda.cpp",
                "bin_linear_cuda_kernel.cu",
            ],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
