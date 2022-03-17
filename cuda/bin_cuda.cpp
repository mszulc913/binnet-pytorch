#include <torch/extension.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")


torch::Tensor bin_matmul_cuda(const torch::Tensor mat1, const torch::Tensor mat2);


torch::Tensor bin_linear(const torch::Tensor input, const torch::Tensor weight, const c10::optional<torch::Tensor> bias_opt) {
    CHECK_CUDA(input);
    CHECK_CUDA(weight);
    if (bias_opt.has_value()) {
        CHECK_CUDA(bias_opt.value());
    }
    auto result = bin_matmul_cuda(input, weight.t());
    if (bias_opt.has_value()) {
        result.add_(bias_opt.value());
    }
    return result;
}

torch::Tensor bin_matmul(const torch::Tensor mat1, const torch::Tensor mat2) {
    CHECK_CUDA(mat1);
    CHECK_CUDA(mat2);

    return bin_matmul_cuda(mat1, mat2);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bin_linear", &bin_linear, "Binary linear projection (CUDA)");
    m.def("bin_matmul", &bin_matmul, "Binary matrix multiplication (CUDA)");
}