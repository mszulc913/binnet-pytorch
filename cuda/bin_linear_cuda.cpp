#include <torch/extension.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")


torch::Tensor bin_linear_cuda(const torch::Tensor input, const torch::Tensor weight, const c10::optional<torch::Tensor> bias_opt);


torch::Tensor bin_linear(const torch::Tensor input, const torch::Tensor weight, const c10::optional<torch::Tensor> bias_opt) {
    CHECK_CUDA(input);
    CHECK_CUDA(weight);
    if (bias_opt.has_value()) {
        CHECK_CUDA(bias_opt.value());
    }
    return bin_linear_cuda(input, weight, bias_opt);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &bin_linear, "Binary Matrix Multiplication (CUDA)");
}