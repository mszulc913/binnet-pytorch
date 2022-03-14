#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

const auto ENCODE_SIZE = 32;


template <typename scalar_t>
__global__ void bin_linear_kernel(
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> input,
    const torch::PackedTensorAccessor<int32_t, 2, torch::RestrictPtrTraits, size_t> encoded_rows,
    const torch::PackedTensorAccessor<int32_t, 2, torch::RestrictPtrTraits, size_t> encoded_cols,
    const int m,
    const int n,
    const int k,
    int encoded_dim
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < m && col < k) {
        scalar_t val = 0;

        #pragma unroll
        for (int i = 0; i < encoded_dim; i++) {
            val += __popc(encoded_rows[row][i] ^ encoded_cols[i][col]);
        }
        input[row][col] += n - 2 * val;
    }
}


template <typename scalar_t>
__global__ void encode_rows_kernel(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> matrix,
    torch::PackedTensorAccessor<int32_t, 2, torch::RestrictPtrTraits, size_t> result,
    int n,
    int encode_size
) {
    int col = threadIdx.x * encode_size;
    int row = blockIdx.x;

    int32_t encoded_value = 0;

    #pragma unroll
    for (int i = 0; i < encode_size && col < n; i++) {
        encoded_value = (encoded_value << 1) | (matrix[row][col] > 0);
        col++;
    }
    result[row][threadIdx.x] = encoded_value;
}

template <typename scalar_t>
__global__ void encode_cols_kernel(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> matrix,
    torch::PackedTensorAccessor<int32_t, 2, torch::RestrictPtrTraits, size_t> result,
    int n,
    int encode_size
) {
    int col = blockIdx.x;
    int row = threadIdx.x * encode_size;

    int32_t encoded_value = 0;

    #pragma unroll
    for (int i = 0; i < encode_size && row < n; i++) {
        encoded_value = (encoded_value << 1) | (matrix[row][col] > 0);
        row++;
    }
    result[threadIdx.x][col] = encoded_value;
}


torch::Tensor bin_linear_cuda(
    const torch::Tensor input,
    const torch::Tensor weight,
    const c10::optional<torch::Tensor> bias_opt
) {
    torch::Tensor input_expanded;

    const int m = input.size(0);
    const int n = input.size(1);
    const int k = weight.size(0);

    auto result = input.new_zeros({m, k});
    if (bias_opt.has_value()) {
        result.add_(bias_opt.value());
    }

    const int encoded_dim = ceil(double(n) / double(ENCODE_SIZE));

    auto optionsEncode = torch::TensorOptions()
        .device(weight.device())
        .dtype(torch::kInt32);
    auto encoded_rows = torch::zeros({m, encoded_dim}, optionsEncode);
    AT_DISPATCH_FLOATING_TYPES(input.type(), "encode_rows_kernel", ([&] {
        encode_rows_kernel<scalar_t><<<m, encoded_dim>>>(if (input.dim() == 2) {
            input.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            encoded_rows.packed_accessor<int32_t, 2, torch::RestrictPtrTraits, size_t>(),
            n,
            ENCODE_SIZE
        );
    }));

    auto encoded_cols = torch::zeros({encoded_dim, k}, optionsEncode);
    auto weight_transposed = weight.t();
    AT_DISPATCH_FLOATING_TYPES(weight_transposed.type(), "encode_cols_kernel", ([&] {
        encode_cols_kernel<scalar_t><<<k, encoded_dim>>>(
            weight_transposed.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            encoded_cols.packed_accessor<int32_t, 2, torch::RestrictPtrTraits, size_t>(),
            n,
            ENCODE_SIZE
        );
    }));

    dim3 blockSize(32, 32);
    dim3 blocksPerGrid(
        ceil(double(m) / double(blockSize.x)),
        ceil(double(k) / double(blockSize.y))
    );
    AT_DISPATCH_FLOATING_TYPES(result.type(), "bin_linear_forward_kernel", ([&] {
        bin_linear_kernel<scalar_t><<<blocksPerGrid, blockSize>>>(
            result.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            encoded_rows.packed_accessor<int32_t, 2, torch::RestrictPtrTraits, size_t>(),
            encoded_cols.packed_accessor<int32_t, 2, torch::RestrictPtrTraits, size_t>(),
            m,
            n,
            k,
            encoded_dim
        );
    }));

    return result;
}
