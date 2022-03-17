#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

const auto ENCODE_SIZE = 32;


template <typename scalar_t>
__global__ void bin_matmul_kernel(
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


torch::Tensor bin_matmul_cuda(
    const torch::Tensor mat1,
    const torch::Tensor mat2
) {
    const int m = mat1.size(0);
    const int n = mat1.size(1);
    const int k = mat2.size(1);

    auto result = mat1.new_zeros({m, k});

    const int encoded_dim = ceil(double(n) / double(ENCODE_SIZE));

    auto optionsEncode = torch::TensorOptions()
        .device(mat1.device())
        .dtype(torch::kInt32);
    auto encoded_rows = torch::zeros({m, encoded_dim}, optionsEncode);
    AT_DISPATCH_FLOATING_TYPES(mat1.type(), "encode_rows_kernel", ([&] {
        encode_rows_kernel<scalar_t><<<m, encoded_dim>>>(
            mat1.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            encoded_rows.packed_accessor<int32_t, 2, torch::RestrictPtrTraits, size_t>(),
            n,
            ENCODE_SIZE
        );
    }));

    auto encoded_cols = torch::zeros({encoded_dim, k}, optionsEncode);
    AT_DISPATCH_FLOATING_TYPES(mat2.type(), "encode_cols_kernel", ([&] {
        encode_cols_kernel<scalar_t><<<k, encoded_dim>>>(
            mat2.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
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
    AT_DISPATCH_FLOATING_TYPES(result.type(), "bin_matmul_kernel", ([&] {
        bin_matmul_kernel<scalar_t><<<blocksPerGrid, blockSize>>>(
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
