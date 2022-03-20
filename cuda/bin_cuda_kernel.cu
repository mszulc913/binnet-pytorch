#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

const auto ENCODE_SIZE = 32;
const auto BLOCK_SIZE = 32;


// See: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory
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
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    scalar_t val = 0;

    for (int i = 0; i < ceil(double(encoded_dim) / double(BLOCK_SIZE)); i++) {
        // we're iterating over (BLOCK_SIZE x BLOCK_SIZE) sub-matrices and use share memory to
        // minimize number of global memory accesses
        __shared__ int32_t rowsSub[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ int32_t colsSub[BLOCK_SIZE][BLOCK_SIZE];

        int encodedCol = i * BLOCK_SIZE + threadIdx.x;
        int encodedRow = i * BLOCK_SIZE + threadIdx.y;

        rowsSub[threadIdx.y][threadIdx.x] = (row < m && encodedCol < encoded_dim) ? encoded_rows[row][encodedCol] : 0;
        colsSub[threadIdx.y][threadIdx.x] = (col < k && encodedRow < encoded_dim) ? encoded_cols[encodedRow][col] : 0;

        __syncthreads();
        for (int j = 0; j < BLOCK_SIZE; j++) {
            val += __popc(rowsSub[threadIdx.y][j] ^ colsSub[j][threadIdx.x]);
        }
        __syncthreads();
    }

    if (row < m && col < k) {
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

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid(
        ceil(double(k) / double(blockSize.y)),
        ceil(double(m) / double(blockSize.x))
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
