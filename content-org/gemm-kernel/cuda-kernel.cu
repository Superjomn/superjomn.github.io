// @org-executor :id common-header :code-block-begin
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <stdio.h>
#include <stdlib.h>

#define CEIL(x, y) (((x) + (y) - 1) / (y))

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4*>(&(value))[0])
#define CFLOAT4(value) (reinterpret_cast<const float4*>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])
#define FLOAT2(value) (reinterpret_cast<float2*>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2*>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162*>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4*>(&(value))[0])

inline void check_torch_dtype(torch::Tensor tensor,
                              torch::ScalarType expected_dtype) {
  if (tensor.dtype() != expected_dtype) {
    throw std::runtime_error("Tensor dtype mismatch");
  }
}

#define TORCH_BINDING_COMMON_EXTENSION(func) m.def(#func, &func, #func);
// @org-executor :code-block-end

// @org-executor :id gemm-naive-f32-kernel :code-block-begin
// Basic GEMM kernel for float32
__global__ void gemm_naive_f32_kernel(const float* A, const float* B,
                                      const float* C, float* D, int M, int N,
                                      int K, int lda, int ldb, int ldc) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && col < N) {
    float D_value = 0.0f;

    for (int i = 0; i < K; i++) {
      D_value += A[row * lda + i] * B[i * ldb + col];
    }

    D[row * ldc + col] = D_value + C[row * ldc + col];
  }
}
// @org-executor :code-block-end

// @org-executor :id gemm-naive-tiled-f32-kernel :code-block-begin
template <int TILE_DIM>
__global__ void gemm_naive_tiled_f32_kernel(const float* A, const float* B,
                                            const float* C, float* D, int M,
                                            int N, int K, int lda, int ldb,
                                            int ldc) {
  __shared__ float sA[TILE_DIM][TILE_DIM];
  __shared__ float sB[TILE_DIM][TILE_DIM];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  const int row = blockIdx.y * TILE_DIM + ty;
  const int col = blockIdx.x * TILE_DIM + tx;

  float D_value = 0.0f;

  // Loop over the tiles in the K dimension
  for (int i = 0; i < CEIL(K, TILE_DIM); i++) {
    // 1. Load A and B tiles into shared memory
    int a_row = row;
    int a_col = i * TILE_DIM + tx; // K dim
    sA[ty][tx] = (a_row < M && a_col < K) ? A[a_row * lda + a_col] : 0.0f;

    int b_row = i * TILE_DIM + ty; // K dim
    int b_col = col;
    sB[ty][tx] = (b_row < K && b_col < N) ? B[b_row * ldb + b_col] : 0.0f;

    __syncthreads();

    // 2. Compute the dot product of the tiles
    for (int k = 0; k < TILE_DIM; k++) {
      D_value += sA[ty][k] * sB[k][tx];
    }

    __syncthreads();
  }

  if (row < M && col < N) {
    D[row * ldc + col] = D_value + C[row * ldc + col];
  }
}
// @org-executor :code-block-end

// @org-executor :id gemm-tiled-mma-f16-kernel :code-block-begin
using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

template <int TILE_M, int TILE_N, int TILE_K> class GemmTiledMmaF16 {
  using FragmentA = wmma::fragment<wmma::matrix_a, //
                                   WMMA_M,         //
                                   WMMA_N,         //
                                   WMMA_K,         //
                                   half,           //
                                   wmma::row_major>;

  using FragmentB = wmma::fragment<wmma::matrix_b, //
                                   WMMA_M,         //
                                   WMMA_N,         //
                                   WMMA_K,         //
                                   half,           //
                                   wmma::row_major>;

  using FragmentAcc =
      wmma::fragment<wmma::accumulator, //
                     WMMA_M,            //
                     WMMA_N,            //
                     WMMA_K,            //
                     float>; // Note: float to ensure numerical stability

  __forceinline__ __device__ void compute(const half* A, const half* B,
                                          const float* C, float* D, int M,
                                          int N, int K, int lda, int ldb,
                                          int ldc) {
    int warp_m = threadIdx.y / WMMA_M;
    int warp_n = threadIdx.x / WMMA_N;

    FragmentA a_frag;
    FragmentB b_frag;
    FragmentAcc acc;

    int c_row_start = blockIdx.y * TILE_M + warp_m * WMMA_M;
    int c_col_start = blockIdx.x * TILE_N + warp_n * WMMA_N;
    if (c_row_start < M && c_col_start < N) {
      // Note that, we ignore the case where M or N is not a multiple of TILE_M
      // or TILE_N
      wmma::load_matrix_sync(acc, C + c_row_start * ldc + c_col_start, N,
                             wmma::mem_row_major);
    } else {
      wmma::fill_fragment(acc, 0.0f);
    }

    __shared__ half sA[TILE_M][TILE_K];
    __shared__ half sB[TILE_K][TILE_N];

    for (int i = 0; i < CEIL(K, TILE_K); i++) {
      // 1. Load A and B tiles into shared memory
      int a_row = blockIdx.y * TILE_M + threadIdx.y;
      int a_col = i * TILE_K + threadIdx.x;
      int b_row = i * TILE_K + threadIdx.y;
      int b_col = blockIdx.x * TILE_N + threadIdx.x;

      sA[threadIdx.y][threadIdx.x] = (a_row < M && a_col < K)
                                         ? A[a_row * lda + a_col]
                                         : __float2half(0.0f);
      sB[threadIdx.y][threadIdx.x] = (b_row < K && b_col < N)
                                         ? B[b_row * ldb + b_col]
                                         : __float2half(0.0f);

      __syncthreads();

      // 2. Perform the matrix multiplication on the tiles
      for (int j = 0; j < TILE_K; j += WMMA_K) {
        half* a_ptr = &sA[warp_m * TILE_M][j];
        half* b_ptr = &sB[j][warp_n * TILE_N];

        // Load matrices into fragments
        wmma::load_matrix_sync(a_frag, a_ptr, TILE_K);
        wmma::load_matrix_sync(b_frag, b_ptr, TILE_N);

        // Perform the matrix multiplication
        wmma::mma_sync(acc, a_frag, b_frag, acc);
      }

      __syncthreads();
    }

    // 3. Store the result in D
    int c_row = blockIdx.y * TILE_M + warp_m * WMMA_M;
    int c_col = blockIdx.x * TILE_N + warp_n * WMMA_N;

    if (c_row < M && c_col < N) {
      wmma::store_matrix_sync(D + c_row * ldc + c_col, acc, N,
                              wmma::mem_row_major);
    }
  }
};
// @org-executor :code-block-end
