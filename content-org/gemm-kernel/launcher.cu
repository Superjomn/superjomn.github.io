#include "cuda-kernel.cuh"
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/types.h>


void gemm_naive_f32(torch::Tensor A, torch::Tensor B, torch::Tensor C, torch::Tensor D,
                    int M, int N, int K, int lda, int ldb, int ldc) {
  dim3 block(16, 16);
  dim3 grid(CEIL(N, 16), CEIL(M, 16));

  gemm_naive_f32_kernel<<<grid, block>>>((float*)A.data_ptr(), (float*)B.data_ptr(), (float*)C.data_ptr(), (float*)D.data_ptr(), M, N, K, lda, ldb, ldc);
}

template <int TILE_DIM=16>
void gemm_naive_tiled_f32(torch::Tensor A, torch::Tensor B, torch::Tensor C,
                          torch::Tensor D, int M, int N, int K, int lda, int ldb,
                          int ldc) {
  dim3 block(TILE_DIM, TILE_DIM);
  dim3 grid(CEIL(N, TILE_DIM), CEIL(M, TILE_DIM));

  gemm_naive_tiled_f32_kernel<TILE_DIM>
      <<<grid, block>>>((float*)A.data_ptr(), (float*)B.data_ptr(), (float*)C.data_ptr(), (float*)D.data_ptr(), M, N, K, lda, ldb, ldc);
}

void gemm_tiled_mma_f16(torch::Tensor A, torch::Tensor B, torch::Tensor C,
                        torch::Tensor D, int M, int N, int K, int lda, int ldb,
                        int ldc) {
  constexpr int TILE_M = 16 * 2;
  constexpr int TILE_N = 16 * 2;
  constexpr int TILE_K = 16;

  dim3 block(TILE_N, TILE_M);
  dim3 grid(CEIL(N, TILE_N), CEIL(M, TILE_M));

  gemm_tiled_mma_f16_kernel<TILE_M, TILE_N, TILE_K>
      <<<grid, block>>>((half*)A.data_ptr(), (half*)B.data_ptr(), (float*)C.data_ptr(), (float*)D.data_ptr(), M, N, K, lda, ldb, ldc);
}

// Register torch functions

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_naive_f32", &gemm_naive_f32, "GEMM Naive");
    m.def("gemm_naive_tiled_f32", &gemm_naive_tiled_f32<16>, "GEMM Naive Tiled");
    m.def("gemm_tiled_mma_f16", &gemm_tiled_mma_f16, "GEMM Tiled MMA");
}