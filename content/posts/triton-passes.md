+++
title = "OpenAI/Triton 中的核心 Pass 理解"
author = ["Chunwei Yan"]
date = 2022-11-15
tags = ["triton;AI compiler"]
draft = true
+++

## OpenAI/Triton 中的核心 Pass {#openai-triton-中的核心-pass}


### Swizzle Pass {#swizzle-pass}

Swizzle Pass 是一个 Analysis Pass，用于确定 [Shared Layout](https://github.com/Superjomn/triton/blob/42db3538e4257cac70c6e9c209214bef0a43ca98/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td#L46) 中与 swizzle 相关的如下参数：

-   vec: 表示单次 memory 操作的连续数据长度
-   perPhase: 表示 128bytes 能够包含的 vec 的个数
-   maxPhase: 表示一个 tile 中所需要的 phase 的个数

三者有一些数学关系

-   `vec = max(128 / row, 1)`
-   `perPhase = 128 / vec`
-   `maxPhase = max(load_size / perPhase, 1)`

获得了这样的参数，具体 swizzle 过程是


### Pipeline Pass {#pipeline-pass}

Pipeline Pass 针对 memory 的 Load 操作实现了经典的 pipeline 的数据 load 的优化，有如下特点

1.  面向 global memory 到 shared memory 的 load 操作
2.  只对 DotOp 的操作数依赖的 memory load 做优化
3.  会在 shared memory 中构建 double-buffer 或者 N-buffer
4.  通过一个 numStages 的超参决定 pipeline 优化的预加载轮数，这个参数也直接暴露到了 Python Autotune 的 [triton.Config ](https://triton-lang.org/master/python-api/generated/triton.Config.html?highlight=num_stages#triton.Config)API 中

<!--listend-->

```llvm
// matmul: 128x32 @ 32x128 -> 128x128
#AL = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#BL = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#C = #triton_gpu.mma<{version = 2, warpsPerCTA = [4, 1]}>
#A = #triton_gpu.dot_op<{opIdx = 0, parent = #C}>
#B = #triton_gpu.dot_op<{opIdx = 1, parent = #C}>

func @matmul_loop(%lb : index, %ub : index, %step : index, %A : !tt.ptr<f16>, %B : !tt.ptr<f16>) {
  %a_ptr_init = tt.broadcast %A : (!tt.ptr<f16>) -> tensor<128x32x!tt.ptr<f16>, #AL>
  %b_ptr_init = tt.broadcast %B : (!tt.ptr<f16>) -> tensor<32x128x!tt.ptr<f16>, #BL>

  %a_mask = arith.constant dense<true> : tensor<128x32xi1, #AL>
  %a_other = arith.constant dense<0.00e+00> : tensor<128x32xf16, #AL>
  %b_mask = arith.constant dense<true> : tensor<32x128xi1, #BL>
  %b_other = arith.constant dense<0.00e+00> : tensor<32x128xf16, #BL>
  %c_init = arith.constant dense<0.00e+00> : tensor<128x128xf32, #C>

  %a_off = arith.constant dense<4> : tensor<128x32xi32, #AL>
  %b_off = arith.constant dense<4> : tensor<32x128xi32, #BL>

  scf.for %iv = %lb to %ub step %step iter_args(%a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %prev_c = %c_init) -> (tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>) {
    %a_ = tt.load %a_ptr {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #AL>
    %a = triton_gpu.convert_layout %a_ : (tensor<128x32xf16, #AL>) -> tensor<128x32xf16, #A>
    %b_ = tt.load %b_ptr, %b_mask, %b_other {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #BL>
    %b = triton_gpu.convert_layout %b_ : (tensor<32x128xf16, #BL>) -> tensor<32x128xf16, #B>

    %c = tt.dot %a, %b, %prev_c {allowTF32 = true, transA = false, transB = false} : tensor<128x32xf16, #A> * tensor<32x128xf16, #B> -> tensor<128x128xf32, #C>

    %next_a_ptr = tt.addptr %a_ptr, %a_off : tensor<128x32x!tt.ptr<f16>, #AL>
    %next_b_ptr = tt.addptr %b_ptr, %b_off : tensor<32x128x!tt.ptr<f16>, #BL>
    scf.yield %next_a_ptr, %next_b_ptr, %c : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>
  }
  return
}
```

会优化为

```llvm
#blocked0 = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #triton_gpu.mma<{version = 2, warpsPerCTA = [4, 1]}>
#shared0 = #triton_gpu.shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#shared1 = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
module {
  func @matmul_loop(%arg0: index, %arg1: index, %arg2: index, %arg3: !tt.ptr<f16>, %arg4: !tt.ptr<f16>) {
    %c3_i32 = arith.constant 3 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %cst = arith.constant dense<4> : tensor<32x128xi32, #blocked1>
    %cst_0 = arith.constant dense<4> : tensor<128x32xi32, #blocked0>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<32x128xf16, #blocked1>
    %0 = tt.broadcast %arg3 : (!tt.ptr<f16>) -> tensor<128x32x!tt.ptr<f16>, #blocked0>
    %1 = tt.broadcast %arg4 : (!tt.ptr<f16>) -> tensor<32x128x!tt.ptr<f16>, #blocked1>
    %2 = arith.cmpi slt, %arg0, %arg1 : index
    %3 = triton_gpu.alloc_tensor : tensor<3x128x32xf16, #shared0>
    %4 = tt.splat %2 : (i1) -> tensor<128x32xi1, #blocked0>
    %5 = triton_gpu.insert_slice_async %0, %3, %c0_i32, %4 {axis = 0 : i32, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32x!tt.ptr<f16>, #blocked0> -> tensor<3x128x32xf16, #shared0>
    %6 = triton_gpu.alloc_tensor : tensor<3x32x128xf16, #shared1>
    %7 = tt.splat %2 : (i1) -> tensor<32x128xi1, #blocked1>
    %8 = triton_gpu.insert_slice_async %1, %6, %c0_i32, %7, %cst_2 {axis = 0 : i32, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128x!tt.ptr<f16>, #blocked1> -> tensor<3x32x128xf16, #shared1>
    %9 = tt.addptr %0, %cst_0 : tensor<128x32x!tt.ptr<f16>, #blocked0>
    %10 = tt.addptr %1, %cst : tensor<32x128x!tt.ptr<f16>, #blocked1>
    %11 = arith.addi %arg0, %arg2 : index
    %12 = arith.cmpi slt, %11, %arg1 : index
    %13 = tt.splat %12 : (i1) -> tensor<128x32xi1, #blocked0>
    %14 = triton_gpu.insert_slice_async %9, %5, %c1_i32, %13 {axis = 0 : i32, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32x!tt.ptr<f16>, #blocked0> -> tensor<3x128x32xf16, #shared0>
    %15 = tt.splat %12 : (i1) -> tensor<32x128xi1, #blocked1>
    %16 = triton_gpu.insert_slice_async %10, %8, %c1_i32, %15, %cst_2 {axis = 0 : i32, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128x!tt.ptr<f16>, #blocked1> -> tensor<3x32x128xf16, #shared1>
    %17 = tt.addptr %9, %cst_0 : tensor<128x32x!tt.ptr<f16>, #blocked0>
    %18 = tt.addptr %10, %cst : tensor<32x128x!tt.ptr<f16>, #blocked1>
    triton_gpu.async_wait {num = 2 : i32}
    %19 = tensor.extract_slice %14[0, 0, 0] [1, 128, 32] [1, 1, 1] : tensor<3x128x32xf16, #shared0> to tensor<128x32xf16, #shared0>
    %20 = tensor.extract_slice %16[0, 0, 0] [1, 32, 128] [1, 1, 1] : tensor<3x32x128xf16, #shared1> to tensor<32x128xf16, #shared1>
    %21:12 = scf.for %arg5 = %arg0 to %arg1 step %arg2 iter_args(%arg6 = %0, %arg7 = %1, %arg8 = %cst_1, %arg9 = %14, %arg10 = %16, %arg11 = %19, %arg12 = %20, %arg13 = %18, %arg14 = %17, %arg15 = %11, %arg16 = %c2_i32, %arg17 = %c1_i32) -> (tensor<128x32x!tt.ptr<f16>, #blocked0>, tensor<32x128x!tt.ptr<f16>, #blocked1>, tensor<128x128xf32, #mma>, tensor<3x128x32xf16, #shared0>, tensor<3x32x128xf16, #shared1>, tensor<128x32xf16, #shared0>, tensor<32x128xf16, #shared1>, tensor<32x128x!tt.ptr<f16>, #blocked1>, tensor<128x32x!tt.ptr<f16>, #blocked0>, index, i32, i32) {
      %22 = triton_gpu.convert_layout %arg11 : (tensor<128x32xf16, #shared0>) -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma}>>
      %23 = triton_gpu.convert_layout %arg12 : (tensor<32x128xf16, #shared1>) -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma}>>
      %24 = tt.dot %22, %23, %arg8 {allowTF32 = true, transA = false, transB = false} : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma}>> -> tensor<128x128xf32, #mma>
      %25 = tt.addptr %arg6, %cst_0 : tensor<128x32x!tt.ptr<f16>, #blocked0>
      %26 = tt.addptr %arg7, %cst : tensor<32x128x!tt.ptr<f16>, #blocked1>
      %27 = arith.addi %arg15, %arg2 : index
      %28 = arith.cmpi slt, %27, %arg1 : index
      %29 = arith.remsi %arg16, %c3_i32 : i32
      %30 = arith.remsi %arg17, %c3_i32 : i32
      %31 = arith.index_cast %30 : i32 to index
      %32 = tt.splat %28 : (i1) -> tensor<128x32xi1, #blocked0>
      %33 = triton_gpu.insert_slice_async %arg14, %arg9, %29, %32 {axis = 0 : i32, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32x!tt.ptr<f16>, #blocked0> -> tensor<3x128x32xf16, #shared0>
      %34 = tt.splat %28 : (i1) -> tensor<32x128xi1, #blocked1>
      %35 = triton_gpu.insert_slice_async %arg13, %arg10, %29, %34, %cst_2 {axis = 0 : i32, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128x!tt.ptr<f16>, #blocked1> -> tensor<3x32x128xf16, #shared1>
      %36 = tt.addptr %arg14, %cst_0 : tensor<128x32x!tt.ptr<f16>, #blocked0>
      %37 = tt.addptr %arg13, %cst : tensor<32x128x!tt.ptr<f16>, #blocked1>
      triton_gpu.async_wait {num = 2 : i32}
      %38 = tensor.extract_slice %33[%31, 0, 0] [1, 128, 32] [1, 1, 1] : tensor<3x128x32xf16, #shared0> to tensor<128x32xf16, #shared0>
      %39 = tensor.extract_slice %35[%31, 0, 0] [1, 32, 128] [1, 1, 1] : tensor<3x32x128xf16, #shared1> to tensor<32x128xf16, #shared1>
      %40 = arith.addi %arg16, %c1_i32 : i32
      %41 = arith.addi %arg17, %c1_i32 : i32
      scf.yield %25, %26, %24, %33, %35, %38, %39, %37, %36, %27, %40, %41 : tensor<128x32x!tt.ptr<f16>, #blocked0>, tensor<32x128x!tt.ptr<f16>, #blocked1>, tensor<128x128xf32, #mma>, tensor<3x128x32xf16, #shared0>, tensor<3x32x128xf16, #shared1>, tensor<128x32xf16, #shared0>, tensor<32x128xf16, #shared1>, tensor<32x128x!tt.ptr<f16>, #blocked1>, tensor<128x32x!tt.ptr<f16>, #blocked0>, index, i32, i32
    }
    triton_gpu.async_wait {num = 0 : i32}
    return
  }
}
```


### Prefetch Pass {#prefetch-pass}

类似 Pipeline Pass，Prefetch Pass 通过 double buffer 的方式帮助实现 shared memory 到 register 之间的 Memory loading。


### Coalesce Pass {#coalesce-pass}

Coalesce Pass 会优化 memory access 更加连续，以实现更好的指令 vectorization 的效果。
该 Pass 主要的实现逻辑是，根据 tensor 的 AxisInfo 中的 contiguity 对 axis 进行倒序排序，序会更新到对应的 order 中。 这样的效果是 order[0] 对应的 axis 连续性最好。


### Combine Pass {#combine-pass}
