+++
title = "OpenAI/Triton 中的 Data Layout"
author = ["Chunwei Yan"]
date = 2022-11-02
tags = ["triton;AI compiler"]
draft = true
+++

## OpenAI/Triton 中的 Data Layout {#openai-triton-中的-data-layout}

在 GPU 编程中，为并发的线程分发数据是比较复杂的，其难度主要在两个层面

1.  并发的粒度不一，最简单是 thread，但近年来 Tensor Core 又引入了 warp 级的指令，不同的指令可能需要不同的数据排布
2.  shared memory 上的数据排布需要考虑不同并发粒度上的 bank conflict

triton 作为一个面向 GPU 的 编译器，面向多线程的 workload 分发逻辑是必不可少的。 此外，随着 Tensor Core 的引入，面向 warp 的分发也让分发逻辑更加复杂。


### triton 中数据分发相关的基础概念 {#triton-中数据分发相关的基础概念}

triton 中有一些数据分发相关的基础概念贯穿始终

| 概念             | 数据类型 | 含义                                       |
|----------------|------|------------------------------------------|
| `order`          | Vector | 表示 tensor 维度排布的顺序，其中 `order[0]` 是连续的维度 |
| `shapePerCTA`    | Vector | 单个 CTA 处理的 shape                      |
| `sizePerThread`  | Vector | 每个 thread 处理的 **连续排布** 的元素数目 |
| `threadsPerWarp` | Vector | 每个 Warp 在不同维度上的线程数，用向量表示 |
| `ElemsPerThread` | Vector | 每个线程处理的元素总个数                   |
| `warpsPerCTA`    | Vector | 每个 CTA 对应的 Warp 数目，这个由用户在 Python 层制定 |
| `rep`            | Vector | repeat，表示单个线程需要处理 `sizePerThread` 对应数据量的次数 |

这些概念彼此之间是有一些数学关系的

-   \\(shapePerCTA = warpsPerCTA \times threadsPerWarp \times sizePerThread \times rep\\)
-   \\(shapePerCTA = ElemsPerThread \times threadsPerWarp\\)
-   \\(ElemsPerThread = sizePerThread \times pre\\)


### triton 中的 data layout {#triton-中的-data-layout}

为了在编译过程中为 thread 或者 warp 分发 workload，triton 中引入了 layout 的抽象，来表示和传递 workload 分发的逻辑。
这类 layout 的具体是作为 attribute 绑定到每个 tensor 中，在 MLIR 中的体现是作为 `RankedTensorType` 的 encoding。

layout 在 `TritonToTritonGPU` Pass 插入到 IR 中的 tensor 中，此后在 Coalesce 和 TritonGPUCombine 等 Pass 中均会参与分析和优化，最终在 `TritonGPUToLLVM` Pass 中直接指导 CodeGen，生成 LLVM dialect IR。

triton 中常见的几种 layout 如下

1.  blocked layout：表示 thread 间平均分配 workload 的情况，每个线程可以 own 一块 memory 上连续的 data
2.  shared layout：表示位于 shared layout 多线程上访问方式的描述，比如 swizzle 的一些逻辑
3.  mma layout：表示 Tensor Core 中 MMA 指令产出的 tensor 的 data layout
4.  dot operand layout：表示 MMA 指令的操作数的 data layout
5.  slice layout：单个维度上的数据反向索引

下面依次详细介绍


#### 1. Blocked Layout {#1-dot-blocked-layout}

Blocked layout 用于在 CTA 范围内，将数据平均分配给线程，具体地，就是将 `shapePerCTA` 的数据均分给 `warpsPerCTA \times threadsPerCTA` 个线程。

<!--list-separator-->

-  包含信息

    Blocked layout 包含了如下三个信息

    -   sizePerThread
    -   threadsPerWarp
    -   warpsPerCTA

    这里举一个实际的例子，将一个 shape 为 `32x64` 的 tensor 分配到 4 个 warp 上，相关配置是

    ```python
    warpsPerCTA = np.array([2, 2])
    threadsPerWarp = np.array([4, 8])
    shapePerCTA = np.array([32, 64])
    sizePerThread = np.array([2, 4])
    ```

    参考最开始的数学关系，通过如下一些简单的计算，我们可以进一步获得一些数值

    ```python
    rep = shapePerCTA // (sizePerThread * threadsPerWarp * warpsPerCTA)
    elemsPerThread = shapePerCTA // (warpsPerCTA * threadsPerWarp)
    shapePerWarp = shapePerCTA // warpsPerCTA
    ```

    -   rep: [2 1]
    -   shapePerWarp: [16 32]

<!--list-separator-->

-  数据排布演示

    workload 在线程上的发布如下图

    {{< figure src="/static/triton-blocked-layout-static.png" >}}

    图中，t0,t1,t2 表示是 thread 的 ID，属于同一个 thread 的 workload 用相同的颜色进行了标识。
    对应图，上述的一些数值有如下对应关系：

    -   以 t0 为例，一个连续的数据（下文以contig代称）的 shape 是 2x4(sizePerThread)，对应图中就是一片黄色的格子有 2行4列
    -   一个 thread 在一个 warp 对应的数据中，会有 2 个 contig（对应到 shapePerWarp），在图中，t0 在 warp0 的数据中有两片黄色的区域

    细化到一个 warp 中的 32 个 thread 执行的模拟效果如下

    {{< figure src="/static/triton-blocked-layout-animation.gif" >}}

    需要关注到

    1.  thread 得依次执行 rep 个 contig 中的元素（这里忽略 instruction vectorization）


#### 2. Shared Layout {#2-dot-shared-layout}

Shared Layout 用于表示数据位于 shared memory 中的 tensor 的layout。

<!--list-separator-->

-  包含信息

    其包含如下字段

    -   vec, 支持 vectorization 的单位
    -   perPhase, 每个 phase 包含多少个 vec
    -   maxPhase, tensor 总共包含多少个 phase
    -   order, axis 的次序

    其中，vec, perPhase, maxPhase 是用于避免 bank conflict 的 swizzle 操作需要的参数。

<!--list-separator-->

-  数据排布演示

    triton 采用经典的 swizzle 操作来避免 bank conflicit，这里也演示经典 swizzle。


#### 3. MMA Layout {#3-dot-mma-layout}

[MMA Layout](https://github.com/openai/triton/blob/f40c63fb03acefcd32e8ab18d31eaee1708ca212/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td#L276) 表示 DotOp 的 layout，帮助确定 DotOp 到 mma 指令的具体映射方法。

<!--list-separator-->

-  包含信息

    MMA Layout 主要包含两个字段：

    -   `version` ，表示 TensorCore 的版本
        -   1 为 Volta
        -   2 为 Ampere
    -   `warpsPerCTA`

<!--list-separator-->

-  数据排布演示

    具体的数据排布，主要在最终 Codegen 的时候进行，MMA Layout 会辅助指导 mma 指令的生成。
    不同的计算精度以及不同的 `version` 最终会映射到不同的 mma 指令，也会决定了完全不同的数据排布。

    为了简化演示，我们进行如下设定

    -   计算精度为 FP16
    -   `version=2`
    -   `warpsPerCTA=[8,4]`

    这里演示 FP16 精度下， `version=2` 的数据排布（会映射到 `mma.m16n8k16` 指令）的 Accumulators (C or D) 的 [数据排布](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-16816-float)。

    {{< figure src="/static/mma.16816.png" >}}


#### 4. DotOperand Layout {#4-dot-dotoperand-layout}

[DotOperand Layout](https://github.com/openai/triton/blob/f40c63fb03acefcd32e8ab18d31eaee1708ca212/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td#L398) 表示的是 DotOp 的 $a $b 两个参数的数据排布，必须跟 MMA Layout 是配套使用。

<!--list-separator-->

-  包含信息

    其主要包含如下信息

    1.  `opIdx` ， Operand 的 ID
        -   `opIdx=0` 表示 DotOp 的 $a
        -   `opIdx=1` 表示 DotOp 的 $b
    2.  `parent` ，存储其对应的 MMA Layout，这里 DotOperand 的数据排布也间接的由 MMA Layout 确定（不同的 mma 指令需要的 operand 的数据排布也是不同的）

<!--list-separator-->

-  数据排布演示

    这里为了方便演示，我们采用 MMA Layout 中的 `mma.m16n8k16.f16` 指令

    -   计算精度为 FP16
    -   MMA Layout
        -   `version=2`
        -   `warpsPerCTA=[8,4]`
    -   对于 $a，对应的 DotOperand 的 opIdx = 0
    -   $b, 对应的 DotOperand 的 opIdx = 1

        {{< figure src="/static/mma.16816.op.png" >}}


#### <span class="org-todo todo TODO">TODO</span> 5. Slice Layout {#5-dot-slice-layout}


### Layout 转换 {#layout-转换}


#### ConvertLayoutOp {#convertlayoutop}

在 MLIR 中，不同的 layout 是作为 RankedTensorType 的 Encoding 存储的，是类型系统的一部分。
具体的类型最终由相应的 Op 来决定，比如 DotOp 的输入一定是 DotOperand Layout，输出一定是 MMA Layout 等等。

这就必然存在，不同的 layout 之间相互转化的问题，这就是 ConvertLayoutOp 的作用。

目前，ConvertLayoutOp 支持如下layout的转化：

-   Blocked Layout -&gt; Shared Layout
-   Shared Layout -&gt; DotOperand Layout
-   {Blocked, MMA, Slice} Layout -&gt; {Blocked, MMA, Slice} Layout

<!--list-separator-->

-  Blocked -&gt; Shared Layout 的转换

<!--list-separator-->

-  Shared Layout -&gt; DotOperand Layout 的转换

<!--list-separator-->

-  {Blocked, MMA, Slice} Layout -&gt; {Blocked, MMA, Slice} Layout 的转换
