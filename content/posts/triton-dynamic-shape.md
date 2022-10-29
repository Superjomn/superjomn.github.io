+++
title = "OpenAI/Triton 与 dynamic shape"
author = ["Chunwei Yan"]
date = 2022-10-27
tags = ["triton;AI compiler"]
draft = true
+++

<div class="ox-hugo-toc toc">

<div class="heading">Table of Contents</div>

- [OpenAI/Triton 与 dynamic shape](#openai-triton-与-dynamic-shape)
    - [GPU 高性能 Kernel 的编写方式](#gpu-高性能-kernel-的编写方式)
    - [triton 支持 dynamic shape 的方式：block + padding](#triton-支持-dynamic-shape-的方式-block-plus-padding)

</div>
<!--endtoc-->


## OpenAI/Triton 与 dynamic shape {#openai-triton-与-dynamic-shape}

OpenAI/Triton（以下简称为 triton）是一种编写 Tensor 相关的 GPU kernel 的编程语言，目前主要面向 AI 场景。
考虑到 AI 场景中，Op 的输入都是任意 shape的，那么 triton 是如何支持 dynamic shape 的呢？


### GPU 高性能 Kernel 的编写方式 {#gpu-高性能-kernel-的编写方式}

在开发 GPU 上高性能 Kernel 时，我们基本要考虑到如下因素

1.  IO: 让 memory hierarchy 充分利用起来，比如 shared memory 128B 的 bandwidth
2.  Parallism: 并行计算资源充分利用起来，比如充分使用 TensorCore warp 级的指令

这两点均需要多个层次的 tile 构建一组 tile hierarchy，来实现 IO 和 Parallism 的充分利用。

例如，经典的 GEMM 用 TensorCore 实现中存在三层 tile hierarcy:

1.  Global memory -&gt; Shared memory
2.  shared memory -&gt; register file
3.  Tensor Core

{{< figure src="/2022-10-27_10-05-31_screenshot.png" >}}

其中，这些 tile 的 shape 往往需要是固定的几种尺寸。
具体原因主要有两个

1.  指令的宽度是确定的，导致对应的 tile shape 也是部分确定的；这里比如 shared memory -&gt; register file 阶段会用到的 `ldmatrix.sync.aligned.m8n8.x4.shared.b16` 指令
    -   其 load 的数据量是 `` 8x8x4xb16=16x16xb16` `` ，这也决定了这层的 tile shape 是 `16x16` 的整数倍
2.  Tile hierarchy 之间有对应比例
    -   比如 shared memory -&gt; register file(ldmatrix指令) 这一层与 Tensor Core(mma指令) 计算量之间有数量关系
        -   单个Pass 上 ldmatrix 的tile load 的数据量应该等于mma的计算所需输入参数的数据量

综上，tile shape 是部分固定的，无法做到完全的 dynamic shape。

在 triton 中，tile shape 是必须固定的，相应的 dim width 必须通过 triton.constexpr 指定为常量，比如其 [matmul tutorial](https://github.com/openai/triton/blob/master/python/tutorials/03-matrix-multiplication.py) 中的 kernel 定义

```python
@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
    # by to get the element one row down (A has M rows)
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    pass
```

其中 `[BLOCK_SIZE_K,BLOCK_SIZE_N,BLOCK_SIZE_K]` 等就是最外层 tile 的 shape，不过都定义成了 `tl.constexpr` 。
**`tl.constexpr` 需要在 compile 前指定为具体的数值，不同的数值组合会 compile 出完全不同的 cubin。**

综上，高性能kernel需要通过tile的技巧实现，tile shape由于多种限制必须是部分固定的，原理类似的 triton 的 block shape 进一步是完全固定的，不过留了 knob 让用户调整。


### triton 支持 dynamic shape 的方式：block + padding {#triton-支持-dynamic-shape-的方式-block-plus-padding}

那 triton 是如何支持 dynamic shape 的呢？ 答案就是 **padding** 。

上文有说明 triton 的 block shape 是完全固定的，那用于输入的 dynamic shape 到固定的 block shape 之间必然有可能是非整数倍，triton 通过 padding 的方式确保整数倍，牺牲一部分算力（对于并行程序，理论上耗时应该没有变化)，以对 tile 友好为前提，支持了 dynamic shape。

triton 的 padding 主要是通过 `masked_load` 和 `masked_store` 来实现的，具体的 python 语法是 `triton.language.load(pointer, mask=None, other=None, cache_modifier='', eviction_policy='', volatile=False)`
_（注意这里的 `load/store` 的功能是从 global memory 到 register file）_ 。

`masked_load` 和 `masked_store` 的语义类似 Select 语句：

```python
def masked_load(addrs:array, mask:array, other:array): # non-tiled
    reg = array(len(addrs))
    for offset, (addr,maskv,otherv) in enumerate(zip(addrs, mask, other)):
        loaded_v = loadv(addr) if maskv else other
        reg[offset] = loaded_v
    return reg

```

这里我们以更简单的 [01-vec-add.py](https://github.com/openai/triton/blob/master/python/tutorials/01-vector-add.py#L21) 教程中的代码为例：

```python
@triton.jit
def add_kernel(
    x_ptr,  # *Pointer* to first input vector.
    y_ptr,  # *Pointer* to second input vector.
    output_ptr,  # *Pointer* to output vector.
    n_elements,  # Size of the vector.
    BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
                 # NOTE: `constexpr` so it can be used as a shape value.
):
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Create a mask to guard memory operations against out-of-bounds accesses.
    mask = offsets < n_elements
    # Load x and y from DRAM, masking out any extra elements in case the input is not a
    # multiple of the block size.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    # Write x + y back to DRAM.
    tl.store(output_ptr + offsets, output, mask=mask)
```

示例中定义了两个 vector 的加操作，vector 的长度为 `n_elements` 。
由于 vecadd 是 elementwise 的操作，只需要 1D 的 tile，示例中的 tile shape 为 `1xBLOCK_SIZE` ，其中 `BLOCK_SIZE` 的 typing 为 `tl.constexpr` 。

从 `n_elements` 到 `1xBLOCK_SIZE` 的 tile 之间，示例采用了 `masked_load/store` 的方法，具体是如下几行

```python
mask = offsets < n_elements
# Load x and y from DRAM, masking out any extra elements in case the input is not a
# multiple of the block size.
x = tl.load(x_ptr + offsets, mask=mask)
y = tl.load(y_ptr + offsets, mask=mask)
# ...
# Write x + y back to DRAM.
tl.store(output_ptr + offsets, output, mask=mask)
```

上述代码中，masked load 和后续的计算的配合可以用下图来表示

{{< figure src="/2022-10-28_10-54-19_screenshot.png" >}}

在 `tt.load` 之后，加载得到的数据量一定是 tile shape 的整数倍，所以对 block 内的任意计算都是合法的。

整个 triton 对 dynamic shape 的支持均构建自上述的 padding 思想， **通过 masked load/store 将数据无缝适配到 tile hierarchy，如此即利用了固定 tile 来保证性能，又支持了用户层的 dynamic shape** 。
