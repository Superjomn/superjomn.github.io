+++
title = "OpenAI/Triton 中的 IR 层次理解"
author = ["Chunwei Yan"]
date = 2022-11-02
tags = ["triton"]
draft = true
+++

_本文中的 OAI/Triton 指的是 OpenAI 的 [triton](https://github.com/openai/triton) 项目（这里区别于 NVIDIA 的 triton inference server），以下直接称为 triton。_

_当前，triton 正在向 MLIR 做迁移，大部分核心代码都在做重构。我个人也在参与其中，持续学习和贡献。我之前并没有接触过 compute bound 的 compiler，因此接下来会在 blog 中记录 triton 里面的一些设计和实现方面的理解。_

_文中的代码主要是 triton 下 [triton-mlir](https://github.com/openai/triton/tree/triton-mlir) 这个分支的代码，由于近期这个分支合并到 master 分支后可能会删除，所以本文中的代码链接均指向我 fork 的 repo 中对应分支的某个 commit 以保证链接稳定可见。_


## OpenAI/Triton 中的 IR 层次理解 {#openai-triton-中的-ir-层次理解}

triton 项目在当前（2022-11-02）正在进行的重构的核心内容就是将核心代码迁移到 MLIR 设施上。目前已经完成了大部分迁移工作。

最新的 triton 的 IR 完全基于 MLIR，IR 层次主要包含两层

1.  Triton dialect：硬件无关，承接了 triton DSL 语义的表达
2.  TritonGPU dialect：硬件相关，包含了一些 GPU 相关的 Op

这里需要注意的是，除了 Triton dialect 和 TritonGPU dialect 外，triton 还大量复用了很多社区的 dialect，比如

-   std：tensor，int, float 等等数据类型
-   arith：数学计算的表示
-   scf：控制流
-   nvvm：获取 GPU 中的 thread_id 等少数操作
-   gpu：printf 等少数操作

不过这些社区的 dialect 的使用贯穿了 Triton 和 TritonGPU 两个层次，因此本文侧重介绍 Triton dialect 和 TritonGPU dialect 两层。


### Triton dialect {#triton-dialect}

Triton dialect 主要用于表示 triton 的 Python DSL 表示的 kernel 语义，本身是硬件无关的。

尽管 triton 已经服用了不少社区的 dialect，但依旧需要表示一些 triton 特有的语义，详细的 Op 列表可以参考 [TritonOps.td](https://github.com/Superjomn/triton/blob/12d60cb4a306e8397ee00717486eb0f36c6eddcb/include/triton/Dialect/Triton/IR/TritonOps.td)。

其中一些核心 Op 如下


#### 功能性 Op {#功能性-op}

| Op               | 语义                                                                   |
|------------------|----------------------------------------------------------------------|
| get_program_id   | Returns the id of the current program instance along the given axis.   |
| get_num_programs | Returns the number of program instances launched along the given axis. |


#### 计算相关 Op {#计算相关-op}

| Op           | 语义                                                             |
|--------------|----------------------------------------------------------------|
| dot          | Returns the matrix product of two blocks.                        |
| bitcast      | Cast between types of the same bidwidth.                         |
| reduce       | reduce along the provided axis.                                  |
| ext_elemwise | Call the elementwise Op in an external lib.                      |
| make_range   | Returns contiguous values within the open interval [start, end). |


#### Shape 相关 Op {#shape-相关-op}

| Op         | 语义                                       |
|------------|------------------------------------------|
| splat      | Returns a tensor filled with a scalar.     |
| expand_dim | Insert a new axis at a specific axis.      |
| view       | Reset the shape of a tensor.               |
| broadcast  | Broadcast the given tensor to a new shape. |
| cat        | Concatenate two tensors                    |


#### memory相关 Op {#memory相关-op}

| Op                    | 语义                                 |
|-----------------------|------------------------------------|
| load                  | Load from global memory to register  |
| store                 | Store from register to global memory |
| atomic_rmw/atomic_cas | atomic ops.                          |


#### <span class="org-todo todo TODO">TODO</span> 核心 Pass {#核心-pass}


### TritonGPU dialect {#tritongpu-dialect}

TritonGPU dialect 包含了 GPU 相关的一些 核心 Op，

| Op                 | 语义                                             |
|--------------------|------------------------------------------------|
| convert_layout     | convert the data layout                          |
| async_wait         | represent the `cp.async.wait_group` instruction. |
| insert_slice_async |                                                  |
| extract_slice      |                                                  |
| alloc_tensor       | allocate a tensor in shared memory.              |


#### <span class="org-todo todo TODO">TODO</span> 核心 Passes {#核心-passes}


### IR 工作流 {#ir-工作流}

IR 的整个工作流如下

{{< figure src="/2022-11-02_20-31-32_screenshot.png" width="400" >}}

主要阶段包括

1.  用户在 Python Frontend 编写 kernel
2.  Python Frontend 的信息 codegen 成 triton dialect 对应的 IR，这部分逻辑主要在 [compiler.py](https://github.com/Superjomn/triton/blob/12d60cb4a306e8397ee00717486eb0f36c6eddcb/python/triton/compiler.py)
    1.  triton dialect 内部调用 `combine` 和 `dce` 等等 pass，进行化简产生更加紧凑的 triton dialect IR
3.  Triton dialect IR convert 到 TritonGPU dialect，这个步骤会调用很多重要的 Pass 进行优化，比如 Pipeline, Coalesce 等等
4.  TritonGPU dialect IR 进一步 convert 到 LLVM dialect IR，这一步就是所谓的 codegen，包含了从 TritonGPU dialect 逐个 Op lower 到 LLVM dialect op 的大量代码，主要逻辑位于 [TritonGPUToLLVM.cpp](https://github.com/Superjomn/triton/blob/12d60cb4a306e8397ee00717486eb0f36c6eddcb/lib/Conversion/TritonGPUToLLVM/TritonGPUToLLVM.cpp)
    1.  在 Conversion 过程中， triton 会依赖不少的 PTX inline asm，来直接操作底层的 PTX 指令以保证重点操作的效率，这些包括了 dot和load/store/atomic等等 NVPTX 后端有高效指令的 Op
5.  LLVM dialect 进一步 translate 到 LLVMIR，这一部分是是 MLIR 的设施支持
6.  LLVMIR lower 到 PTX，这一部分直接利用 NVDIA 的 ptxas 工具进行编译
