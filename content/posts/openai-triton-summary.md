+++
title = "OpenAI/Triton 原理理解"
author = ["Chunwei Yan"]
date = 2022-10-28
tags = ["triton;AI compiler"]
draft = true
+++

## OpenAI/Triton 原理理解 {#openai-triton-原理理解}

[openai/triton](https://github.com/openai/triton)（后面简称为 triton） 是一个编写 compute-bound 的面向 GPU 的高性能编程语言。
其19年开源，在不长的时间内达到能力相对健全，性能一梯队的状态，最近也持续获得业界的合作和使用。包括 Pytorch，JAX 等在内的 AI framework 也选择其作为高性能算子的后端编译器，相关的集成工作正在积极推进，比如 [torch-inductor](https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747), [jax-triton](https://github.com/jax-ml/jax-triton) 等等。

triton 的架构是一个经典的编译器的设计，目前也正在从 handcraft IR 向 MLIR 的设施进行整体的迁移和重构。

本人有幸承担相关工作。尽管之前有着 memory bound 相关编译器的直接经验（[PaddlePaddle/CINN](https://github.com/PaddlePaddle/CINN)），但 triton 这样的 compute-bound compiler 我理解有更多不一样的挑战：

1.  需要深入的GPU方面体系结构的理解和针对的优化设计
    -   面向的不是 CUDA 层次，而是 PTX 的指令集
    -   面向 Memory hierarchy 高效利用的专门的设计
    -   面向高效 Tile hierarchy 的内置分析和生成
2.  经典编译器设计
    -   IR 的层次：Frontend -&gt; Triton IR -&gt; TritonGPU IR -&gt; LLVM + Inline ASM -&gt; cubin
    -   Pass的层次：Analysis + Transform + Conversion + Translation（来自 MLIR 的概念）
3.  硬件无关语言的设计和抽象
    -   前端语言精确但抽象描述一个 Kernel 内的 tensor 计算
    -   user friendliness &amp; 必要的 GPU 信息暴露之前的权衡

triton 是一个很多公司组织协作的项目，我个人也只能承担其中很小一部分工作，为了避免一叶蔽目，所以会持续 dump triton 一些理解到这里。
不出意外这里应该会有一个系列的文章，主要侧重对当前的 triton/mlir 分支的理解。

相关的内容我会分为如下多个文章：

-   Triton Frontend
    -   [ ] Triton Python synatx &amp; semantics
        -   control flow
        -   Triton IR
-   [X] dynamic shape support in Triton
-   [ ] Triton IR hierarchy based on MLIR
-   Triton optimizer
    -   [ ] Alignment analysis in Triton
    -   [ ] Allocation pass in Triton
    -   [ ] swizzling pass in Triton
    -   [ ] layout in Triton
    -   [ ] Pipeline pass in Triton
-   Triton backend
    -   [ ] DotOp conversion
    -   [ ] ConvertLayoutOp conversion

由于我本人也是刚进 GPU arch 领域，对于 MLIR 也是速成，所以难免有一些错误，欢迎指正和建议，谢谢！
