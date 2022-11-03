+++
title = "OpenAI/Triton 中的 Data Alignment 分析"
author = ["Chunwei Yan"]
date = 2022-10-23
tags = ["triton;AI compiler"]
draft = true
+++

_本文中的 OAI/Triton 指的是 OpenAI 的 [triton](https://github.com/openai/triton) 项目（这里区别于 NVIDIA 的 triton inference server），以下直接称为 triton。_

_当前，triton 正在向 MLIR 做迁移，大部分核心代码都在做重构。我个人也在参与其中，持续学习和贡献。我之前并没有接触过 compute bound 的 compiler，因此接下来会在 blog 中记录 triton 里面的一些设计和实现方面的理解。_

_文中的代码主要是 triton 下 [triton-mlir](https://github.com/openai/triton/tree/triton-mlir) 这个分支的代码，由于近期这个分支合并到 master 分支后可能会删除，所以本文中的代码链接均指向我 fork 的 repo 中对应分支的某个 commit 以保证链接稳定可见。_


## OpenAI/Triton 中的 Data Alignment 分析 {#openai-triton-中的-data-alignment-分析}

所谓 Data Alignement，在 [维基百科](https://en.wikipedia.org/wiki/Data_structure_alignment) 中有相关的描述：

> The CPU in modern computer hardware performs reads and writes to memory most efficiently when the data is naturally aligned, which generally means that the data's memory address is a multiple of the data size

这里的高效的 memory read/write 需要的所谓 `aligned` 对应着两点要求

1.  数据在 memory 中的排布（data layout） 是 aligned，即数据的 memory address 是某种基本单位的整数倍
2.  memory access 的指令是 aligned，即读写的 memory address 是基本单位的整数倍

GPU 处理器在 data alignment 的要求与文中的 CPU 类似，在 aligned 的数据上用 aligned 的 memory access 指令能做到更高效，
比如在保证正确性的前提下可以贪心使用一些 vectorized 的指令加速 memory read/write。

在 triton IR 中，tensor 是贯穿始终的数据类型，tensor 的 data alignment 则对其 IO 的读取性能（指令的 vectorization）有直接影响。

triton 中的 alignment 的信息用 `AxisInfo` 数据结构表示，对应的分析 Pass 叫做 `AxisInfoAnalysis` 。


### triton 中的 Alignment 数据结构： AxisInfo {#triton-中的-alignment-数据结构-axisinfo}

对应着上述 alignement 的两点要求，其中第一点的 data layout 是外界给入，triton 其实无法改变，只能尽可能优化第二点，也就是 memory access 的 alignment。
为了保证在 GPU 上的执行效率，triton 会进行 data alignment 的分析，并尝试调整合适的 memory access 指令以达到尽可能高的 memory access 带宽。

triton 里面有一个 AxisInfo 的数据结构用于 alignment 的分析， 参考 [AxisInfo.h](https://github.com/openai/triton/blob/triton-mlir/include/triton/Analysis/AxisInfo.h) 中的定义，其包含的信息主要有三个：

-   contiguity
-   divisibility
-   constancy

这三类信息包含了 tensor 每个 axis 数据的数值特性， 综合起来指导最后 codegen 的 memory access 相关指令的挑选。

考虑到最终读取的时候一般会在数据排布连续的维度上进行，（比如 row-major 的 tensor，其实一般就在列的那一维度进行读写），因此最终 codegen 的时候其实只需要连续的那一维的 AxisInfo 信息便可。
triton 里面对每个维度都进行分析，主要考虑到 tensor 的维度可能会重排（比如 tensor-transpose 的情况），
这里 triton 里面每个在 shared memory 上的 tensor（与 memory 相关） 会有一个 order 的向量， `order[0]` 就表示数据排布连续的维度的id。


#### 信息一：contiguity {#信息一-contiguity}

```C++
/// The _contiguity_ information maps the `d`-th
/// dimension to the length of the shortest
/// sequence of contiguous integers along it
/// For example:
/// [10, 11, 12, 13, 18, 19, 20, 21]
/// [20, 21, 22, 23, 28, 29, 30, 31]
/// Would have contiguity [1, 4].
/// and
/// [12, 16, 20, 24]
/// [13, 17, 21, 25]
/// [14, 18, 22, 26]
/// [15, 19, 23, 27]
/// [18, 22, 26, 30]
/// [19, 23, 27, 31]
/// Would have contiguity [2, 1].
DimVectorT contiguity;
```

顾名思义， contiguity 表示的是每个 axis 上连续的最小步长。


#### 信息二：divisibility {#信息二-divisibility}

```C++
/// The _divisibility_ information maps the `d`-th
/// dimension to the largest power-of-two that
/// divides the first element of all the values along it
/// For example:
/// [10, 11, 12, 13, 18, 19, 20, 21]
/// [20, 21, 22, 23, 28, 29, 30, 31]
//  would have divisibility [1, 2]
//  and
/// [12, 16, 20, 24]
/// [13, 17, 21, 25]
/// [14, 18, 22, 26]
/// [15, 19, 23, 27]
//  would have divisibility [4, 1]
DimVectorT divisibility;
```

divisibility 表示的是每个 axis 上首地址能被整除的最大的 2 指数的值


#### 信息三：constancy {#信息三-constancy}

```C++
/// The _constancy_ information maps the `d`-th
/// dimension to the length of the shortest
/// sequence of constant integer along it. This is
/// particularly useful to infer the contiguity
/// of operations (e.g., add) involving a constant
/// For example
/// [8, 8, 8, 8, 12, 12, 12, 12]
/// [16, 16, 16, 16, 20, 20, 20, 20]
/// would have constancy [1, 4]
DimVectorT constancy;
```

constancy 表示每个轴上最短的连续常数序列的长度


### AxisInfo 分析逻辑：AxisInfoAnalysis Pass {#axisinfo-分析逻辑-axisinfoanalysis-pass}

triton 中有一个  [AxisInfoAnalysis](https://github.com/Superjomn/triton/blob/c31d331fc090d78b162345d84ae222335df2d429/include/triton/Analysis/AxisInfo.h#L111) Pass 用于对整个 IR 内的 tensor 进行 AxisInfo 的分析和指定。

主要逻辑如下图

{{< figure src="/2022-11-01_18-43-12_screenshot.png" >}}

其中主要有四块需要关注的


#### (1) 每个 Tensor 以 forward data flow 的顺序进行访问 {#1--每个-tensor-以-forward-data-flow-的顺序进行访问}

[AxisInfoAnalysis](https://github.com/Superjomn/triton/blob/c31d331fc090d78b162345d84ae222335df2d429/include/triton/Analysis/AxisInfo.h#L111) Pass 直接继承了 MLIR 中的 [ForwardDataflowAnalysis Driver](https://mlir.llvm.org/docs/Tutorials/DataFlowAnalysis/#forwarddataflowanalysis-driver)，这是一个用来基于 data flow 做分析的工具类。
AxisInfoAnalysis 访问 IR node 的顺序由 ForwardDataflowAnalysis 决定。


#### (2) 判定 tensor 是否是 module 的输入 {#2--判定-tensor-是否是-module-的输入}

`ForwardDataflowAnalysis` 中有一些 Hook，其中一个是 `getPessimisticValueState` ，对于 module 的输入 tensor（没有办法推导的)给定一个默认值。


#### (3) 设置一个默认的 AxisInfo {#3--设置一个默认的-axisinfo}

这里延续步骤 (2) 里的操作， `getPessimisticValueState` 里面负责给默认值，triton 有一些特殊的的 hint，用来在定义 kernel 时，为特定的 argument 指定 divisibility。

比如 kernel 中的 fp16 tensor 的 ptr argument，triton 会自动添加 `tt.divisibility=16` 到 argument 的 meta info 里。

除了自动设置外，triton 的 Python API 里面有个 [multiple_of](https://triton-lang.org/master/python-api/triton.language.html#compiler-hint-ops) 来显式地指定 divisibility。


#### (4) 通过 Op 的规则推导出 result 的 AxisInfo {#4--通过-op-的规则推导出-result-的-axisinfo}

除了 entry 的 tensor，其他 tensor 理论上都是由 Op 产生的 result。
AxisInfoAnalysis pass 里为特定种类的 Op 绑定了启发式规则来帮助推导这些 tensor 的 AxisInfo。

其中一些典型的规则如下

-   MakeRangeOp
    -   `contiguity = {end - start}`
    -   `divisibility = highestPowOfDivisior(start)`
    -   `constancy = {1}`
-   AddIOp or AddPtrOp
    -   `contiguity = { max(gcd(lhs.contiguity[d], rhs.constancy[d]), gcd(lhs.constancy[d], rhs.contiguity[d])) ...}`
    -   `divisibility = { gcd(lhs.divisibility[d], rhs.divisibility[d]) ...}`
    -   `constancy = { gcd(lhs.constancy[d], rhs.constancy[d]) ... }`
-   SplatOp
    -   `contiguity = {1 ...}`
    -   `divisibility = { OpInfo.divisibility[0] ...}`
    -   `constancy = { retType.shape[d] ...}`

细节可以参考[代码](https://github.com/Superjomn/triton/blob/c31d331fc090d78b162345d84ae222335df2d429/lib/Analysis/AxisInfo.cpp#L108)。

这里以 AddIOp 的规则为例详解，其有两个参数，分别为 lhs 和 rhs，由于 AddI 操作的特殊性，一串连续的数组加上一串常数的结果还是连续的，这也就是规则中
  `contiguity = { max(gcd(lhs.contiguity[d], rhs.constancy[d]), gcd(lhs.constancy[d], rhs.contiguity[d])) ...}` 的由来。
另外两个信息 `divisibility` 和 `constancy` 取 gcd 便可。

比如，一维 tensor 的情况

-   lhs 的数值: `[1, 2, 3, 7, 8, 9]`
    -   contiguity: `{3}`
    -   divisibility: `{1}`
    -   constancy: `{1}`
-   rhs 的数值: `[1, 1, 1, 1, 1, 1]`
    -   contiguity: `{1}`
    -   divisibility: `{1}`
    -   constancy: `{6}`

数值相加得到结果:

-   res 的数值： `[2, 3, 4, 8, 9, 10]` ，实际的 AxisInfo 如下
    -   contiguity: `{3}`
    -   divisibility: `{2}`
    -   constancy: `{1}`

按照 AddIOp 的规则，

-   `contiguity = max(gcd(3, 6), gcd(1, 1)) = 3`
-   `divisibility = gcd(1, 1) =` \\(1 \neq 2\\)
-   `constancy = gcd(6, 1) = 1`

啊哈，这里暴露了 `divisibility` 的计算错了，应该是 2，但规则只给了 1。 这里我的理解是，包括 divisibility 在内的 AxisInfo 的分析规则是比较保守的，需要保证最终 memory access 的 Alignment 的正确性，因此规则出来的结果总会 `le` 实际的结果。极端情况，规则可能会给出全是 1 的结果，这个不会导致正确性的问题，只会影响最终性能。

接下来会介绍 AxisInfo 如何影响最终的 memory read/write 的 codegen。


### AxisInfo 信息如何优化 memory access {#axisinfo-信息如何优化-memory-access}

triton 中有关 memory access 的 Op 主要有一下几种

-   `LoadOp` ，从 global memory 读取数据到 register
-   `StoreOp` ，从 register 写到 global memory
-   `DotOp` ，需要从 shared memory 读数据到 register
-   atomic 操作
    -   `AtomicRMWOp`
    -   `AtomicCASOp`
-   `gpu::InsertSliceAsyncOp`, 从 register 写 shared memory
-   `gpu::ExtractSliceOp` ， 从 shared memory 读到 register

这些 Op 至少会用到不少跟 memory access 相关的指令（NVIDIA Ampere PTX，详细可以参考 [Data Movement and Conversion Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld) 章节），这里篇幅原因只列出其中两种典型的指令定义和比较关键的字段。

**`ld` , load memory**

```ptx
ld{.weak}{.ss}{.level::eviction_priority}{.level::cache_hint}{.level::prefetch_size}{.vec}.type
                                                                            d, [a]{, cache-policy};

.ss =                       { .const, .global, .local, .param, .shared{::cta, ::cluster} };
.vec =                      { .v2, .v4 };
.type =                     { .b8, .b16, .b32, .b64,
                              .u8, .u16, .u32, .u64,
                              .s8, .s16, .s32, .s64,
                              .f32, .f64 };
```

**`st`, store memory**

```ptx
st{.weak}{.ss}{.level::eviction_priority}{.level::cache_hint}{.vec}.type
                                                      [a], b{, cache-policy};

.ss =                       { .global, .local, .param, .shared{::cta, ::cluster} };
.vec =                      { .v2, .v4 };
.type =                     { .b8, .b16, .b32, .b64,
                              .u8, .u16, .u32, .u64,
                              .s8, .s16, .s32, .s64,
                              .f32, .f64 };
```

`ld` 和 `st` 指令定义类似，其中比较关键的 `.vec` 表示 vector width，支持 2 和 4，表示一次处理 2 或 4 个 `.type` 的数据，当然不加 `.vec` 的话表示 vector width=1。

triton 会利用 AxisInfo 的信息来指导 codegen 阶段中对 memory read/store 相关指令的选择，triton 会尝试挑选最大化 vector width 的指令。

比如 Load 指令的 vector width 的逻辑是在memory连续的维度上  `min(contiguity, divisibility, sizePerThread)` ，其中 `contiguity` 和 `divisibility` 就是上面 AxisInfo 中的概念， `sizePerThread` 表示每个 thread 需要处理的在 memory 中连续排布的元素个数。

详细细节可以参考 [code](https://github.com/Superjomn/triton/blob/12d60cb4a306e8397ee00717486eb0f36c6eddcb/lib/Conversion/TritonGPUToLLVM/TritonGPUToLLVM.cpp#L811-L818)，主要逻辑摘要如下

```C++
unsigned getAlignment(Value val, const Attribute &layout) const {
    auto axisInfo = getAxisInfo(val);
    auto order = getOrder(layout);
    unsigned maxMultiple = axisInfo->getDivisibility(order[0]);
    unsigned maxContig = axisInfo->getContiguity(order[0]);
    unsigned alignment = std::min(maxMultiple, maxContig);
    return alignment;
  }
```

具体确定 vec 的代码：

```C++
// Here order should be ordered by contiguous first, so the first element
// should have the largest contiguous.
auto order = getOrder(layout);
unsigned align = getAlignment(ptr, layout);

unsigned contigPerThread = getSizePerThread(layout)[order[0]];
unsigned vec = std::min(align, contigPerThread);
vec = std::min<unsigned>(shape[order[0]], vec);
```

确定了 vec，最终的指令挑选如下 [code](https://github.com/Superjomn/triton/blob/12d60cb4a306e8397ee00717486eb0f36c6eddcb/lib/Conversion/TritonGPUToLLVM/TritonGPUToLLVM.cpp#L938-L948)

```C++
// Define the instruction opcode
ld.o("volatile", op.isVolatile())
    .global()
    .o("ca", op.cache() == triton::CacheModifier::CA)
    .o("cg", op.cache() == triton::CacheModifier::CG)
    .o("L1::evict_first", op.evict() == triton::EvictionPolicy::EVICT_FIRST)
    .o("L1::evict_last", op.evict() == triton::EvictionPolicy::EVICT_LAST)
    .o("L1::cache_hint", hasL2EvictPolicy)
    .v(nWords)
    .b(width);
```

这里的 `.v(nWords)` 中的 nWords 就等于上面的 vec，最终会按需为 `ld` 指令添加 `.v2` 或 `.v4` 的后缀最终实现向量化的 load。

联系到前面 AxisInfo 分析中讲到的，contiguity, constancy, divisibility 这三维信息会直接影响这里指令的向量化，如果分析的结果不准确会有如下问题

1.  指令的 vec 规模过大，尾部的 memory read/write 会溢出，导致程序崩溃的严重错误
2.  指令的 vec 规模太小，指令向量化的效果变差，影响 memory access 的性能，但不会导致其他问题

因此，在 AxisInfo 没办法特别精确的情况下（这块逻辑还需要更多的优化），保守一些的策略是比较靠谱的做法（在极端情况下甚至可以都设置成 1，避免指令的向量化），先构建好框架，策略后续可以接着迭代来提升。


### Reference {#reference}

-   [CUDA PTX Documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld)，文中必要的指令这里都有
-   [triton/triton-mlir 分支](https://github.com/openai/triton/tree/triton-mlir)，包含最新的代码，但近期可能会 merge 到 master 分支
-   [Superjomn/triton/triton-mlir](https://github.com/Superjomn/triton/tree/12d60cb4a306e8397ee00717486eb0f36c6eddcb)，我 Repo 中的代码较新的一个 commit，文中引用到的代码会稳定存在
