+++
title = "OAI/Triton 中的 alignment 分析"
author = ["Chunwei Yan"]
date = 2022-10-23
tags = ["triton;AI compiler"]
draft = true
+++

<div class="ox-hugo-toc toc">

<div class="heading">Table of Contents</div>

- [OAI/Triton 中的 alignment 分析](#oai-triton-中的-alignment-分析)
    - [alignment 包含信息介绍](#alignment-包含信息介绍)
    - [AxisInfo analysis pass 逻辑](#axisinfo-analysis-pass-逻辑)
    - [alignment 信息的使用](#alignment-信息的使用)

</div>
<!--endtoc-->

_本文中的 OAI/Triton 指的是 OpenAI 的 [triton](https://github.com/openai/triton) 项目（这里区别于 NVIDIA 的 triton inference server），以下直接称为 triton。_

_当前，triton 正在向 MLIR 做迁移，大部分核心代码都在做重构。_
_我个人有幸参与其中，我会在 blog 中记录 triton 里面的一些设计和实现。_


## OAI/Triton 中的 alignment 分析 {#oai-triton-中的-alignment-分析}


### alignment 包含信息介绍 {#alignment-包含信息介绍}

参考 [AxisInfo.h](https://github.com/openai/triton/blob/triton-mlir/include/triton/Analysis/AxisInfo.h) 中的定义，其包含的信息主要有三个


#### contiguity {#contiguity}

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


#### divisibility {#divisibility}

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


#### constancy {#constancy}

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


### AxisInfo analysis pass 逻辑 {#axisinfo-analysis-pass-逻辑}


### alignment 信息的使用 {#alignment-信息的使用}
