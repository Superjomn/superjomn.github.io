+++
title = "CE (Continuous Evaluation) – 一种神经网络任务自适应的稳定性保证方法"
author = ["Chunwei Yan"]
date = 2022-10-23
tags = ["CE"]
draft = true
+++

<div class="ox-hugo-toc toc">

<div class="heading">Table of Contents</div>

- [CE (Continuous Evaluation) -- 一种神经网络任务自适应的稳定性保证方法](#ce--continuous-evaluation--一种神经网络任务自适应的稳定性保证方法)
    - [本文背景](#本文背景)
    - [任务背景](#任务背景)
    - [CE 的设计思路](#ce-的设计思路)
    - [实施问题](#实施问题)
    - [项目现状](#项目现状)
    - [思考](#思考)
    - [Reference](#reference)

</div>
<!--endtoc-->


## CE (Continuous Evaluation) -- 一种神经网络任务自适应的稳定性保证方法 {#ce--continuous-evaluation--一种神经网络任务自适应的稳定性保证方法}


### 本文背景 {#本文背景}

本文介绍的是我 2017 年在 Paddle 时候，开源出来的一种测试和保证 DNN 模型效果的实践方法。

在那个时候，Paddle 的 Fluid 框架初步构建完了 MVP，随着参与迭代的团队规模扩大，框架也变化很快，相应的模型的训练效果也变化剧烈，经常出现正常开发一顿时间后，测试时出现大片模型精度异常，倒头痛苦定位问题的情况。

按常理这样的问题应该是 QA团队帮解决， 但当时 Paddle 团队还算是小而美的状态，基本一个团队 cover Paddle 开发的一切。
类似于牵扯大部分人开发效率这样的基础设施的问题，团队 Lead 非常鼓励大家积极参与解决。
所以当时我们的 CI 系统是中美团队的几个资深同学直接负责的， 另外也有过 T9大佬加入亲自重构 CMake 构建系统。

模型效果在高频开发中的持续保证的确是一个比较新的综合问题，当时在公司内外貌似都没有特别成熟的解决方案。 彼时我正好从一段时间的集中开发中抽离出来，有时间有意愿去思考一些有意思的问题。

经过对问题的抽象一级一定的泛化，我们构建了一套保证模型效果的持续验证系统叫做 CE (Continuous Evaluation)，这个系统和思路至今在 Paddle 也算是一个不错的实践。

本文包含了一些特定的概念：

-   DNN 模型效果：DNN模型执行中的，精度，性能，资源消耗等等需要关注的指标的综合
-   模型稳定性: 上述 DNN 模型效果的稳定


### 任务背景 {#任务背景}


#### DL Framework 的训练执行过程 {#dl-framework-的训练执行过程}

在 DNN 模型的训练过程中主要是 Layout/Operator 的依次前向（Forward）和反向（Backward）执行，总体的执行过程如下

{{< figure src="/static/CE.drawio.png" >}}

其中重要的三个步骤是

1.  Forward propagation（前向传播），这个过程主要是在用户构建的 DNN 模型里面，计算出每个 Op（算子）的 Output
2.  Backward propagation（后向传播）：对应的前向传播之后，在 Loss Layer 的计算得到 loss，之后随着对应的 OpGrad 的序列逐层计算 gradient
3.  Optimizer：根据获得的 gradient，批量更新模型参数（Weights）

_这里有个算法表示粒度需要关注下，最初如 Caffe 的 DL Framework 的模型构建基本粒度叫做 Layer；后续 TensorFlow 等出现了更细的表示粒度叫做 Op (Operation)，以更细的粒度来更灵活地构建模型。_

在成熟的 DL Framework（比如 TensorFlow） 中，这三个步骤在实际的执行国产中，均由一系列的 Op 序列来构成。

比如，在 ResNet50 的论文描述中，其有 Forward propagation 大概有三十几层 Layer 构成，在 Caffe 的实现中可能有差不多 50左右个 Layer，但如果是基于更加精细的算子描述的 TensorFlow 或者 Pytorch，其前向可能有接近上百个 Op；前向的 Op 序列一般会一一对应到反向 Op，因此 **单单 ResNet50 这样一个比较简单的模型在训练过程中就可能涉及到几百次的Op调用** 。

{{< figure src="/static/resnet50.png" >}}

这其中包含了复杂的组合问题，一旦某个Op出现了问题，模型的精度就会出现问题。


#### DL Framework 快速迭代对模型效果的影响 {#dl-framework-快速迭代对模型效果的影响}

那么问题回到，为何在 DL Framework 的快速迭代中，DNN模型的精度会频繁出现问题呢？

模型执行中依赖的一个任意规模的 Op 序列需要关注两点：

1.  多种 Op 的组合调用，比如 Conv2d, Activation, BatchNorm 等等 Op
2.  一种 Op 可能会被多次调用，比如 ResNet50 中重复调用了 Conv2D 多次

这个 Op 序列的实现上，每种 Op 的实现均不能出现精度BUG，甚至一种 Op 的重复执行也不能出现数值异常，否则精度问题的累积就会在最终 Optimizer 阶段得到完全错误的 Weights 更新，表现在模型训练中无法快速拟合收敛。

在 Paddle Fluid 的开发初期，除了框架本身执行机制变化较快外，在各类 Op 的补充实现中也难免出现问题，经常出现某个 Op 的意外问题导致相关的一大类模型的训练均暴露出问题。


#### 为何在快速迭代中模型精度问题难以快速排查 {#为何在快速迭代中模型精度问题难以快速排查}

常规上，我们在类似项目中均有 CI (Continuous Integration) 系统，用以对每个 PR (Pull request）进行快速测试，现实里面，GitHub 上常见的系统包括 Jenkins，Circle CI等。
在项目中，我们会有大量的快速执行的 unittest（单元测试），用以验证系统里面每个小模块的执行正确性。
CI 在常规敏捷开发中是必不可少的测试系统，现有的 DL Framework 都有自己的 CI 系统。
但其对应的 unittest 测试方式在模型层面的训练精度的测试因为有诸多限制而无法实施，具体问题如下

| 问题   | 基于 Unittest 的 CI | DNN 模型测试需求           |
|------|------------------|----------------------|
| 测试粒度 | 一般为 Op 或者单个模块粒度 | 整个 DNN 模型              |
| 测试执行时间 | 分钟级别         | 单个模型训练测试可能需要半小时以上，总体需要数十小时 |
| 验证指标 | 通过与否等简单指标 | 精度、性能、资源消耗等等繁多的指标 |
| 执行环境 | 部分资源混合调度 | 单显卡或者整机独占，避免干扰 |
| 测试对象 | 每个 PR/commit   | 多个 commit 整体测试       |

如此，当时我们在考虑能否在现有的 CI 系统外单独构建一个专用于检测模型效果的验证系统，这也就是后续 CE 系统的由来。


### CE 的设计思路 {#ce-的设计思路}

CE 系统的设计直接针对上面描述的模型测试的需求，对应地

1.  测试粒度以模型为单位
2.  测试指标至少考虑了精度，性能，资源消耗等验证，这些验证会精确到模型训练的每个 batch 或者 epoch
3.  资源调度最初是以整机为要求
4.  测试单位是多个 PR，完全独立于 CI 系统，周期性验证 master branch上最新的代码


### 实施问题 {#实施问题}


### 项目现状 {#项目现状}


### 思考 {#思考}


### Reference {#reference}

-   [PaddlePaddle/continuous_evaluation](https://github.com/PaddlePaddle/continuous_evaluation), forked [Superjomn/continuous_evaluation](https://github.com/Superjomn/continuous_evaluation)
