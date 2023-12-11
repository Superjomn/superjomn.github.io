+++
title = "TorchDynamo 简单实践(1)"
author = ["Chunwei Yan"]
date = 2023-03-01
tags = ["pytorch", "dynamo"]
draft = true
+++

<div class="ox-hugo-toc toc">

<div class="heading">Table of Contents</div>

- [背景知识](#背景知识)
- [用户 API](#用户-api)
    - [optimize()](#optimize)
    - [带 Control Flow 的简单模型的 compile](#带-control-flow-的简单模型的-compile)
- [FYI](#fyi)
    - [Debug on Dynamo](#debug-on-dynamo)

</div>
<!--endtoc-->

Pytorch 2.0 发布后，增加了 TorchDynamo 和 TorchInductor 两个新东西，前者是一个比较长远的动态图的解决方案，用于直接分析 Python 的 binary code 并部分转换为 FX Graph；
后者则是在 FX Graph 后接着优化的编译器。

对于终端用户，这两者统一的界面就是 `torch.compile` ，一个高层的 API，往下就是完全透明的黑盒。这个透明性对用户是好的，但优化者，还是得要深入其中。

本文尝试从实践的角度，大致理清楚 TorchDynamo 的原理和修改优化方法。


## 背景知识 {#背景知识}

TorchDynamo 的 workflow 如下，

{{< figure src="/static/TorchDynamo/TorchDynamo.png" >}}

原有的 [torch.fx](https://pytorch.org/docs/stable/fx.html) 通过 symbolic tracer 追踪 PyTorch 模型的执行步骤，并最终用 FX Graph 表示成 Backend 可以理解的格式。不过其本质上跟 TorchScript 类似，都只能表示 Torch Op 相关的操作，而用户在动态图里经常混杂其他无法表示的 Python 代码，对于这些 `torch.fx` 就歇菜了。

TorchDynamo 对比 `torch.fx` 最大的变化是

1.  直接面向 Python JIT 实时的 binary code 进行切图和编译，从当前 Frame 里面可以获取部分变量值进行进一步的 code simplification（比如 Prune 一些 if-else）
2.  直接修改 Python 执行的 Frame 的 Code object，将 Backend 编译好的 function 跟其他 Torch 无法表示的 Pythonic 的操作混合执行，水乳交融了

下面可能会涉及一些 Python VM 的概念，可以参考 [Python VM 执行方式简要探索](https://superjomn.github.io/posts/python-vm-brief-introduction/) 。


## 用户 API {#用户-api}

参考 [code](https://github.com/pytorch/pytorch/blob/b5ff41a47a36def38b01aec8a2aaba2532833f35/torch/__init__.py#L1441)， `torch.compile` 的实现主要是 `_dynamo.optimize`


### optimize() {#optimize}

```python
import torch
from torch import _dynamo
from torch._dynamo import optimize

# get the available backends
print(_dynamo.list_backends())
```


### 带 Control Flow 的简单模型的 compile {#带-control-flow-的简单模型的-compile}

常规的不带 control flow 的 TorchDynamo 应该都能解决，这里我们比较关注带 control flow 的情况


#### Control Flow 导致的 graph break {#control-flow-导致的-graph-break}

```python
def foo2(a:torch.tensor, b:torch.tensor):
  x = a + b
  if b.sum() < 0:
    x = x * -1
  if a.sum() < 0:
    x = x * -1
  x = 2 * x
  return x

foo2_ = optimize(my_compiler)(foo2)
```

为了方便分析，我们用一个表格的形式

| lineno | code              |
|--------|-------------------|
| 2      | `x = a + b`       |
| 3      | `if b.sum() < 0:` |
| 4      | `x = x * -1`      |
| 5      | `if a.sum() < 0:` |
| 6      | `x = x * -1`      |
| 7      | `x = 2 * x`       |
| 8      | `return x`        |

执行之， Dynamo 应该能够捕捉 JIT 的 Python binary code，执行相应的 graph。

<!--list-separator-->

-  执行一种 case

    ```python
    torch._dynamo.reset() # reset all che compilation cache
    my_graph_id = 0

    a = torch.ones((2, 3))
    b = torch.ones((2, 3))

    # It should tigger only one case of the if-else
    foo2_(a, b)
    ```

    这里 a, b 均是 1，应该只会执行 kernel 中 2,3,5,7,8 行，其中

    -   第 3 行判定 condition 为 false，因此没有进入 then block
    -   第 5 行同理

    查看编译的 FX Graph，注意其中对应的代码行已经在生成的代码的注释里了

    ```python
    my_compiler() called with FX graph-0:
    class GraphModule(torch.nn.Module):
        def forward(self, a : torch.Tensor, b : torch.Tensor):
            # File: <ipython-input-43-f6e4dc936826>:2, code: x = a + b
            add = a + b;  a = None

            # File: <ipython-input-43-f6e4dc936826>:3, code: if b.sum() < 0:
            sum_1 = b.sum();  b = None
            lt = sum_1 < 0;  sum_1 = None
            return (add, lt


    my_compiler() called with FX graph-1:
    class GraphModule(torch.nn.Module):
        def forward(self, a : torch.Tensor):
            # File: <ipython-input-43-f6e4dc936826>:5, code: if a.sum() < 0:
            sum_1 = a.sum();  a = None
            lt = sum_1 < 0;  sum_1 = None
            return (lt,)


    my_compiler() called with FX graph-2:
    class GraphModule(torch.nn.Module):
        def forward(self, x : torch.Tensor):
            # File: <ipython-input-43-f6e4dc936826>:7, code: x = 2 * x
            mul = 2 * x;  x = None
            return (mul,)
    ```

    可以看到，这次执行，TorchDynamo 给切了 3 张 graph：

    1.  对应 2,3 行代码
    2.  对应第 5 行代码
    3.  对应第 7 行代码

    看起来 TorchDynamo 会因为 control flow 直接进行 graph break，具体 break 的方法放到后面 python binary code 进行讨论。

<!--list-separator-->

-  执行所有 4 种 case

    上面我们讨论了一种情况，比较所有输入的情况下，control flow 会带来的影响。

    ```python
    torch._dynamo.reset() # reset all che compilation cache
    my_graph_id = 0

    # It should tigger all the four combinations of the if-conditions
    foo2_(a, b)
    foo2_(a, -b)
    foo2_(-a, b)
    foo2_(-a, -b)
    ```

    这 4 个 case 理论上会激活 Kernel 中所有 2~8 行代码。

    实际输出了 5 个 graph：

    ```python
    my_compiler() called with FX graph-0:
    class GraphModule(torch.nn.Module):
        def forward(self, a : torch.Tensor, b : torch.Tensor):
            # File: <ipython-input-43-f6e4dc936826>:2, code: x = a + b
            add = a + b;  a = None

            # File: <ipython-input-43-f6e4dc936826>:3, code: if b.sum() < 0:
            sum_1 = b.sum();  b = None
            lt = sum_1 < 0;  sum_1 = None
            return (add, lt)


    my_compiler() called with FX graph-1:
    class GraphModule(torch.nn.Module):
        def forward(self, a : torch.Tensor):
            # File: <ipython-input-43-f6e4dc936826>:5, code: if a.sum() < 0:
            sum_1 = a.sum();  a = None
            lt = sum_1 < 0;  sum_1 = None
            return (lt,)


    my_compiler() called with FX graph-2:
    class GraphModule(torch.nn.Module):
        def forward(self, x : torch.Tensor):
            # File: <ipython-input-43-f6e4dc936826>:7, code: x = 2 * x
            mul = 2 * x;  x = None
            return (mul,)


    my_compiler() called with FX graph-3:
    class GraphModule(torch.nn.Module):
        def forward(self, a : torch.Tensor, x : torch.Tensor):
            # File: <ipython-input-43-f6e4dc936826>:4, code: x = x * -1
            mul = x * -1;  x = None

            # File: <ipython-input-43-f6e4dc936826>:5, code: if a.sum() < 0:
            sum_1 = a.sum();  a = None
            lt = sum_1 < 0;  sum_1 = None
            return (mul, lt)


    my_compiler() called with FX graph-4:
    class GraphModule(torch.nn.Module):
        def forward(self, x : torch.Tensor):
            # File: <ipython-input-43-f6e4dc936826>:6, code: x = x * -1
            mul = x * -1;  x = None

            # File: <ipython-input-43-f6e4dc936826>:7, code: x = 2 * x
            mul_1 = 2 * mul;  mul = None
            return (mul_1,)
    ```

    这些 graph 与原有 kernel 的对应关系：

    {{< figure src="/static/TorchDynamo/2.png" >}}

    简单理解如果两个 if 的 condition 都是 true，那激活 graph0, graph3, graph4 便可。

    从上图看，可以看出 graph break 有如下几个特点：

    1.  一个 kernel 切出的 graph 覆盖范围可能会重叠
    2.  TorchDynamo 基本沿着 python code 从上之下切出连续的 code block 形成 graph，所以所谓的 graph(FX Graph) 只是个数据结构，并非真正的图
        -   沿着 graph 不会包含 control flow 的角度去看， **FX Graph 很像 Compiler 里面的 Basic Block**

    从 AST 层面分析 graph break 还不算直观，毕竟 TorchDynamo 主要是从 bytecode 层次转的 graph，接下来我们 bytecode 角度分析。

<!--list-separator-->

-  Python bytecode 角度分析

    ```python
    import dis
    dis.dis(foo2)
    ```

    ```text
     2           0 LOAD_FAST                0 (a)
                 2 LOAD_FAST                1 (b)
                 4 BINARY_ADD
                 6 STORE_FAST               2 (x)

     3           8 LOAD_FAST                1 (b)
                10 LOAD_METHOD              0 (sum)
                12 CALL_METHOD              0
                14 LOAD_CONST               1 (0)
                16 COMPARE_OP               0 (<)
                18 POP_JUMP_IF_FALSE       28

     4          20 LOAD_FAST                2 (x)
                22 LOAD_CONST               2 (-1)
                24 BINARY_MULTIPLY
                26 STORE_FAST               2 (x)

     5     >>   28 LOAD_FAST                0 (a)
                30 LOAD_METHOD              0 (sum)
                32 CALL_METHOD              0
                34 LOAD_CONST               1 (0)
                36 COMPARE_OP               0 (<)
                38 POP_JUMP_IF_FALSE       48

     6          40 LOAD_FAST                2 (x)
                42 LOAD_CONST               2 (-1)
                44 BINARY_MULTIPLY
                46 STORE_FAST               2 (x)

     7     >>   48 LOAD_CONST               3 (2)
                50 LOAD_FAST                2 (x)
                52 BINARY_MULTIPLY
                54 STORE_FAST               2 (x)

     8          56 LOAD_FAST                2 (x)
                58 RETURN_VALUE
    ```

    注意其中出现了两次 `POP_JUMP_IF_FALSE` ，这个对应着 AST 里面的 `if` ，简单理解下其语义， `POP_JUMP_IF_FALSE 48` 可以理解为，如果 stack 顶部的 value 为 false，则 goto 到第 48 行 opcode，也就是 Python 第 7 行代码，这个是对应的。

    在 bytecode 上看 TorchDynamo 的逻辑比较清晰，上面的例子的中切出 5 个 graph 如下对应

    {{< figure src="/static/TorchDynamo/3.png" >}}

    可以看出明确的规律：

    1.  先沿着 `POP_JUMP_IF_FALSE` 为边界切割几个大的 graph；不会有任何 graph 能在中间 hold 一个 `POP_JUMP_IF_FALSE`
        -   上图 graph0, graph3, graph4 满足这个规律
    2.  `POP_JUMP_IF_FALSE` 的 argument 中表示的 goto 的行号会导致进一步的 graph break
        -   上图 graph1, graph2 满足此规律

    上面只是从静态的角度做的解释。


#### <span class="org-todo todo TODO">TODO</span> InlineCall 融合一些小的 function frame {#inlinecall-融合一些小的-function-frame}


#### constant control flow 的化简 {#constant-control-flow-的化简}


## FYI {#fyi}


### Debug on Dynamo {#debug-on-dynamo}

如下设置开启可以让 TorchDynao 打印出不少中间结果帮助理解

```python
from torch._dynamo import config
import logging

config.log_level = logging.INFO
config.output_code = True
```
