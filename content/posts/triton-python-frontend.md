+++
title = "OpenAI/Triton 的 Python 前端逻辑的理解"
author = ["Chunwei Yan"]
date = 2022-10-23
tags = ["triton;AI compiler"]
draft = true
+++

_本文中的 OAI/Triton 指的是 OpenAI 的 [triton](https://github.com/openai/triton) 项目（这里区别于 NVIDIA 的 triton inference server），以下直接称为 triton。_

_当前，triton 正在向 MLIR 做迁移，大部分核心代码都在做重构。我个人也在参与其中，持续学习和贡献。我之前并没有接触过 compute bound 的 compiler，因此接下来会在 blog 中记录 triton 里面的一些设计和实现方面的理解。_

_文中的代码主要是 triton 下 [triton-mlir](https://github.com/openai/triton/tree/triton-mlir) 这个分支的代码，由于近期这个分支合并到 master 分支后可能会删除，所以本文中的代码链接均指向我 fork 的 repo 中对应分支的某个 commit 以保证链接稳定可见。_


## OpenAI/Triton 的 Python 前端逻辑的理解 {#openai-triton-的-python-前端逻辑的理解}

triton 的 github repo 的介绍是

> Development repository for the Triton language and compiler

其中， "language" 主要指的就是其 Python 前端支持的 DSL。

triton 的 python DSL 是在 Python 基础上，引入了一些类似 numpy 的原语，整体遵循 pythonic 的风格，对 Python 社区的用户会比较友好。

本文接下来会剖析 triton Python 前端的工作原理。


### triton syntax 简介 {#triton-syntax-简介}

参考官方的 [vec-add](https://github.com/openai/triton/blob/master/python/tutorials/01-vector-add.py) 教程，

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
    # There are multiple 'programs' processing different data. We identify which program
    # we are here:
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    # This program will process inputs that are offset from the initial data.
    # For instance, if you had a vector of length 256 and block_size of 64, the programs
    # would each access the elements [0:64, 64:128, 128:192, 192:256].
    # Note that offsets is a list of pointers:
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

需要关注的是，代码中包含了一些 triton 特有的原语，比如

-   `tl.program_id` ，获取当前 program id，需要关注的是，这里的 program 的基本单位是 thread block/CTA
-   `tl.arange` ，这个操作类似 `numpy.arange` ，都是返回一个 tensor
-   `tl.load` ，这里是从 global memory 加载数据到 register
    -   `ptr` 参数：tensor 类型，表示需要 load 的每个元素的 address
    -   `mask` 参数：tensor 类型，对应 `ptr` 中每个元素，是否执行 load
-   `tl.store` ，从 register 存储数据到 global memory
-   `tl.constexpr` ，用于标记 jit function 的参数的属性，表示其为常数，在编译期必须指定一个 python 的常量
    -   一般用来表示 block 的 size
    -   `tl.constexpr` 的参数，指定不同的值会编译出不同的 kernel cubin

[triton language](https://triton-lang.org/master/python-api/triton.language.html) 页面包含了所有 triton 原语的介绍。


### triton kernel 的 python AST 到 triton IR 的映射方法 {#triton-kernel-的-python-ast-到-triton-ir-的映射方法}


#### triton semantics {#triton-semantics}

[semantic.py](https://github.com/Superjomn/triton/blob/12d60cb4a306e8397ee00717486eb0f36c6eddcb/python/triton/language/semantic.py) 中定义了 triton 支持的所有的原语，具体包含了该原语映射到 triton ir 的方法。

简单的比如上面例子中用到的 `program_id` 原语对应的原语实现是

```python
def program_id(axis: int, builder: ir.builder) -> tl.tensor:
    return tl.tensor(builder.create_get_program_id(axis), tl.int32)
```

其中， `builder` 参数表示的就是 triton IR builder

再比如， triton 的 add 操作：

```python
def add(input: tl.tensor,
        other: tl.tensor,
        builder: ir.builder) -> tl.tensor:
    input, other = binary_op_type_checking_impl(input, other, builder, True, True)
    input_scalar_ty = input.type.scalar
    other_scalar_ty = other.type.scalar

    # offset + ptr
    # ptr + offset
    if other_scalar_ty.is_ptr() and not input_scalar_ty.is_ptr():
        input, other = other, input
    if input_scalar_ty.is_ptr():
        return tl.tensor(builder.create_addptr(input.handle, other.handle), input.type)
    # float + float
    elif input_scalar_ty.is_floating():
        return tl.tensor(builder.create_fadd(input.handle, other.handle), input.type)
    # int + int
    elif input_scalar_ty.is_int():
        return tl.tensor(builder.create_add(input.handle, other.handle), input.type)
    assert False
```

**triton 的原语实现可以有很复杂的实现来精细地描述 python 语法到 tirton ir 之间的构建关系。**
比如上面例子中花了不少篇幅来判定数据类型，主要原因是 triton ir 的类型系统是比较精细的，不同的输入类型可能会映射到不同的 IR Op 上。


#### 解析 Python AST {#解析-python-ast}

基于 Python 的 [ast](https://docs.python.org/3/library/ast.html) 模块， [CodeGenerator](https://github.com/Superjomn/triton/blob/12d60cb4a306e8397ee00717486eb0f36c6eddcb/python/triton/compiler.py#L105) 类中定义了 Python AST 到 triton semantics 之间的调用关系。

<!--list-separator-->

-  常规 AST 转换

    比如 python 中的 `+,-,*,/` 等等操作符会通过如下逻辑映射到对应的 binary 相关的 semantic 函数上，

    ```python
    def visit_BinOp(self, node):
            lhs = self.visit(node.left)
            rhs = self.visit(node.right)
            if isinstance(lhs, triton.language.constexpr):
                lhs = lhs.value
            if isinstance(rhs, triton.language.constexpr):
                rhs = rhs.value
            fn = {
                ast.Add: '__add__',
                ast.Sub: '__sub__',
                ast.Mult: '__mul__',
                ast.Div: '__truediv__',
                ast.FloorDiv: '__floordiv__',
                ast.Mod: '__mod__',
                ast.Pow: '__pow__',
                ast.LShift: '__lshift__',
                ast.RShift: '__rshift__',
                ast.BitAnd: '__and__',
                ast.BitOr: '__or__',
                ast.BitXor: '__xor__',
            }[type(node.op)]
            if self.is_triton_tensor(lhs):
                return getattr(lhs, fn)(rhs, _builder=self.builder)
            elif self.is_triton_tensor(rhs):
                fn = fn[:2] + 'r' + fn[2:]
                return getattr(rhs, fn)(lhs, _builder=self.builder)
            else:
                return getattr(lhs, fn)(rhs)
    ```

<!--list-separator-->

-  Function Call 转换

    Call AST Node 的 [visit_Call](https://github.com/Superjomn/triton/blob/12d60cb4a306e8397ee00717486eb0f36c6eddcb/python/triton/compiler.py#L666) 逻辑比较重要，简化的逻辑如下

    ```python
    self.builtins = {
                'range': range,
                'min': triton.language.minimum,
                'float': float,
                'int': int,
                'print': print,
                'isinstance': isinstance,
                'getattr': getattr,
            }

    def visit_Call(self, node):
        fn = self.visit(node.func)
        if isinstance(fn, triton.language.constexpr):
            fn = fn.value
        args = [self.visit(arg) for arg in node.args]
        if isinstance(fn, triton.runtime.JITFunction):
            # 1. Process the case where calling another triton.jit kernel
        if hasattr(fn, '__self__') and self.is_triton_tensor(fn.__self__) or \
           sys.modules[fn.__module__] is triton.language.core or \
           isinstance(fn, triton.language.extern.ExternalFunction):
            # 2. Process the case where calling triton primitives
        if fn in self.builtins.values():
            args = [arg.value if isinstance(arg, triton.language.constexpr) else arg
                        for arg in args]
            # 3. Process the cases when calling Python range, min, float methods on triton constexpr

        # 4. Also works on the other cases, like calling other native python function, will eval here
        return fn(*args, **kws)
    ```

    具体可以看出 triton 前端灵活性还比较高，支持如下几种情况

    <!--list-separator-->

    -  1. 如果 Call 的函数 （fn） 是 `triton.runtime.JITFunction` ，则 IR 中增加 Call node

        **`triton.jit` kernel 是支持直接 call 另外一个 `kernel.jit` kernel 的，这给 kernel fusion 带来了一些可能。**

        具体的例子可以参考官方 [matmul 教程](https://github.com/openai/triton/blob/master/python/tutorials/03-matrix-multiplication.py#L187)

        ```python
        # Define a leaky_relu activation kernel
        @triton.jit
        def leaky_relu(x):
            x = x + 1
            return tl.where(x >= 0, x, 0.01 * x)

        # Call leaky_relu in matmul_kernel
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
            # ...
            if ACTIVATION == "leaky_relu":
                accumulator = leaky_relu(accumulator)
        ```

        上面例子中有两点值得关注

        1.  在 `matmul_kernel` jit kernel 中中直接调用了另外一个 jit kernel： `leaky_relu`
            -   理论上可以自由地互相调用
        2.  `ACTIVATION` 在编译时传入了一个 string 并且在 jit kernel 中做了判定，这里如上文讲 `tl.constexpr` 在编译期会替换为 python 的常数，理论上其实任何 python 的数据类型都支持，因为不会混入到 triton ir 中

    <!--list-separator-->

    -  2. 如果 fn 是 `triton.language.core` 模块下的，则编译时 eval 对应的 semantic method

        上文介绍了 triton 的 `semantics.py` 包含了大量的自定义的原语(python method)，这些 method 需要在 kernel 中原地展开，以实现用户调用的原语对应映射到 triton ir的逻辑。

        这个分支就实现了这样的效果，判定用户调用的 method 的属性，如果发现是 triton 自定义的原语，则 eval 对应的方法，进而触发展开 triton ir 的效果。

    <!--list-separator-->

    -  3. 其他情况（比如调用了 python 定义的函数），则编译时 eval

        任何其他情况， triton 都会尝试原地 eval 这个函数，这样在编译时直接在 IR 上产生效果。可以想象，这样的情况可以包括调用一些 input 在编译期已经确定了的 python method，比如 `n = math.min(1, 2)` 然后 `n` 参与到 python 的 semantic 函数的调用，这些是没有问题的；
        当然，eval 也可能失败，这种情况可能是编译时函数的输入还没有确定，python eval 应该会失败。

        第 2 种情况和第 3 种情况很类似，都是编译期时 eval，区别是情况 2 会处理 kernel 中非 `tl.constexpr` 的参数，将之作为 variable (MLIR 中的 value) 插入到 triton ir中，如此可以支持 dynamic shape 的情况。而情况 3 会将当时的 python 变量的值直接插入到 triton ir 中，变成对应的 `arith.constant` node。


### triton 中的 control flow 的支持原理 {#triton-中的-control-flow-的支持原理}

triton 中的 control flow 的支持方法总体上很类似上文 Function Call 的支持方法，区别是一些 python 语法以及 triton 原语的区别。
tirton 支持 for, if-else, select, while 等 control-flow 的操作，详细的处理代码依旧位于 [CodeGenerator](https://github.com/Superjomn/triton/blob/12d60cb4a306e8397ee00717486eb0f36c6eddcb/python/triton/compiler.py#L105) 类定义中。


#### for {#for}

triton 支持 `for i in range(...)` 的 forloop，对应转化为 MLIR 中的 `scf.for` 。

比如，

```python
for i in range(0, 128):
    ...
```

会映射到

```llvm
scf.for %i = 0 to 128 step 1 {

    scf.yield ...
}
```

这里有个细节，如果是静态的 forloop（range 的参数都是常数），则 triton 会在 step 数较小时（目前代码里面是 10）直接在 编译期间 unroll。


#### if-else {#if-else}

Kernel 中的 if-else 语句会对应转换为 MLIR `scf.if` 节点，细节类似 `scf.for` 。

这里有个细节，如果是静态的 if-else（condition 为常量），则编译期间会直接确定分支并清理掉 if-else 分支。


#### select {#select}

这里的 select 不是 python 的 `v0 if cond else v1` 语句，但语义是一样的，但 triton 用了 `tl.where` 这个原语来做了实现，具体可以参考 [tl.where](https://triton-lang.org/master/python-api/triton.language.html#indexing-ops)。


#### while {#while}

while 语句会对应转化为 MLIR 的 `scf.where` node，这里不再展开。


### References {#references}

-   [Introducing Triton: Open-Source GPU Programming for Neural Networks](https://openai.com/blog/triton/)
-   [triton/triton-mlir 分支](https://github.com/openai/triton/tree/triton-mlir)，包含最新的代码，但近期可能会 merge 到 master 分支
-   [Superjomn/triton/triton-mlir](https://github.com/Superjomn/triton/tree/12d60cb4a306e8397ee00717486eb0f36c6eddcb)，我 Repo 中的代码较新的一个 commit，文中引用到的代码会稳定存在
