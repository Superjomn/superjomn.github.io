+++
title = "Python VM 执行方式简要探索"
author = ["Chunwei Yan"]
date = 2023-02-24
tags = ["python"]
draft = true
+++

<div class="ox-hugo-toc toc">

<div class="heading">Table of Contents</div>

- [Python Code Object](#python-code-object)
    - [有用的 attribute](#有用的-attribute)
    - [Python Bytecode 理解](#python-bytecode-理解)
- [Python Frame](#python-frame)
    - [修改 Frame 的 Code Object](#修改-frame-的-code-object)
- [总结](#总结)
- [FYI](#fyi)

</div>
<!--endtoc-->

最近在看 TorchDynamo 的东西，里面需要对 Python 执行机制有一些了解，所以单独拆开放到了这篇文章里。

本文会从可复现的角度，多一些可以执行的例子。


## Python Code Object {#python-code-object}

这里是简单的介绍，详细的可以参考 [code objects](https://leanpub.com/insidethepythonvirtualmachine/read#leanpub-auto-code-objects) 这本书的章节。

Code object 用来记录 Python 的 byte code，对应的粒度是 Block，这里的 Block 可以包含从 Module 到 Class definition 到 Function body 的各类结构（有别于编译器里面的 BasicBlock）。 可以理解 Code object 是可以嵌套的。

我们先从一个简单的函数开始

<!--more-->

```python
def foo(name, age):
    ''' Get the information of a person. '''
    born = 2023 - age
    return f"hello {name}, born at {born}!"
```

查看其 Code object:

```python
foo.__code__
```

```text
<code object foo at 0x7f4da6f59500, file "/tmp/ipykernel_2935902/3583013317.py", line 1>
```


### 有用的 attribute {#有用的-attribute}

那 `foo.__code__` 中包含的有用的 attribute 如下

```python
dir(foo.__code__)
```

得到

```python
['__class__',
 '__delattr__',
 '__dir__',
 '__doc__',
 '__eq__',
 '__format__',
 '__ge__',
 '__getattribute__',
 '__gt__',
 '__hash__',
 '__init__',
 '__init_subclass__',
 '__le__',
 '__lt__',
 '__ne__',
 '__new__',
 '__reduce__',
 '__reduce_ex__',
 '__repr__',
 '__setattr__',
 '__sizeof__',
 '__str__',
 '__subclasshook__',
 'co_argcount',
 'co_cellvars',
...
 'co_nlocals',
 'co_posonlyargcount',
 'co_stacksize',
 'co_varnames',
 'replace']
```

其中比较多打交道的属性如下

-   **`co_names`** is a tuple containing global attributes and methods used inside the scope,
-   **`co_varname`** is the tuple containing local variable names used in function,
-   **`co_consts`** returns the literals used by bytecode.

对应属性的内容如下

```python
call_foo.__code__.co_names
```

`co_names` 是空的，因为 `foo` 函数里面没有调用任何的外部函数或属性。

```python
foo.__code__.co_varnames
```

`co_varnames` 比较丰富，看到 local scope 可以用的 variable name 都在里面，包括两个 argument： name 和 age，以及一个 local variable： born。

```python
foo.__code__.co_consts
```

`co_consts` 意料外有好几个，一类是 code 里面用到的 **2023** 这个常量，另外几个字符串是 string format 里面分割开的几个字段。


### Python Bytecode 理解 {#python-bytecode-理解}

接下来我们尝试理解下简单的 Python bytecode，接着上面的例子。

```python
dis.dis(foo)
```

```text
 2           0 LOAD_CONST               1 (2023)
             2 LOAD_FAST                1 (age)
             4 BINARY_SUBTRACT
             6 STORE_FAST               2 (born)

 3           8 LOAD_CONST               2 ('hello ')
            10 LOAD_FAST                0 (name)
            12 FORMAT_VALUE             0
            14 LOAD_CONST               3 (', born at ')
            16 LOAD_FAST                2 (born)
            18 FORMAT_VALUE             0
            20 LOAD_CONST               4 ('!')
            22 BUILD_STRING             5
            24 RETURN_VALUE
```

上面 dump 出来的内容包含了 `dis` 增加的一些 human readable 的 hint，具体内容可以分为 3 列：

1.  表示原始 Python 源代码中的行号
2.  bytecode 中的行号以及对应的 opcode
3.  argument ID，圆括弧里面是一些 hint

这里需要提一下， Python 的执行方式是基于 Stack 而非 Register，带来的好处就是 bytecode 逻辑非常简单，常见的操作如下

-   在执行一个 Opcode 前，将其所需的 variable 压栈
-   执行这个 Opcode 时，从 stack 中 Pop 所需数目的 variable
-   将执行结果压入 stack 中，备后续使用

上面的前三行 code

```nil
2           0 LOAD_CONST               1 (2023)
            2 LOAD_FAST                1 (age)
            4 BINARY_SUBTRACT
```

表示的是

1.  将 1st 常量 2023 压栈
2.  将 1st 变量 age 压栈
3.  Pop 两个值，并执行 2023 - age
4.  将结果压栈

Opcode 有一些具体的含义，比如其中几个比较重要的

-   `LOAD_CONST` 表示是从 Code object 的 `co_consts` 里面 load
    -   参考上节，~co_consts~ 第 1 个 value 是 2023， 因此 `LOAD_CONST 1` 表示 Load 2023 进 stack，跟末尾的 hint 对应起来了
-   `LOAD_FAST` 是从 `co_varnames` 里面 Load 到 stack
-   `STORE_FAST` 表示将 stack 的 head 中的值 Store 到 `co_varnames` 里面的一个 variable
-   `RETURN_VALUE` 表示将 stack 中包含的所有值返回

完整的 opcode 及对应的 Python 的处理行为可以参考 [Python/generated_cases.c.h](https://github.com/python/cpython/blob/22b8d77b98a5944e688be0927b8139c49d4a7257/Python/generated_cases.c.h)，内容非常清晰，截取 `LOAD_FAST` 对应的代码：

```C++
TARGET(LOAD_FAST) {
            PyObject *value;
            value = GETLOCAL(oparg);
            assert(value != NULL);
            Py_INCREF(value);
            STACK_GROW(1);
            POKE(1, value);
            DISPATCH();
        }
```


## Python Frame {#python-frame}

Python Code Object 存储了待执行的 Python bytecode，但这些 bytecode 无法直接执行，还需要专门的 Interpreter 机制。
这跟 C/C++ 完全不同，毕竟 bytecode 和 machine code 完全是两码事。

为了执行 bytecode， Python 对应有 Frame Object 的数据结构，简单可以认为 Frame Object 跟 Code Object 对应，前者 hold 执行一个 Code Object 所需要的所有 runtime 的信息，而后者则记录了具体需要执行的 bytecode。
因此，当从一个 Block 跳到另外一个 Block 时候，会有 Frame 的切换，例如，function call 时，会先 hold 住当前的 Frame，创建一个新的 Frame 接着执行该 function 对应的 Code Object； 执行完毕，则跳回先前的 Frame。

Frame 具体数据结构我们需要参考下 [pycore_frame.h/_frame](https://github.com/python/cpython/blob/main/Include/internal/pycore_frame.h#L16) 相关的 code：

```C++
struct _frame {
    PyObject_HEAD
    PyFrameObject *f_back;      /* previous frame, or NULL */
    struct _PyInterpreterFrame *f_frame; /* points to the frame data */
    PyObject *f_trace;          /* Trace function */
    int f_lineno;               /* Current line number. Only valid if non-zero */
    char f_trace_lines;         /* Emit per-line trace events? */
    char f_trace_opcodes;       /* Emit per-opcode trace events? */
    char f_fast_as_locals;      /* Have the fast locals of this frame been converted to a dict? */
    /* The frame data, if this frame object owns the frame */
    PyObject *_f_frame_data[1];
};
```

其中 `f_back` 指针用来 chain 多个跳转的 Frame，这样上面的 function call 才可以实施。

`f_frame` 记录了一个 Frame 具体的信息，接着看下如下 code

```C++
typedef struct _PyInterpreterFrame {
    PyCodeObject *f_code; /* Strong reference */
    struct _PyInterpreterFrame *previous;
    PyObject *f_funcobj; /* Strong reference. Only valid if not on C stack */
    PyObject *f_globals; /* Borrowed reference. Only valid if not on C stack */
    PyObject *f_builtins; /* Borrowed reference. Only valid if not on C stack */
    PyObject *f_locals; /* Strong reference, may be NULL. Only valid if not on C stack */
    PyFrameObject *frame_obj; /* Strong reference, may be NULL. Only valid if not on C stack */
    // NOTE: This is not necessarily the last instruction started in the given
    // frame. Rather, it is the code unit *prior to* the *next* instruction. For
    // example, it may be an inline CACHE entry, an instruction we just jumped
    // over, or (in the case of a newly-created frame) a totally invalid value:
    _Py_CODEUNIT *prev_instr;
    int stacktop;  /* Offset of TOS from localsplus  */
    uint16_t yield_offset;
    char owner;
    /* Locals and stack */
    PyObject *localsplus[1];
} _PyInterpreterFrame;
```

这里比较明确的是

-   `f_code` 肯定是指向对应的 Code Object
-   `f_globals`, `f_locals` 应该直接对应到 Code Object 里面的 `co_names` 和 `co_varnames`
-   `stacktop` 对应着 stack 中的 top 位置


### 修改 Frame 的 Code Object {#修改-frame-的-code-object}

得益于 [PEP 523](https://peps.python.org/pep-0523/)，从 Python 3.6 开始，一个 `PyEval_EvalFrameEx()` 函数加入了 Python API，不同于之前的 `PyEval_EvalFrameDefault()` ， 新函数允许用户自定义 Frame Object 执行过程。

类似 TorchDynamo，与 Python 交互最核心的也是通过这个 API。

API 的实现很简单：

```C++
PyObject *
PyEval_EvalFrameEx(PyFrameObject *frame, int throwflag)
{
    PyThreadState *tstate = PyThreadState_GET();
    return tstate->interp->eval_frame(frame, throwflag);
}
```

只要用户设置了自定义的 interpreter，那就执行自定义的 `eval_frame` 逻辑，这一点也是 TorchDynamo 的核心。


## 总结 {#总结}

本文主要介绍了 Python 执行机制中 Code Object 和 Frame Object 两个重要的概念，具体特点如下

-   Code Object 和 Frame Object 对应到 Block 粒度
-   Code Object 主要记录了 bytecode 以及 `co_names`, `co_varnames` 等一大类静态信息
-   Frame Object 跟 Code Object 基本一一对应，记录了执行所需的信息，当出现类似 function call，新的 Frame Object 会创建接着执行，完毕后再返回当前 Frame


## FYI {#fyi}

-   [Inside the Python Virtual Machine](https://leanpub.com/insidethepythonvirtualmachine/read#leanpub-auto-code-objects) ，一本详细讲解 Python VM 的书
-   [Python behind the scenes #1: how the CPython VM works](https://tenthousandmeters.com/blog/python-behind-the-scenes-1-how-the-cpython-vm-works/)
