+++
title = "Python 高端写法记录"
author = ["Chunwei Yan"]
date = 2023-02-22
tags = ["python", "tech"]
draft = true
+++

在阅读 Pytorch 代码时，发现很多 Python 的新的用法比较有趣，这里整理和记录一些有趣的用法。

这篇文章会持续更新 :-)

<!--more-->


## `functools` - Misc 功能 {#functools-misc-功能}


### @total_ordering shortcut to make class orderable {#total-ordering-shortcut-to-make-class-orderable}

为了让一个 Class 能够比较，我们正常需要定义一堆 slots， `__gt__()`, `__ge__()`,~\__lt__()~, `__le__()`, `__eq__()` 等等。
但其实只需要 `__eq__()` 和 其他任一方法(比如 `__gt__()` ) 便可以组合出其他方法。

`@total_ordering` 便用于这类场景的 helper，如下代码

```python
from functools import total_ordering

@total_ordering
class Stuff:
    def __init__(self, x):
        self.x = x
    def __eq__(self, other:"Stuff"):
        return self.x == other.x
    def __gt__(self, other:"Stuff"):
        return self.x > other.x

a = Stuff(0)
b = Stuff(1)

print(f"a < b? {a < b}")
```

```text
a < b? True
```


### partial() to reuse a existing method by fixing some arguments {#partial-to-reuse-a-existing-method-by-fixing-some-arguments}

```python
from functools import partial

def func0(a, b):
    print(f"a:{a}, b:{b}")

func1 = partial(func0, a = 0)

print(func1)

func1(b=10)
# reset argument a
func1(a=1, b=10)
```

```text
functools.partial(<function func0 at 0x7faf700531e0>, a=0)
a:0, b:10
a:1, b:10
```

从例子可以看出， `partial` 效果有点类似设定了默认值，新的函数依旧依旧可以设置被 fixed 的 argument。


### @warps to help define better decorators {#warps-to-help-define-better-decorators}

之前，定义 decorator 的 naivie 的方式是

```python
def decorator(func):
    def actual_func(*args, **kwargs):
        print(f"Before Calling {func.__name__}")
        func(*args, **kwargs)
        print(f"After Calling {func.__name__}")

    return actual_func

@decorator
def greet(name):
    print(f"Hello, {name}!")

greet("Martin")
```

```text
import codecs, os;__pyfile = codecs.open('''/var/folders/41/6sd124ws3t982vl6bmw1gvjxdmh6b5/T/py1mTNF0''', encoding='''utf-8''');__code = __pyfile.read().encode('''utf-8''');__pyfile.close();os.remove('''/var/folders/41/6sd124ws3t982vl6bmw1gvjxdmh6b5/T/py1mTNF0''');exec(compile(__code, '''/var/folders/41/6sd124ws3t982vl6bmw1gvjxdmh6b5/T/py1mTNF0''', 'exec'));
Before Calling greet
Hello, Martin!
After Calling greet
```

如上， `greet` 方法实现了既有的功能，但有一个问题，当执行

```python
print(greet.__name__)
```

```text
import codecs, os;__pyfile = codecs.open('''/var/folders/41/6sd124ws3t982vl6bmw1gvjxdmh6b5/T/py9bWq4K''', encoding='''utf-8''');__code = __pyfile.read().encode('''utf-8''');__pyfile.close();os.remove('''/var/folders/41/6sd124ws3t982vl6bmw1gvjxdmh6b5/T/py9bWq4K''');exec(compile(__code, '''/var/folders/41/6sd124ws3t982vl6bmw1gvjxdmh6b5/T/py9bWq4K''', 'exec'));
actual_func
```

也就是原有的 `greet` 的属性都变化了（其实是 greet 替换成了 actual_func），这个不是我们希望的。
`warps` 方法就用来将 `actual_func` 伪装回 `greet` ，让装饰器看起来没有改变表象的东西。

简单用法如下

```python
from functools import wraps

def decorator(func):
    @wraps(func)
    def actual_func(*args, **kwargs):
        print(f"Before Calling {func.__name__}")
        func(*args, **kwargs)
        print(f"After Calling {func.__name__}")

    return actual_func

@decorator
def greet(name):
    print(f"Hello, {name}!")

print(greet.__name__)
```

```text
greet
```

哈哈，greet 还是 greet。


### `@lru_cache` {#lru-cache}

默认 `maxsize=128` ，可以设置 `maxsize=None` 来确定无限 cache。

```python
@functools.lru_cache(maxsize=1000)
def factorial(n):
    return n * factorial(n-1) if n else 1
```


## Reference {#reference}

-   [Useful Decorators and Functions in Python's Functools](https://dzone.com/articles/functools-useful-decorators-amp-functions-1)
-   [Functools — The Power of Higher-Order Functions in Python](https://towardsdatascience.com/functools-the-power-of-higher-order-functions-in-python-8e6e61c6e4e4)
