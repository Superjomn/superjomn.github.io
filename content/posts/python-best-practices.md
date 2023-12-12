+++
title = "Best Practices for Python Programming (Continuously Updated)"
author = ["Chunwei Yan"]
date = 2023-02-22
tags = ["python", "tech"]
draft = false
+++

When delving into the codebases of some successful large Python projects such as PyTorch, I am consistently impressed by their code -- whether it's clean yet precise, or leveraging lesser-known built-in or third-party packages to significantly enhance functionality.

High-quality code snippets, handy packages, and modules have greatly facilitated my work. In this blog, I'll be sharing noteworthy findings and insights learned from the open-source codebase.

<!--more-->


## Basics {#basics}


### `__new__` {#new}

The `__new__` method is used for creating a new instance of a class. It is a static method that gets called before the `__init__` method.

The default `__new__` method could be

```python
class MyClass:
    def __new__(cls, *args, **kwargs):
        instance = super(MyClass, cls).__new__(cls, *args, **kwargs)
        return instance
```

Note that, different from `__init__`, whose first argument is an instance `self`, `__new__`'s first argument is a class.

You can override `__new__` if something special need to be done with the object creation.

There are some classical use cases for the `__new__` method:


#### Singleton Pattern {#singleton-pattern}

```python
class Singleton:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

# Usage
singleton1 = Singleton()
singleton2 = Singleton()
print(singleton1 is singleton2)
```

```text
True
```


#### Subclassing Immutable Types {#subclassing-immutable-types}

When subclassing immutable types like `str`, `int`, `unicode` or `tuple`, the properties of immutable cannot be changed after they are created, you can override `__new__` instead:

```python
class UpperStr(str):
    def __new__(cls, value):
        return str.__new__(cls, value.upper())

# Usage
upper_string = UpperStr("hello")
print(upper_string)  # Output: HELLO
```

```text
HELLO
```


#### Factory Methods {#factory-methods}

`__new__` can be used to implement factory methods that return instances of different classes based on input parameters.

```python
class Shape:
    def __new__(cls, *args, **kwargs):
        if cls is Shape:
            shape_type = args[0]
            if shape_type == 'circle':
                return Circle()
            elif shape_type == 'square':
                return Square()
        return super(Shape, cls).__new__(cls, *args, **kwargs)

class Circle(Shape):
    pass

class Square(Shape):
    pass

# Usage
shape = Shape('circle')
print(isinstance(shape, Circle))  # Output: True
```

```text
True
```


## Handy modules or packages {#handy-modules-or-packages}


### `functools` {#functools}


#### `partial` to get new function by partially fixing some arguments of an existing one {#partial-to-get-new-function-by-partially-fixing-some-arguments-of-an-existing-one}

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
functools.partial(<function func0 at 0x7fa3980e31e0>, a=0)
a:0, b:10
a:1, b:10
```


#### `@warps` to help define better decorators {#warps-to-help-define-better-decorators}

Below is a classical way to define an decorator

```python
def decorator(func):
    def actual_func(*args, **kwargs):
        ''' The actual func. '''
        print(f"Before Calling {func.__name__}")
        func(*args, **kwargs)
        print(f"After Calling {func.__name__}")

    return actual_func

@decorator
def greet(name):
    ''' The greet func. '''
    print(f"Hello, {name}!")

greet("Martin")
```

```text
import codecs, os;__pyfile = codecs.open('''/var/folders/41/6sd124ws3t982vl6bmw1gvjxdmh6b5/T/pyNDoRTx''', encoding='''utf-8''');__code = __pyfile.read().encode('''utf-8''');__pyfile.close();os.remove('''/var/folders/41/6sd124ws3t982vl6bmw1gvjxdmh6b5/T/pyNDoRTx''');exec(compile(__code, '''/var/folders/41/6sd124ws3t982vl6bmw1gvjxdmh6b5/T/pyNDoRTx''', 'exec'));
Before Calling greet
Hello, Martin!
After Calling greet
```

The name and docstring of the decorated function will be hidden in the decorator function, and this makes the usage a bit opaque when debugging.

```python
print(greet.__name__)
print(greet.__doc__)
```

```text
import codecs, os;__pyfile = codecs.open('''/var/folders/41/6sd124ws3t982vl6bmw1gvjxdmh6b5/T/py7wwbdu''', encoding='''utf-8''');__code = __pyfile.read().encode('''utf-8''');__pyfile.close();os.remove('''/var/folders/41/6sd124ws3t982vl6bmw1gvjxdmh6b5/T/py7wwbdu''');exec(compile(__code, '''/var/folders/41/6sd124ws3t982vl6bmw1gvjxdmh6b5/T/py7wwbdu''', 'exec'));
actual_func
 The actual func.
```

In other words, the name and the docstring of the decorated function is overwritten by the decorator, which is not expected.

We can fix such issue with `@wraps`, for example, the original code could be replaced with

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
    ''' The greet func. '''
    print(f"Hello, {name}!")

print(greet.__name__)
print(greet.__doc__)
```

```text
greet
 The greet func.
```


#### `@lru_cache` : decorator to wrap a function with a LRU cache {#lru-cache-decorator-to-wrap-a-function-with-a-lru-cache}

<!--list-separator-->

-  Accelerating DP-like recursive function call

    ```python
    @functools.lru_cache(maxsize=1000)
    def factorial(n):
        return n * factorial(n-1) if n else 1
    ```

<!--list-separator-->

-  Initialization for some heavy states without introducing global variables

    Suppose we have some global states that should be initialized only once, the naive way to do it is by introducing some global variables,

    ```python
    state = None

    def get_state(args):
        if state is None:
            state = construct_state(args)
        return state
    ```

    We can eliminate the need for a global variable with a cache:

    ```python
    @lru_cache
    def get_state(args):
        return construct_state(args)
    ```
