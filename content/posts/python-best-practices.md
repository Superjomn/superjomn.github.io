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


## Handy builtin utilities {#handy-builtin-utilities}


### setter and getter {#setter-and-getter}

When there is some logic bound to a member when it is got or updated, then the getter and setter could be used.

```python
class App:
    def __init__(self):
        self.update_count = 0
        self._name = ""

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, v:str):
        self._name = v
        self.update_count += 1

app = App()
app.name = 'a'
app.name = 'b'

print('name:', app.name) # b
print('update_count:', app.update_count) # 2
```

```text
name: b
update_count: 2
```


### `@dataclass` {#dataclass}

`@dataclass` is a decorator that can be used to create classes that **mainly store data**.
It can automatically generate some common methods for the class, such as `__init__`, `__repr__`, and `__eq__`, based on the type hints of the class attributes.

```python
from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float

p = Point(1., 2.)
print(p)
```

```text
Point(x=1.0, y=2.0)
```

There are several classical practices using `@dataclass`


#### Use default values or default factories {#use-default-values-or-default-factories}

```python
from dataclasses import dataclass, field
from random import randint
from typing import List

@dataclass
class DummyContainer:
    sides: int = 6
    value: int = field(default_factory=lambda: randint(1, 6))
    alist: List[int] = field(default_factory=list) # avoid assign [] directly

dummy = DummyContainer()
print(dummy)
```

```text
DummyContainer(sides=6, value=2, alist=[])
```


#### Use `frozen=True` to make the class immutable {#use-frozen-true-to-make-the-class-immutable}

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Circle:
    radius: float

const_circle = Circle(2.0)
```


#### Use `order=True` to enable comparison operators based on the class attributes {#use-order-true-to-enable-comparison-operators-based-on-the-class-attributes}

```python
from dataclasses import dataclass

@dataclass(order=True)
class Circle:
    radius: float

c0 = Circle(1.)
c1 = Circle(2)

print(c0 > c1)
```

```text
False
```


#### Use inheritance to create subclasses of data classes {#use-inheritance-to-create-subclasses-of-data-classes}

```python
from dataclasses import dataclass

@dataclass
class Animal:
    name: str
    sound: str

@dataclass
class Dog(Animal):
    # inherits name and sound from Animal
    watch_house: bool

dog = Dog(name="Huang", sound="Wang", watch_house=False)
print(dog)
```

```text
Dog(name='Huang', sound='Wang', watch_house=False)
```


### functools `partial` to get new function by partially fixing some arguments of an existing one {#functools-partial-to-get-new-function-by-partially-fixing-some-arguments-of-an-existing-one}

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
functools.partial(<function func0 at 0x7f8a480e31e0>, a=0)
a:0, b:10
a:1, b:10
```


### functools `@warps` to help define better decorators {#functools-warps-to-help-define-better-decorators}

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


### functools `@lru_cache` : decorator to wrap a function with a LRU cache {#functools-lru-cache-decorator-to-wrap-a-function-with-a-lru-cache}


#### Accelerating DP-like recursive function call {#accelerating-dp-like-recursive-function-call}

```python
@functools.lru_cache(maxsize=1000)
def factorial(n):
    return n * factorial(n-1) if n else 1
```


#### Initialization for some heavy states without introducing global variables {#initialization-for-some-heavy-states-without-introducing-global-variables}

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
