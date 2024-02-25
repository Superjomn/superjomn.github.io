+++
title = "LLVM Utilities (keep updating)"
author = ["Chunwei Yan"]
date = 2023-10-17
tags = ["llvm", "cpp", "tech"]
draft = false
+++

There are many handy functions or data structures in LLVM project, which are widely used by other projects that rely on LLVM. In this page, I will introduce some common utilities that are worthy of using in your own project or frequently used in LLVM code that you should be familiar with.


## Basic data type {#basic-data-type}


### llvm::StringRef {#llvm-stringref}

It is a lightweight, non-owning reference to a sequence of characters.
It is similar to `std::string_view` introduced in `C++17`.

An example:

```C++
// from a C-string
const char* cStr = "hello";
llvm::StringRef strRef(cStr);

// from a C++-string
std::string cppStr = "hello";
llvm::StringRef strRef1(cppStr);

// from pointer and length
llvm::StringRef strRef2(cppStr.c_str(), cppStr.size());
```


### llvm::ArrayRef {#llvm-arrayref}

It is a lightweight, non-owning reference to an array of elements. It is similar to `std::span` introduced in C++20.

An example:

```C++
int myArray[] = {1, 2, 3, 4, 5};
llvm::ArrayRef<int> arrayRef(myArray, 5);
```


### llvm::Twine {#llvm-twine}

`llvm::Twine` is a class used to efficiently concatenate strings in both memory and performance.

To concate two strings:

```C++
llvm::Twine twine1("Hello, ");
llvm::Twine twine2("world!");
llvm::Twine result = twine1 + twine2;
```

To concate string with other elements:

```C++
llvm::Twine twine1("The answer is ");
int value = 42;
llvm::Twine result = twine1 + llvm::Twine(value);
```

It is possible to concate multiple elements:

```C++
llvm::Twine result = llvm::Twine("Hello, ") + "world!" + 42 + 3.14;
```

In the example above, the first "Hello" is a `Twine` instance, and all the following "+" will use `Twine`'s `operator+` and get new `Twine` instances, so it is able to concate any number of elements in the real usages.


### llvm::NullablePtr {#llvm-nullableptr}

It is used to represent a pointer that can be either a valid pointer or null.

```C++
llvm::NullablePtr<MyType> ptr;

ptr = new MyType();
// or ptr = nullptr;

if (ptr.isNull()) {
  // ...
} else {
  delete ptr.get(); // get the underlying pointer
}
```


## Container {#container}


### llvm::DenseMap {#llvm-densemap}

`llvm::DenseMap` has higher performance than `std::unordered_map` and a similar usage.

An example:

```C++
llvm::DenseMap<int, float> map;
map[20] = 20.f;
map.insert(std::make_pair(20, 20.f));
```


### llvm::DenseMapInfo {#llvm-densemapinfo}

`llvm::DenseMapInfo` is a utility class that provides information and hashing for custom types used as keys in `llvm::DenseMap`. To use it, you should define your custom type with the following methods provided:

-   `static KeyTy getEmptyKey()`: This function should return a unique value representing an "empty" or "deleted" key in your custom type
-   `static KeyTy getTombstoneKey()`: It should return a unique value representing a "tombstone" key, which is used when a key is removed.
-   `static unique getHashValue(const KeyTy& key)`: This function returns the hash value of a given key.
-   `static bool isEqual(const KeyTy& a, const KeyTy &b)`: This function compares two keys and returns true if they are equal, or false if they are not.

An example:

```C++
struct MyKeyType {
    int value;

    static MyKeyType getEmptyKey() { return MyKeyType{-1}; }
    static MyKeyType getTombstoneKey() { return MyKeyType{-2}; }
    static unsigned getHashValue(const MyKeyType &key) { return llvm::hash_value(key.value); }
    static bool isEqual(const MyKeyType &a, const MyKeyType &b) { return a.value == b.value; }
};
```

After this, you should specialize the `llvm::DenseMapInfo` template for your custom type:

```C++
namespace llvm {
template <>
struct DenseMapInfo<MyKeyType> {
    static MyKeyType getEmptyKey() { return MyKeyType::getEmptyKey(); }
    static MyKeyType getTombstoneKey() { return MyKeyType::getTombstoneKey(); }
    static unsigned getHashValue(const MyKeyType &key) { return MyKeyType::getHashValue(key); }
    static bool isEqual(const MyKeyType &a, const MyKeyType &b) { return MyKeyType::isEqual(a, b); }
};
}
```


### llvm::StringMap {#llvm-stringmap}

It is a map-like container that is specially optimized for string keys.

```C++
llvm::StringMap<int> stringToIntMap;

stringToIntMap["name"] = "Tim";
stringToIntMap.insert(std::map_pair("name", "Tom"));
```


### llvm::SmallVector {#llvm-smallvector}

It is a dynamic array container that quite similar to `std::vector` but optimized for situations where the number of elements is expected to be small.

An example:

```C++
llvm::SmallVector<int, 4> vec;
vec.push_back(1);
```


## Misc {#misc}


### llvm::BumpPtrAllocator {#llvm-bumpptrallocator}

This is an allocator used to allocate memory in a highly efficient manner. But note that, it doesn't support deallocation for the elements allocated.
Once the `llvm::BumpPtrAllocator` instance is freed, all the allocated elements will be deallocated in bulk automatically.

An example:

```C++
llvm::BumpPtrAllocator allocator;

int* intPtr = allocator.Allocate<int>();

*intPtr = 100;
```


## Updating log {#updating-log}

-   <span class="timestamp-wrapper"><span class="timestamp">[2024-02-25 Sun] </span></span> Publish the post
