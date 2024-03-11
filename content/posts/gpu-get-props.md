+++
title = "Get GPU Properties"
author = ["Chunwei Yan"]
date = 2024-03-11
tags = ["gpu", "basics", "tech"]
draft = false
+++

In \`cuda_runtime.h\`, there are several APIs for retrieving properties for the installed GPU.

-   [cudaDeviceGetAttribute(int\* value, cudaDeviceAttr attr, int  device)](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1gb22e8256592b836df9a9cc36c9db7151): a C api
-   [cudaGetDeviceProperties ( cudaDeviceProp\* prop, int  device ) ](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g1bf9d625a931d657e08db2b4391170f0): a C++ api

[Here](https://github.com/Superjomn/cuda-from-scratch/blob/dev/dump-gpu-props.cpp) is the code of the example.

On a Nvidia GTX 3080 GPU, the properties are as below:

```text
Device 0 properties:
  Max block dimensions: 1024 x 1024 x 64
  Max grid dimensions: 2147483647 x 65535 x 65535
  Shared memory bank size: 4 bytes
  Max shared memory per block: 49152 bytes
  Max registers per block: 65536
  Warp size: 32
  Multiprocessor count: 68
  Max resident threads per multiprocessor: 1536 = 48 warps
  L2 cache size: 5242880 bytes
  Global L1 cache supported: yes
  Total global memory: 9 GB
  Processor clock: 1 MHZ
  Memory clock: 9 MHZ
```
