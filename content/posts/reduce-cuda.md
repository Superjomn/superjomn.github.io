+++
title = "Reduce kernel in CUDA"
author = ["Chunwei Yan"]
date = 2024-03-25
tags = ["cuda", "basics", "tech"]
draft = false
+++

## Question definition {#question-definition}

Given an array of \\(n\\) integers, get the sum of all the elements.


## Solutions {#solutions}


### Naive version with atomicAdd {#naive-version-with-atomicadd}

The most naive way is to make all the threads trigger atomicAdd on the output.

```C++
__global__ void reduce_naive_atomic(int* g_idata, int* g_odata, unsigned int n)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int gridSize = blockDim.x * gridDim.x;

    int sum = 0;
    for (unsigned int i = idx; i < n; i += gridSize)
    {
        sum += g_idata[i];
    }

    atomicAdd(g_odata, sum);
}
```

And the kernel launcher is simple, it launches the kernel only once:

```C++
int launch_reduce(int* g_idata, int* g_odata, unsigned int n, int block_size, kernel_fn kernel, cudaStream_t stream)
{
    int* idata = g_idata;
    int* odata = g_odata;

    uint32_t num_warps = block_size / 32;
    int smem_size = num_warps * sizeof(int);

    int num_blocks = ceil(n, block_size);

    // Launch the kernel
    kernel<<<num_blocks, block_size, smem_size, stream>>>(idata, odata, n);
    if (!FLAGS_profile)
        cudaStreamSynchronize(stream);

    // Copy the final result back to the host
    int h_out;
    NVCHECK(cudaMemcpyAsync(&h_out, odata, sizeof(int), cudaMemcpyDeviceToHost, stream));

    return h_out;
}
```

On GTX 4080, the throughput could get roughly 82GB/s.


### Tiled reduction with shared memory {#tiled-reduction-with-shared-memory}

One classical way is to utilize the thread block to perform reduction on a tile locally on shared memory.

There are several kernel versions to do this.


#### Basic version {#basic-version}

The basic implementation is as below:

1.  load a tile of data into the shared memory collectively
2.  perform partial reduction on the data tile inside a thread block and get the sum of the tile
3.  write the sum of the tile on the corresponding place on the output slot in global memory, note that, this kernel requires a temporary buffer to write a partial result
4.  shrink \\(n\\) to \\(\frac{n}{blockSize}\\), and repeat the steps above until \\(n=1\\)

<!--listend-->

```C++
__global__ void reduce_smem_naive(int* g_idata, int* g_odata, unsigned int n)
{
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Read a block of data into shared memory collectively
    sdata[tid] = (i < n) ? g_idata[i] : 0;
    __syncthreads();

    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2)
    {
        // ISSUE: divergent warps
        if (tid % (2 * stride) == 0)
        {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads(); // need to sync per level
    }

    // Write the result for this block to global memory
    if (tid == 0)
    {
        g_odata[blockIdx.x] = sdata[0];
    }
}
```

This kernel needs multiple times of launching, the launcher is a bit more complex:

```C++
int launch_reduce(int* g_idata, int* g_odata, unsigned int n, int block_size, kernel_fn kernel, cudaStream_t stream,
    uint32_t num_blocks, uint32_t smem_size)
{
    int* idata = g_idata;
    int* odata = g_odata;
    if (smem_size == 0)
        smem_size = block_size * sizeof(int);

    // Calculate the number of blocks
    num_blocks = (num_blocks > 0) ? num_blocks : (n + block_size - 1) / block_size;

    if (!FLAGS_profile)
        printf("- launching: num_blocks: %d, block_size:%d, n:%d\n", num_blocks, block_size, n);

    int level = 0;
    // Launch the kernel
    kernel<<<num_blocks, block_size, smem_size, stream>>>(idata, odata, n);
    if (!FLAGS_profile)
        cudaStreamSynchronize(stream);

    level++;

    // Recursively reduce the partial sums
    while (num_blocks > 1)
    {
        std::swap(idata, odata);
        n = num_blocks;
        num_blocks = (n + block_size - 1) / block_size;
        kernel<<<num_blocks, block_size, smem_size, stream>>>(idata, odata, n);
        if (!FLAGS_profile)
            cudaStreamSynchronize(stream);
    }

    // Copy the final result back to the host
    int h_out;
    NVCHECK(cudaMemcpyAsync(&h_out, odata, sizeof(int), cudaMemcpyDeviceToHost, stream));

    return h_out;
}
```

All the tiled reduction kernels share the above launcher.

It can get a throughput of 54GB/s, worse than the atomic naive one(82GB/s).


#### Avoid thread divergence {#avoid-thread-divergence}

The basic version has a serious problem of thread divergence on `if (tid % (2 * stride) == 0)`, here is an optimized version:

```C++
__global__ void reduce_smem_1_avoid_divergent_warps(int* g_idata, int* g_odata, unsigned int n)
{
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? g_idata[i] : 0;
    __syncthreads();

    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2)
    {
        int index = 2 * stride * tid;
        if (index < blockDim.x)
        {
            // Issue: bank conflict
            sdata[index] += sdata[index + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        g_odata[blockIdx.x] = sdata[0];
    }
}
```

It can reach 70GB/s, which is 29% improved than the basic one.


#### Read two elements one time {#read-two-elements-one-time}

The last version has a low DRAM throughput: `DRAM Throughput [%]	20.63`, that may due to the following reasons:

1.  The grid size is small due to a small input, so the resource is not fully utilized with the threads
2.  Each thread reads only one element, considering there are a fixed number of resident thread blocks in SMs for a specific kernel, which means a small number of LD instructions are launched each time.

To improve the DRAM Throughput, for small grid size, we can make the thread read more than one element each time.

```C++
__global__ void reduce_smem_3_read_two(int* g_idata, int* g_odata, unsigned int n)
{
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

#define GET_ELEM(__idx) ((__idx) < n ? g_idata[(__idx)] : 0)

    sdata[tid] = GET_ELEM(i) + GET_ELEM(i + blockDim.x);

    __syncthreads();

    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2)
    {
        int index = 2 * stride * tid;
        if (index < blockDim.x)
        {
            // Issue: bank conflict
            sdata[index] += sdata[index + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        g_odata[blockIdx.x] = sdata[0];
    }
}
```

This kernel achieves `DRAM Throughput [%]	33.78`, which is 63.72% larger than the previous one.

The overall throughput is 96.51GB/s, which is 37.87% better than the previous one(70GB/s).


### Tiled reduction with warp_shlf {#tiled-reduction-with-warp-shlf}

The modern GPU supports threads within a warp to exchange data directly instead of shared memory, which should be much faster by eliminating the shared memory read/write.

The following function helps to do a reduction on a single warp.

```C++
// using warp shuffle instruction
// From book <Professional CUDA C Programming>
__inline__ __device__ int warpReduce(int mySum)
{
    mySum += __shfl_xor(mySum, 16);
    mySum += __shfl_xor(mySum, 8);
    mySum += __shfl_xor(mySum, 4);
    mySum += __shfl_xor(mySum, 2);
    mySum += __shfl_xor(mySum, 1);
    return mySum;
}
```

If a thread block contains multiple warps, it requires synchronization, the shared memory is a good choice. By writing the sum within each warp into the shared memory, and then doing a reduction on the shared memory as above, we can get the sum of a thread block, and the rest logic is identical to the kernels above.

```C++
__global__ void reduce_warp_shlf(int* g_idata, int* g_odata, unsigned int n)
{
    // Helps to share data between warps
    // size should be (blockDim.x / warpSize)
    extern __shared__ int sdata[];

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Necessary to make sure shfl instruction is not used with uninitialized data
    int mySum = idx < n ? g_idata[idx] : 0;

    int lane = threadIdx.x % warpSize;
    int warp = threadIdx.x / warpSize;

    mySum = warpReduce(mySum);

    if (lane == 0)
    {
        sdata[warp] = mySum;
    }

    __syncthreads();

    // last warp reduce
    mySum = (threadIdx.x < blockDim.x / warpSize) ? sdata[lane] : 0;
    if (warp == 0)
    {
        mySum = warpReduce(mySum);
    }

    if (threadIdx.x == 0)
    {
        g_odata[blockIdx.x] = mySum;
    }
}
```

This kernel reads a single element of thread, but it can achieve a throughput of 96GB/s (The shared memory version is 70GB/s). Of course, it can be refactored to read \\(N\\) element each time:

```C++
template <int NT>
__global__ void reduce_warp_shlf_read_N(int* g_idata, int* g_odata, unsigned int n)
{
    // Helps to share data between warps
    // size should be (blockDim.x / warpSize)
    extern __shared__ int sdata[];

    int blockSize = NT * blockDim.x;
    unsigned int idx = blockIdx.x * blockSize + threadIdx.x;

// Necessary to make sure shfl instruction is not used with uninitialized data
#define GET_ELEM(__idx) ((__idx) < n ? g_idata[(__idx)] : 0)
    int mySum = 0;

#pragma unroll
    for (int i = 0; i < NT; i++)
        mySum += GET_ELEM(idx + i * blockDim.x);

    int lane = threadIdx.x % warpSize;
    int warp = threadIdx.x / warpSize;

    mySum = warpReduce(mySum);

    if (lane == 0)
    {
        sdata[warp] = mySum;
    }

    __syncthreads();

    // last warp reduce
    mySum = (threadIdx.x < blockDim.x / warpSize) ? sdata[lane] : 0;
    if (warp == 0)
    {
        mySum = warpReduce(mySum);
    }

    if (threadIdx.x == 0)
    {
        g_odata[blockIdx.x] = mySum;
    }
}
```

With different \\(NT\\), it gets different performance:

| NT | throughput (GB/s) |
|----|-------------------|
| 1  | 96.3187           |
| 2  | 96.2341           |
| 4  | 96.8153           |
| 8  | 107.226           |


### warp shuffle with atomic {#warp-shuffle-with-atomic}

Compared to the tiled solution, the atomicAdd doesn't need a temporary buffer and the kernel needs to launch only once. Let's take atomic together with warp shuffle.

```C++
template <int NT>
__global__ void reduce_warp_shlf_read_N_atomic(int* g_idata, int* g_odata, unsigned int n)
{
    // Helps to share data between warps
    // size should be (blockDim.x / warpSize)
    extern __shared__ int sdata[];

    int blockSize = NT * blockDim.x;
    unsigned int idx = blockIdx.x * blockSize + threadIdx.x;

// Necessary to make sure shfl instruction is not used with uninitialized data
// This only needs one turn of launch
#define GET_ELEM(__idx) ((__idx) < n ? g_idata[(__idx)] : 0)
    int mySum = 0;

#pragma unroll
    for (int i = 0; i < NT; i++)
        mySum += GET_ELEM(idx + i * blockDim.x);

    int lane = threadIdx.x % warpSize;
    int warp = threadIdx.x / warpSize;

    mySum = warpReduce(mySum);

    if (lane == 0)
    {
        sdata[warp] = mySum;
    }

    __syncthreads();

    // last warp reduce
    mySum = (threadIdx.x < blockDim.x / warpSize) ? sdata[lane] : 0;
    if (warp == 0)
    {
        mySum = warpReduce(mySum);
    }

    if (threadIdx.x == 0)
    {
        atomicAdd(g_odata, mySum);
    }
}
```

It can achieve a throughput of 121.777 GB/s, which is the best on the same setting.


## Benchmark {#benchmark}

{{< figure src="/ox-hugo/2024-04-06_16-47-39_screenshot.png" >}}

Note that, in different \\(n\\), the optimum kernel might be different.


## Reference {#reference}

-   [Optimizing Parallel Reduction in CUDA](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)
-   [Faster Parallel Reductions on Kepler | NVIDIA Technical Blog](https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/)
