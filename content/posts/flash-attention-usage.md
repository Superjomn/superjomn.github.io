+++
title = "flash-attention Usage: a Worknote for LLM inference"
author = ["Chunwei Yan"]
date = 2025-03-30
tags = ["llm", "tech"]
draft = false
+++

## Background {#background}

The [flash-attention](https://github.com/Dao-AILab/flash-attention/tree/main) project provides `flash_attn` package in Python, and it provides multiple APIs in the interface.
As the APIs contains many LLM optimization concepts such as paged kv-cache, variant-length (continuous batching) and so on.
This post tries to aggregate related information for the related concepts, and focus on inference only {{< sidenote id="inferece-only" content="We will not cover the modules defined for training, and only focus on several basic functional APIs used in inference" >}}, for using the `flash_attn` APIs.


### The APIs {#the-apis}

We will focus on the following two APIs which are also tested in the [test_flash_attn.py](https://github.com/Dao-AILab/flash-attention/tree/1a58058a6da83bd7baaf4c512e8a1abe0240bb77/tests/test_flash_attn.py).

-   flash_attn_varlen_func
-   flash_attn_with_kvcache

These two APIs can work with Paged KV Cache, which are crucial for the inference of LLM, and they are used in some LLM projects such as SGLang or vLLM.


### The related concepts {#the-related-concepts}


#### Prefilling and Decoding {#prefilling-and-decoding}

The flash-attention provides two different sets of APIs for prefilling and decoding.

Here is some comparasion between the two:

| -             | Prefilling     | Decoding |
|---------------|----------------|----------|
| Input Tokens  | seqlen &gt;= 1 | 1        |
| Output Tokens | 1              | 1        |

The difference in the IO tokens result in different arguments in the APIs.


#### KV Cache {#kv-cache}

The self-attention is computed as below:

```python
# hidden_states for the first transformer layer
# token_embeddings: [batch_size, seq_len, hidden_size]
token_embeddings = token_embeddings[input_token_ids]
# position_embeddings: [batch_size, seq_len, hidden_size]
position_embeddings = position_embeddings[position_ids]

hidden_states = token_embeddings + position_embeddings

def attention(hidden_states):
    query = hidden_states @ Wq
    key = hidden_states @ Wk
    value = hidden_states @ Wv

    # MHA (Multi-Head Attention)
    attn_output = MHA(query, key, value)
    return attn_output # it will be the hidden_states for the next layer
```

Suppose the current sequence length is `seq_len`, and the batch size is `batch_size`. As the hidden_states is of shape `[batch_size, seq_len, hidden_size]`, the query, key and value are of shape `[batch_size, seq_len, hidden_size]`,
for the next token, the hidden_states is of shape `[batch_size, seq_len + 1, hidden_size]`, and both the key and value are of shape `[batch_size, seq_len + 1, hidden_size]`.
Since the `Wk` and `Wv` are fixed, the key and value can be pre-computed and stored in the memory.

Here is the pseudo code for the above process:

```python
# current sequence length is seq_len, we hope to predict the next token of (seq_len + 1)
# kv cache: [batch_size, seq_len, hidden_size]
def kvcached_attention(hidden_states, k_cache, v_cache):
    # query: [batch_size, seq_len, hidden_size] for prefilling phase
    # query: [batch_size, 1, hidden_size] for decoding phase
    query = hidden_states @ Wq
    # key, value: [batch_size, seq_len, hidden_size]
    key = k_cache # key is cached, which eliminates the computation of $hidden_states @ Wk$
    value = v_cache # so is value

    attn_output = MHA(query, key, value)
    return attn_output # it will be the hidden_states for the next layer
```


#### Paged KV Cache {#paged-kv-cache}

KV Cache is vital, while it is not efficient for batch inference. Suppose we have a batch of sequences with different lengths,
and for K and V, we need to store the caches in Tensors, the shape of which is `[batch_size, max_seq_len, hidden_size]` where `max_seq_len` is the maximum sequence length in the batch.
What's more, normally, we will append the new key and value to the end of the cache, thus the shape could be `[batch_size, max_prefill_len + max_generate_len, hidden_size]`, which may waste a lot of memory.

To solve the above problems, we can use the paged KV cache. The idea is to split the cache into several pages, and each sequence will only cache the number of pages that are used.

For example, if we have 3 sequences with length 10, 20, 30, and we set the page size to 10, then the total number of pages is 1 + 2 + 3 = 6. Normally the paged cache is stored with two tensors, one holds the addresses of the pages for the batch, and the other holds the number of pages for each sequence.
Paged KV Cache modified the inputs to the attention function, thus it needs dedicated kernels.


## APIs {#apis}


### flash_attn_varlen_func for prefilling {#flash-attn-varlen-func-for-prefilling}

This API is mainly used for prefilling, as the prefilling could have multiple sequences with different lengths.


#### flash_attn_varlen_func without KV Cache {#flash-attn-varlen-func-without-kv-cache}

It is simpler to use without KV Cache:

```python
def test_flash_attn_varlen_func_without_kvcache(
    device="cuda", seed=42, batch_size=10, num_heads=16, head_dim=16
):
    """Test variable length FlashAttention implementation.

    Args:
        device: Device to run the test on
        seed: Random seed for reproducibility
        batch_size: Number of sequences in batch
        num_heads: Number of attention heads
        head_dim: Dimension of each attention head


    The flash_attn_varlen_func is for prefilling phase.
    """
    # Set seed for reproducibility
    torch.manual_seed(seed)

    # Calculate total hidden dimension
    hidden_dim = num_heads * head_dim

    # Generate random sequence lengths between 10 and 100
    seq_len = torch.randint(10, 100, (batch_size, 1), device=device)
    max_seq_len = torch.max(seq_len).item()
    total_seq_len = torch.sum(seq_len).item()

    # All of the q,k,v packs all the sequence into one tensor
    # Create query, key, value tensors (total_seq_len, num_heads, head_dim)
    q = torch.randn(total_seq_len, num_heads, head_dim, device=device, dtype=torch.float16)
    k = torch.randn(total_seq_len, num_heads, head_dim, device=device, dtype=torch.float16)
    v = torch.randn(total_seq_len, num_heads, head_dim, device=device, dtype=torch.float16)

    # Remove the extra dimension from seq_len
    seq_len = seq_len.squeeze(1)

    # Create cumulative sequence lengths with leading 0
    # This creates offsets: [0, len1, len1+len2, len1+len2+len3, ...]
    cu_seqlens_q = torch.cumsum(seq_len, dim=0, dtype=torch.int32)
    cu_seqlens_q = torch.cat([torch.zeros(1, dtype=torch.int32, device=device), cu_seqlens_q])
    cu_seqlens_k = cu_seqlens_q.clone()  # Keys have same lengths as queries

    # Run flash attention with variable length sequences
    res = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seq_len,
        max_seqlen_k=max_seq_len,
        dropout_p=0.0,
        return_attn_probs=True,
    )

    output = res[0]
    attn_probs = res[1]
    S_mask = res[2]

    # Basic validation - check output shape matches input shape
    assert (
        output.shape == q.shape
    ), f"Output shape {output.shape} doesn't match input shape {q.shape}"

    # Verify output is not all zeros or NaNs
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert torch.any(output != 0), "Output is all zeros"

    print("output", output)
    print("attn_probs", attn_probs)
    print("S_mask", S_mask)

    return output
```


#### flash_attn_varlen_func with KV Cache {#flash-attn-varlen-func-with-kv-cache}

In an LLM framework, the Paged KV Cache is crucial for memory efficiency, this API can work with Paged KV Cache.

Let's define the Paged KV Cache utility function:

```python
def generate_block_kvcache(
    max_seqlen_k: int,
    paged_kv_block_size: int,
    max_batch_size: int,
    nheads_k: int,
    d: int,
    device: torch.device,
    dtype: torch.dtype,
):
    """Generate a block-based KV cache for efficient memory management in attention.

    This function creates a paged key-value cache organized in memory blocks, along with
    a block table that maps logical sequence positions to physical memory blocks.
    This block-based approach allows efficient memory management for variable-length
    sequences in transformer decoding.

    Args:
        max_seqlen_k: Maximum sequence length for keys
        paged_kv_block_size: Size of each block in the paged cache
        max_batch_size: Maximum batch size
        nheads_k: Number of attention heads
        d: Dimension of each attention head
        device: Device to create tensors on
        dtype: Data type for the cache tensors

    Returns:
        Tuple containing:
            - k_cache_paged: Paged key cache tensor [num_blocks, block_size, nheads_k, d]
            - v_cache_paged: Paged value cache tensor [num_blocks, block_size, nheads_k, d]
            - block_table: Mapping from logical to physical blocks [batch_size, num_blocks_per_seq]
    """
    # Calculate total number of blocks needed
    num_blocks = math.ceil(max_seqlen_k / paged_kv_block_size) * max_batch_size

    # Create randomized paged cache storage for keys and values
    k_cache_paged = torch.randn(
        num_blocks, paged_kv_block_size, nheads_k, d, device=device, dtype=dtype
    )
    v_cache_paged = torch.randn(
        num_blocks, paged_kv_block_size, nheads_k, d, device=device, dtype=dtype
    )

    # Create block table - a mapping from logical sequence positions to physical memory blocks
    # Using a random permutation to simulate realistic block allocation
    block_table = rearrange(
        torch.randperm(num_blocks, dtype=torch.int32, device=device),
        "(b nblocks) -> b nblocks",
        b=max_batch_size,
    )

    return k_cache_paged, v_cache_paged, block_table

def create_culens(seq_lens: torch.Tensor, device: torch.device):
    cu_seqlens = torch.cumsum(seq_lens, dim=0, dtype=torch.int32)
    cu_seqlens = torch.cat([torch.zeros(1, dtype=torch.int32, device=device), cu_seqlens])
    return cu_seqlens
```

And here is the code for using the API:

```python
def test_flash_attn_varlen_func_with_kvcache(
    device="cuda", seed=42, batch_size=10, num_heads=16, head_dim=16
):
    """Test flash attention with variable length and KV caching.

    Tests the functionality of flash_attn_varlen_func when using paged key-value cache.

    Args:
        device: Device to run the test on (default: "cuda")
        seed: Random seed for reproducibility (default: 42)
        batch_size: Number of sequences in batch (default: 10)
        num_heads: Number of attention heads (default: 16)
        head_dim: Dimension of each attention head (default: 16)
    """
    # Set seed for reproducibility
    torch.manual_seed(seed)

    # Generate random sequence lengths between 10 and 100
    seq_lens = torch.randint(10, 100, (batch_size, 1), device=device)
    max_seq_len = torch.max(seq_lens).item()
    total_seq_len = torch.sum(seq_lens).item()

    # KV cache parameters
    paged_kv_block_size = 256
    max_k_seq_len = 100
    k_seq_lens = torch.randint(0, max_k_seq_len, (batch_size, 1), device=device)

    # Create query tensor packed with all sequences (total_seq_len, num_heads, head_dim)
    q = torch.randn(total_seq_len, num_heads, head_dim, device=device, dtype=torch.float16)

    # Generate paged KV cache with extra room for new tokens
    k_cache_paged, v_cache_paged, block_table = generate_block_kvcache(
        max_k_seq_len + 100,  # room for new tokens
        paged_kv_block_size,
        batch_size,
        num_heads,
        head_dim,
        device,
        dtype=torch.float16,
    )

    # Prepare sequence length information
    seq_lens = seq_lens.squeeze(1)
    k_seq_lens = k_seq_lens.squeeze(1)

    # Create cumulative sequence lengths for batched attention
    cu_seqlens_q = create_culens(seq_lens, device)
    cu_seqlens_k = create_culens(k_seq_lens, device)

    # Run flash attention with variable length sequences
    output, attn_probs, S_mask = flash_attn_varlen_func(
        q,
        k=k_cache_paged,
        v=v_cache_paged,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seq_len,
        max_seqlen_k=max_k_seq_len,
        block_table=block_table,
        dropout_p=0.0,
        return_attn_probs=True,
    )

    # Verify outputs
    assert output.shape == q.shape, f"Output shape {output.shape} doesn't match query shape {q.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert torch.any(output != 0), "Output is all zeros"
    assert attn_probs is not None, "Attention probabilities not returned"
    assert S_mask is not None, "Attention mask not returned"

    # Return results for potential further testing
    return output, attn_probs, S_mask
```


### flash_attn_with_kvcache for decoding {#flash-attn-with-kvcache-for-decoding}

This API is mainly used for decoding, as the decoding is always a batch of sequences with one token.

```python
def test_flash_attn_with_kvcache(device="cuda", seed=42, batch_size=10, num_heads=16, head_dim=16):
    """Test flash attention with KV cache for incremental decoding.

    This test validates the functionality of flash_attn_with_kvcache which is designed
    for efficient incremental decoding. The function updates the KV cache in-place with
    new key and value tensors while performing attention in a single kernel call.

    Args:
        device: Device to run the test on (default: "cuda")
        seed: Random seed for reproducibility (default: 42)
        batch_size: Number of sequences in batch (default: 10)
        num_heads: Number of attention heads (default: 16)
        head_dim: Dimension of each attention head (default: 16)

    Returns:
        Attention output tensor
    """
    # Set seed for reproducibility
    torch.manual_seed(seed)

    # Create query tensor - for incremental decoding, we only have one token per sequence
    q = torch.randn(batch_size, 1, num_heads, head_dim, device=device, dtype=torch.float16)

    # Generate random sequence lengths for the key-value cache (previous tokens)
    max_seq_len_k = 100
    seq_lens_k = torch.randint(10, max_seq_len_k, (batch_size,), device=device, dtype=torch.int32)
    max_seq_len_k = torch.max(seq_lens_k).item()

    # Create paged KV cache - block-based memory structure for efficient caching
    paged_kv_block_size = 256
    k_cache_paged, v_cache_paged, block_table = generate_block_kvcache(
        max_seq_len_k,
        paged_kv_block_size,
        batch_size,
        num_heads,
        head_dim,
        device,
        dtype=torch.float16,
    )

    # Create new key and value tensors for the current token
    # (These will be added to the cache in-place during the attention call)
    k = torch.randn(batch_size, 1, num_heads, head_dim, device=device, dtype=torch.float16)
    v = torch.randn(batch_size, 1, num_heads, head_dim, device=device, dtype=torch.float16)

    # Run flash attention with KV cache
    # This performs attention and updates the cache in a single operation
    output = flash_attn_with_kvcache(
        q=q,
        k_cache=k_cache_paged,
        v_cache=v_cache_paged,
        k=k,
        v=v,
        cache_seqlens=seq_lens_k,
        block_table=block_table,
    )

    # Validate output
    expected_shape = (batch_size, 1, num_heads, head_dim)
    assert output.shape == expected_shape, f"Output shape {output.shape} doesn't match expected {expected_shape}"
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert torch.any(output != 0), "Output is all zeros"

    # Verify cache was updated by checking if sequences grew by 1
    # (This assumes flash_attn_with_kvcache increments cache_seqlens internally)

    return output
```


## The references {#the-references}

-   The [file](https://github.com/Superjomn/yallm/blob/main/tests/3rdparty/test_flashattn.py) containing all the code
