+++
title = "Count the parameters in LLaMA V1 model"
author = ["Chunwei Yan"]
date = 2024-03-21
tags = ["LLM", "tech"]
draft = false
+++

Let's load the model

```python
from transformers import LlamaModel, LlamaConfig
model = LlamaModel.from_pretrained("llama-7b-hf-path")

def count_params(model, is_human: bool = False):
    params: int = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"{params / 1e6:.2f}M" if is_human else params

print(model)
print("Total # of params:", count_params(model, is_human=True))
```

Print out the layers:

```text
LlamaModel(
  (embed_tokens): Embedding(32000, 4096, padding_idx=0)
  (layers): ModuleList(
    (0-31): 32 x LlamaDecoderLayer(
      (self_attn): LlamaSdpaAttention(
        (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
        (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
        (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
        (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
        (rotary_emb): LlamaRotaryEmbedding()
      )
      (mlp): LlamaMLP(
        (gate_proj): Linear(in_features=4096, out_features=11008, bias=False)
        (up_proj): Linear(in_features=4096, out_features=11008, bias=False)
        (down_proj): Linear(in_features=11008, out_features=4096, bias=False)
        (act_fn): SiLU()
      )
      (input_layernorm): LlamaRMSNorm()
      (post_attention_layernorm): LlamaRMSNorm()
    )
  )
  (norm): LlamaRMSNorm()
)
Total # of params: 6607.34M
```

The Transformers shows that there are 6607.34M float16 parameters, roughly 13GB, that is aligned to the actual weight size.


## The basic setting of the 7B model {#the-basic-setting-of-the-7b-model}

-   model dimension \\(d\_{model}=4096\\)
-   number of heads \\(n\_{head}=32\\)
-   head size \\(d\_{head} = \frac{d\_{model}}{n\_{head}}\\)
-   dimension of the feed-forward network's inner layer \\(d\_{ff}=11008\\)
-   number of tokens \\(n\_{token}=32000\\)
-   number of transformer layers \\(n\_{layer}=32\\)


## Layer-by-Layer Parameter Count {#layer-by-layer-parameter-count}


### Embedding layer {#embedding-layer}

For vocabulary embedding, \\(n\_{token}\times d\_{model}=131.072M\\), while for position embedding, since RoPE doesn't need a separate embedding, so that is 0.


### Transformer layers {#transformer-layers}


#### `input_layernorm` and `post_attention_layernorm` {#input-layernorm-and-post-attention-layernorm}

Both are RMSNorm whose parameters are \\(d\_{model}\\), so both sum to \\(2\times d\_{model}=8M\\)


#### multi-head self-attention {#multi-head-self-attention}

For Q,K,V and O, each is a Linear layer of size \\(d\_{model} \times d\_{model}\\), so in total, there are \\(4\times d\_{model}^2=67.1M\\).

There is one tiny issue here, why a linear layer could generate Q, while in the original transformer paper, each head is calculated separately, for example, \\(Q\_i=QW^Q\_i\\) where \\(i\\) is the head id. That is because, if we concatenate all all the heads, that is identical to a linear of \\(d\_{model} \times (n\_{head} \times d\_{head})\\), that is \\(d\_{model} \times d\_{model}\\) in llama v1.

The self-attention doesn't have extra parameters since they simply applies the following formula

\\[
Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d\_k}})V
\\]


#### mlp {#mlp}

The LlamaMLP layer contains three separate Linear layers:

1.  `gate_proj`: \\(d\_{model} \times d\_{ff}\\)
2.  `up_proj`: \\(d\_{model} \times d\_{ff}\\)
3.  `down_proj`: \\(d\_{ff} \times d\_{model}\\)

So in total, they have \\(3\times d\_{model} \times d\_{ff} = 135.27M\\) parameters.


## Total count of parameters {#total-count-of-parameters}

The overall parameters are composed of two major parts, the vocabulary embedding, and the transformer layers, that is `embed + 32 * (mha + mlp + norm)`:

-   \\(embed=n\_{token}\times d\_{model}=131.07M\\)
-   \\(mha=4\* d\_{model}^2=67.1M\\)
-   \\(mlp=3\* d\_{model}\times d\_{ff}=135.27M\\)
-   \\(norm=2\*d\_{model}=8.19M\\)

And the count of the parameters is 6607.3M, which is aligned to the number from Transformers.

```python
def count_llama_params(d_model, d_ff, n_tokens, n_layers):
    embed = n_tokens * d_model
    mha = 4 * d_model**2
    mlp = 3 * d_moel * d_ff
    norm = 2 * d_model
    return embed + n_layers * (mha + mlp + norm)
```

For example, the Llama 65B model

```text
LlamaModel(
  (embed_tokens): Embedding(32000, 8192, padding_idx=0)
  (layers): ModuleList(
    (0-79): 80 x LlamaDecoderLayer(
      (self_attn): LlamaSdpaAttention(
        (q_proj): Linear(in_features=8192, out_features=8192, bias=False)
        (k_proj): Linear(in_features=8192, out_features=8192, bias=False)
        (v_proj): Linear(in_features=8192, out_features=8192, bias=False)
        (o_proj): Linear(in_features=8192, out_features=8192, bias=False)
        (rotary_emb): LlamaRotaryEmbedding()
      )
      (mlp): LlamaMLP(
        (gate_proj): Linear(in_features=8192, out_features=22016, bias=False)
        (up_proj): Linear(in_features=8192, out_features=22016, bias=False)
        (down_proj): Linear(in_features=22016, out_features=8192, bias=False)
        (act_fn): SiLU()
      )
      (input_layernorm): LlamaRMSNorm()
      (post_attention_layernorm): LlamaRMSNorm()
    )
  )
  (norm): LlamaRMSNorm()
)
Total # of params: 65023.52M
```

And let's use the function

```python
count_llama_params(d_model=8192,
    d_ff=22016,
    n_tokens=32000,
    n_layers=80)
```

It gives 65023.5M, is is roughly aligned.


## References {#references}

-   [Transformer Math (Part 1) - Counting Model Parameters](https://michaelwornow.net/2024/01/18/counting-params-in-transformer)
-   [modeling_llama.py from huggingface transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py)
