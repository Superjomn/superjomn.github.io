+++
title = "Notes on LLM technologies (keep updating)"
author = ["Chunwei Yan"]
date = 2024-03-10
tags = ["LLM", "tech"]
draft = false
+++

Brief notes on LLM technologies.


## Models {#models}


### GPT2 {#gpt2}


#### Model structure {#model-structure}

{{< figure src="/ox-hugo/GPT model structure.png" >}}

The GPT model employs a repeated structure of Transformer Blocks, each containing two sub-layers: a Masked Multi-Head Attention (MMHA) layer and a Position-wise Feed-Forward Network.

The MMHA is a central component of the model. It operates by splitting the input into multiple 'heads', each of which learns to attend to different positions within the input sequence, allowing the model to focus on different aspects of the input simultaneously. The output of these heads is then concatenated and linearly transformed to produce the final output.

The MMHA mechanism can be formally defined as follows:

\\[
MultiHead(Q,K,V) = Concat(head\_1, \cdots, head\_h)W^O
\\]

where each head is computed as:

\\[
head\_i = Attention(QW\_i^Q, KW\_i^K, VW\_i^V)
\\]

In implementation, the computation of \\(Q,K,V\\) can be packed together with Linear operations regardless of the number of heads, like

\begin{split}
Q &= xW^Q \\\\
K &= xW^K \\\\
V &= xW^V
\end{split}

The Attention function is defined as:

\\[
Attention(q\_i, k\_i, v\_i) = softmax(\frac{q\_i k\_i^T}{\sqrt{d\_k}})v\_i
\\]

Here, \\(d\_k\\) represents the dimension of the keys, which is calculated as \\(d\_k = \frac{H}{h}\\), where \\(H\\) is the total dimension of the input and \\(h\\) is the number of heads.

To ensure the MHA mechanism works correctly with sequences of varying lengths, a Mask is applied. This Mask effectively ignores padding elements by setting their values to \\(-\infty\\), allowing the Softmax function to handle them appropriately.
The layer output demensions are as below:

| Layer               | Dimensions           | Note                                  |
|---------------------|----------------------|---------------------------------------|
| Model input         | `[bs, seq_len]`      | Token IDs                             |
| Text &amp; PosEmbed | `[bs, seq_len, H]`   | Text embeddings + position embeddings |
| Layer Norm (0)      | `[bs, seq_len, H]`   |                                       |
| Feed Forward        | `[bs, seq_len, H]`   |                                       |
| Layer Norm (1)      | `[bs, seq_len, H]`   |                                       |
| \\(head\_i\\)       | `[bs, seq_len, H/h]` |                                       |
| MMHA                | `[bs, seq_len, H]`   |                                       |

Where

-   `bs` is the batch size
-   `seq_len` is the max length of the sequence
-   `H` is the size of the hidden state
-   `h` is the number of heads


### Reference {#reference}

-   [Leviathan, Yaniv, Matan Kalman, and Yossi Matias. "Fast inference from transformers via speculative decoding." International Conference on Machine Learning. PMLR, 2023.](https://arxiv.org/abs/2211.17192)
-   [Multi-head attention in Pytorch](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)


## Lora {#lora}


### Algorithm {#algorithm}

![](/ox-hugo/2024-03-17_15-41-23_screenshot.png)
_(image borrowed from [this page](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms))_

A classical workflow to finetune an LLM is to learn an additional parameters denoted as \\(\delta W\\) as long as frooze the original parameters, just as the left part of the figure below.

\\[
h = W\_0 x \Rightarrow (W\_0 + \delta W) x
\\]

This could be applied on the \\(W\_q, W\_k, W\_v\\) and \\(W\_o\\) in the Transformer block, while since the Transformer blocks contains the mojority of the parameters, that workflow could result in significant increase of additional parameters.

For instance, a Llama V1 7B model, the hidden size is 4096, 32 heads and 32 layers.
The \\(W\_q, W\_k, W\_v\\), each shape is \\(4096 \times /frac{4096}{32} = 4096 \times 128=512k\\), and the \\(W\_o\\) is \\(4096\times4096=16384k\\) so on total, the additional parameters will take

$32 &times; (3 &times; 512k+16384k) \* 2 = $

The LoRA is for such scenarios, instead of learning the \\(\delta W\\) itself, it learns decomposed representation of \\(\delta W\\) directly during finetune training. Since the rank could be \\(8\\), that could reduce the number of trainable parameters required for adaptation to downstream tasks.


### Reference {#reference}

-   [Practical Tips for Finetuning LLMs Using LoRA (Low-Rank Adaptation)](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms)
-   Hu, Edward J., et al. "Lora: Low-rank adaptation of large language models." arXiv preprint arXiv:2106.09685 (2021).


## Speculative decoding {#speculative-decoding}


### Motivation {#motivation}

Consider a scenerio where we have a prefix such as "Geoffrey Hinton did his PhD at the University", and the target suffix is "of Edinburgh". When a LLM continues the generation, it is evident that:

1.  The word "of" is simple to generate and could be produced by a smaller model given the same prefix
2.  The word "Edinburgh" is more challenging to generate and may require a larger model with more knowledge

Speculative decoding addresses this by using a smaller model to generate "easy" words like "of" for better throughput, while leaving more challenging words to a larger model for precision.


### Algorithm {#algorithm}

Speculative decoding employs two models:

1.  A draft model, denoted as \\(M\_p\\), which is smaller and much faster (as least 2X) to give a sub-sequence of the next K tokens.
2.  A target model, denoted as \\(M\_q\\), which is larger and more precise. It evaluates the sub-sequence generated by the draft model.

Assuming K to be 4, the prefix to be \\(pf\\), and the draft model generates five tokens based on \\(pf\\):

1.  Token \\(x\_1\\), the probability is \\(p\_1(x) = M\_p(pf)\\)
2.  Token \\(x\_2\\) with probability of \\(p\_2(x) = M\_p(pf, x\_1)\\)
3.  Token \\(x\_3\\) with probability of \\(p\_3(x) = M\_p(pf, x\_1, x\_2)\\)
4.  Token \\(x\_4\\) with probability of \\(p\_4(x) = M\_p(pf, x\_1, x\_2, x\_3)\\)

The target model evalutes K tokens generated by \\(M\_p\\) with a single model forward pass, similar to the training phase:

\\[
q\_1(x), q\_2(x), q\_3(x), q\_4(x) = M\_q(pf, x\_1, x\_2, x\_3, x\_4)
\\]

Let's consider a real example to illustrate the heuristics. Suppose the draft model generate the following sub-sequence with \\(K=4\\):

| Token          | x1   | x2   | x3      | x4      |
|----------------|------|------|---------|---------|
|                | dogs | love | chasing | after   |
| p(x)           | 0.8  | 0.7  | 0.9     | 0.8     |
| q(x)           | 0.9  | 0.8  | 0.8     | 0.3     |
| q(x)&gt;=p(x)? | Y    | Y    | N       | N       |
| accept prob    | 1    | 1    | 0.8/0.9 | 0.3/0.8 |

The rules is as below:

1.  If \\(q(x) >= p(x)\\), then accept the token.
2.  If not, the accept probability is \\(\frac{q(x)}{p(x)}\\), so the token "chasing" has a probability of \\(\frac{0.8}{0.9}=89\\%\\), while the next token "after" has an accept probability of only \\(\frac{0.3}{0.8}=37.5\\%\\).
3.  If a word is unaccepted, the candidate word after it will be dropped as well. It will be resampled by target model, not the draft model.
4.  Repeat the steps above from the next position


### Reference {#reference}

[Speculative Decoding: When Two LLMs are Faster than One - YouTube](https://www.youtube.com/watch?v=S-8yr_RibJ4)
