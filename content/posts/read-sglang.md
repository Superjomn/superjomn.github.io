+++
title = "Reading SGLang Code"
author = ["Hieu Phay"]
date = 2025-07-23
tags = ["cuda", "basics", "tech"]
draft = true
+++

## Structure {#structure}


### `python/sglang/src` {#python-sglang-src}

models/ - Model implementations and configurations
model_executor/ - Model execution engine
model_loader/ - Model loading utilities
entrypoints/ - API entry points (OpenAI-compatible, etc.)
sampling/ - Text sampling and generation logic
speculative/ - Speculative decoding implementations
multimodal/ - Multimodal model support
lora/ - LoRA (Low-Rank Adaptation) support
layers/ - Neural network layer implementations
function_call/ - Function calling capabilities
constrained/ - Constrained generation
distributed/ - Distributed computing support
disaggregation/ - Model disaggregation features
mem_cache/ - Memory caching systems
metrics/ - Performance metrics and monitoring
managers/ - Resource management
connector/ - External system connectors
configs/ - Configuration management
eplb/ - Load balancing components


## `configs` {#configs}

This directory serves as the central configuration hub for model support

**Core configuration files**:

-   `model_config.py` - The main `ModelConfig` class that handles universal model configuration
-   `device_config.py` - Hardware device configuration (CUDA, CPU and so on)
-   `load_config.py` - Model loading strategies and formats

**Model-Specific Configurations**:

-   `chatglm.py` - ChatGLM model family
-   ...

**Core Architecture**:

{{< figure src="/static/model-config.png" >}}


## Model Loader in `model_loader` {#model-loader-in-model-loader}

Modern LLMs can have huge number of parameters, stored in various formats (SafeTensors, PyTorch, GGUF), potentially quantized (4-bit, 8-bit). The model loader is introduced to help

-   Support different file formats and storage locations
-   Handle various quantization schemes
-   Support distributed loading across multiple devices

-   `BaseModelLoader` - The base class for all the model loaders.
    -   `DefaultModelLoader` - Model loader that can load different file types from disk.
        -   `LayeredModelLoader` - Model loader that loads weights layer by layer so that one can quantize a layer before loading another to make the peak memory envelope smaller.
    -   `DummyModelLoader` - Model loader that will set model weights to random values.
    -   `ShardedStateLoader` - Model loader that directly loads each worker's model state dict, which enables a fast load path for large tensor-parallel models where each worker only needs to read its own shard rather than the entire checkpoint.
    -   `BitsAndBytesModelLoader` - Model loader to load model weights with BitAndBytes quantization.
    -   `GGUFModelLoader` - Model loader that can load GGUF files.
    -   `RemoteModelLoader` - Model loader that can load Tensors from remote database.

{{< figure src="/static/model_loader.png" >}}


## lora {#lora}


### The files {#the-files}

-   `lora_config.py` - Configuration management for LoRA adapters, handling HuggingFace adapter configs and parameter validation
-   `lora.py` - Core LoRA adapter and layer classes


### Efficient design {#efficient-design}


#### Two-Tier Memory Architecture {#two-tier-memory-architecture}

There are two levels of memory pools:

1.  The Disk to the host memory managed by `LoRAAdapter`
2.  The host memory to the device memory managed by `LoRAMemoryPool`


#### Buffer reusing {#buffer-reusing}

SGLang will try to reuse the loras in the two levels of memory pools by grouping the requests of the same lora and point them to the same buffer.


### Architecture {#architecture}

{{< figure src="/static/lora.png" >}}
