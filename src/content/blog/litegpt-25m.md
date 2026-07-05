---
title: 'Building a 25M Parameter GPT from Scratch'
description: 'Architecture choices, tokenizer mistakes, training issues, and lessons from training LiteGPT on FineWeb and TinyStories.'
pubDate: 2026-01-05
heroImage: '../../assets/blog-placeholder-4.jpg'
---

"How hard can this be?"

To satisfy my curiosity and understand every design choice firsthand, I built a 25M parameter GPT from scratch. This article is a collection of the lessons I learned along the way.

## TLDR

I trained a 25M parameter language model from scratch on 500M tokens from FineWeb and TinyStories, using modern architecture choices such as RoPE, GQA, SwiGLU, and RMSNorm. The final run used an NVIDIA RTX A5000 GPU for roughly two hours.

Model weights and logs are on [Hugging Face](https://huggingface.co/Aadit-032/LiteGPT-25M). The training code is on [GitHub](https://github.com/Aaidt/Lite-GPT).

## Why bother building this?

Building a model from scratch taught me how much the pre-training data matters, gave me a more concrete intuition for scaling laws, and helped demystify a system that previously felt like a black box.

I was inspired by Karpathy's NanoGPT. It made me think about how I could improve the architecture, try different dataset mixtures, and push for better results with fewer parameters and less compute. I wanted the repository to be easy for beginners to read while still being flexible enough for future experiments.

## Architecture

The earlier LiteGPT-16M followed a GPT-2 style architecture with LayerNorm, multi-head attention, GELU feed-forward layers, and learned positional embeddings.

For the 25M model, I modernized the architecture with RMSNorm, SwiGLU, RoPE, and grouped-query attention. These choices make training faster and scale better.

### Dataset

I started with a mixture of 60% FineWeb, 30% TinyStories, and 10% The Stack Smol. I wanted to fine-tune the base model later into a small coding agent, but the code data introduced a very different token distribution. Since the model was too small to handle that well, I removed the code dataset and focused on natural language.

The final dataset mixture was 60% FineWeb and 40% TinyStories. I expected the loss to drop much more after removing The Stack Smol, but it did not change by as much as expected. A useful next experiment would be training exclusively on FineWeb or TinyStories.

### Tokenizer

I initially used the GPT-2 tokenizer, the same one used in NanoGPT. I could not get validation loss below 4, which suggested something was fundamentally wrong.

The issue was vocabulary size. GPT-2's vocabulary has 50,257 tokens. With `d_model = 320`, the embedding table alone used roughly 16M parameters out of the 25M total trainable parameters. That left too little capacity for the transformer itself.

To fix this, I trained a byte-level BPE tokenizer with a vocabulary size of about 16K using Hugging Face Tokenizers. The dataset is stored as `uint16` binary files and loaded lazily with `np.memmap()`, which lets the data loader grab random chunks without loading the whole dataset into RAM.

### Scaling laws

Chinchilla scaling laws suggest that for a fixed compute budget, model size and number of training tokens need to be balanced. The rough compute-optimal regime is about 20 training tokens per parameter.

For a 25M parameter model, that points toward roughly 500M training tokens. This is why I targeted a 500M-token dataset.

### Learning rate schedule

The learning rate linearly warms up to a peak of `6e-4` and then follows cosine decay. Warmup matters because early losses are large, and using a high learning rate immediately can make updates unstable. Cosine decay helps reduce update size as the model starts converging.

### Transformer block

The model uses a GPT-2 style pre-norm block, but with RMSNorm instead of LayerNorm. RMSNorm normalizes by root-mean-square magnitude and avoids subtracting the mean, which makes it cheaper than LayerNorm.

RoPE rotates the query and key vectors before attention, allowing the attention mechanism to model relative positions. I chose RoPE because it reduces learned positional parameters and tends to generalize better to longer contexts than learned positional embeddings.

Grouped-query attention shares key and value heads across multiple query heads. This reduces memory and compute requirements for the KV cache during inference. Attention is computed with PyTorch's `torch.nn.functional.scaled_dot_product_attention()`, which can dispatch to optimized fused kernels such as FlashAttention when the hardware and inputs support it.

The feed-forward layer uses SwiGLU instead of the GPT-2 GELU MLP. SwiGLU uses a gate projection to decide which features should pass through, making the block more expressive for a similar compute budget.

## Implementation

The hidden dimension follows the LLaMA-style rule of roughly `8 / 3 * d_model` instead of the classic GPT-2 `4 * d_model`, because SwiGLU uses three linear projections rather than two.

Training ran for 40,000 forward and backward iterations with a gradient accumulation factor of 2, resulting in 20,000 optimizer steps.

The total number of tokens seen during training was:

```text
64 x 512 x 40,000 = 1,310,720,000 tokens
```

That is about 1.3B tokens, or roughly 2.6 epochs over the 500M-token dataset.

## Issues I faced

### Hardware constraints

I started on the free-tier T4 GPU on Google Colab, but it was too slow and sessions were terminated aggressively. The 16GB VRAM limit also forced a smaller batch size.

I switched to RunPod and rented an NVIDIA RTX A5000, which let me increase batch size, hidden dimension, and other hyperparameters.

The main ways I considered reducing VRAM pressure were gradient checkpointing and gradient accumulation. Gradient checkpointing recomputes activations during backpropagation instead of storing all of them. Gradient accumulation increases effective batch size by accumulating gradients over multiple passes before one optimizer update.

### Loss not going down

The model's loss was not improving, which meant something in the setup was broken. The main issue was the tokenizer and embedding table size: too many parameters were being spent on embeddings, leaving too little capacity for the transformer.

### Steps are not the right training metric

I initially thought training for more steps would automatically improve the model. That was wrong. The number of quality tokens seen during training is a much better thing to focus on.

For LiteGPT-16M, the training loss was around 2 while validation loss was around 5. That was a clear overfitting signal: the model memorized the training data but failed to generalize.

### Wrong data mixture

The initial mix included coding data, but the model was too small to produce useful code. The code dataset also pulled the token distribution away from the natural language behavior I cared about most.

## Improvements to make

- Train exclusively on FineWeb to test a larger and more diverse natural language distribution.
- Scale to more than 1B tokens while following Chinchilla-style scaling laws.
- Fine-tune on instruction data for a specific downstream task.
- Try alternative optimizers such as AdamW-mini or Sophia.

## Example generations

### Prompt: The capital of France is

The model produced fluent but factually unreliable text. This was expected at 25M parameters because the model does not have enough capacity to store many facts.

### Prompt: Once upon a time, there lived

The model produced a coherent children's-story style continuation, which makes sense given that 40% of the training data came from TinyStories.

### Prompt: The dragon opened its eyes and

The model continued with a simple narrative structure and mostly consistent story elements.

### Prompt: Tom had a little red ball.

The model generated a longer toy-sharing story with repetition and some grammar issues, but it still preserved the general structure of a children's story.

## Observations

The model performs best on story-like prompts, which is not surprising given the TinyStories mixture. Factual accuracy is poor at 25M parameters. Repetition loops become common at higher temperatures. Coding generations are essentially non-functional. Reasoning is limited, but the model does show a basic grasp of narrative structure.

## References

- Attention Is All You Need, Vaswani et al.
- NanoGPT, Karpathy
- Language Models are Unsupervised Multitask Learners, Radford et al.
- LLaMA: Open and Efficient Foundation Language Models, Touvron et al.
- RoFormer: Enhanced Transformer with Rotary Position Embedding, Su et al.
- GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints, Ainslie et al.
- Scaling Laws for Neural Language Models, Kaplan et al.
- Training Compute-Optimal Large Language Models, Hoffmann et al.
- FlashAttention, Dao et al.
- RMSNorm, Zhang and Sennrich
