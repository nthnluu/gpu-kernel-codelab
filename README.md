# Write and Optimize GPU/TPU Kernels with Pallas

Writing a custom GPU kernel sounds intimidating until you actually do it. This codelab takes you from zero kernel experience to implementing Flash Attention yourself, one tile at a time, using [JAX Pallas](https://docs.jax.dev/en/latest/pallas/index.html).

## What you'll build

| Kernel | What it teaches |
|--------|----------------|
| **Fused Softmax** | Kernel fusion, memory-bound optimization, the roofline model |
| **Fused LayerNorm** | Multi-input kernels, affine transforms in SRAM |
| **Tiled Matrix Multiply** | 2D tiling, K-dimension reduction, MXU exploitation |
| **Flash Attention** | Online softmax, cross-iteration state, TPU pipelining |

## What you'll learn

- The Pallas programming model: grids, tiles, BlockSpecs
- Why kernel fusion reduces memory bandwidth pressure
- Hardware-aware optimization: roofline analysis, MXU utilization, DMA pipelining
- How to benchmark kernels properly on accelerators
- TPU vs GPU execution models and tuning strategies

## Prerequisites

- Python and basic linear algebra
- No CUDA or kernel programming experience needed

## Getting started

**Option 1: Google Colab (recommended)**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nthnluu/gpu-kernel-codelab/blob/main/learn_kernels.ipynb)

1. Click the badge above (or upload `learn_kernels.ipynb` to Colab manually)
2. Set runtime to **TPU v5e** (Runtime > Change runtime type > TPU)
3. Work through Parts 0-9 in order
4. Switch to **GPU** runtime and re-run to see how the same kernels behave differently

**Option 2: Local (CPU interpret mode)**

```bash
pip install jax jaxlib
jupyter notebook learn_kernels.ipynb
```

Kernels run in interpret mode on CPU. Useful for debugging logic, but you won't get real performance numbers.

## Structure

| Part | Topic | Type |
|------|-------|------|
| 0 | Environment setup | Setup |
| 1 | Mental model: tiles, grids, BlockSpecs | Concept |
| 2 | Hello world: vector addition | Walkthrough |
| 3 | **Exercise: ReLU kernel** | Exercise |
| 4 | **Exercise: fused softmax** | Exercise |
| 5 | Benchmarking on accelerators | Walkthrough |
| 6 | Hardware-aware optimization (roofline, MXU, pipelining) | Deep dive |
| 7 | **Exercise: tiled matrix multiplication** | Exercise |
| 8 | Online softmax, Flash Attention, `custom_vjp` | Deep dive |
| 8 | **Exercise: fused LayerNorm + residual** | Exercise |
| 9 | **Exercises: scaled softmax, autotuning, fused matmul+GeLU** | Exercises |
| 10 | Resources | Reference |

## Supplementary materials

I put together a [NotebookLM notebook](https://notebooklm.google.com/notebook/c37e48ea-9fba-486c-95ec-dee82f35d3c8) loaded with the codelab, all the referenced papers, and the Pallas docs. You can use it to ask questions as you work through the exercises.

I also generated some video overviews that walk through the key concepts:

| # | Topic | Covers |
|---|-------|--------|
| 1 | [Why Kernels Matter](https://notebooklm.google.com/notebook/c37e48ea-9fba-486c-95ec-dee82f35d3c8?artifactId=620a0aab-b6ae-4bf9-9956-ece8c4263ec0) | Parts 0-3: the mental model, tiling, BlockSpecs |
| 2 | [Kernel Fusion and Benchmarking](https://notebooklm.google.com/notebook/c37e48ea-9fba-486c-95ec-dee82f35d3c8?artifactId=1ef49655-d413-4d86-847e-8e55a2e4d6f9) | Parts 4-5: fused softmax, memory bandwidth, benchmarking |
| 3 | [Hardware-Aware Optimization](https://notebooklm.google.com/notebook/c37e48ea-9fba-486c-95ec-dee82f35d3c8?artifactId=3b1a4dca-b5a1-4798-bcc5-139dd04ba89d) | Part 6: roofline model, MXU, pipelining, LayerNorm |
| 4 | [Matrix Multiplication](https://notebooklm.google.com/notebook/c37e48ea-9fba-486c-95ec-dee82f35d3c8?artifactId=57977328-b6f0-42c2-a3a3-e609c4cede72) | Part 7: tiled matmul, K-tiling, the accumulator pattern |
| 5 | [Flash Attention from Scratch](https://notebooklm.google.com/notebook/c37e48ea-9fba-486c-95ec-dee82f35d3c8?artifactId=0ded5b60-b1ed-4f77-ac0b-23b1b56a9118) | Part 8: online softmax, Flash Attention, custom_vjp |

## Resources

- [JAX Pallas documentation](https://docs.jax.dev/en/latest/pallas/index.html)
- [Pallas TPU examples](https://docs.jax.dev/en/latest/pallas/tpu/examples.html)
- [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html)
- [FlashAttention paper](https://arxiv.org/abs/2205.05538)
- [GPU MODE lectures](https://www.youtube.com/@GPUMODE)
- [What Every Programmer Should Know About Memory](https://people.freebsd.org/~lstewart/articles/cpumemory.pdf)

## License

MIT
