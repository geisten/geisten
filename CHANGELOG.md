# Changelog

All notable changes to geist are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
once it reaches 1.0. While in 0.x, `EXPERIMENTAL`-tagged API may change in any
minor release.

## [Unreleased]

## [0.1.0]

First public release.

### Added
- C23 inference runtime with a stable C ABI (`include/geist.h`), per-symbol
  `STABLE` / `EXPERIMENTAL` stability tags.
- Backends: `cpu_neon` (Apple Silicon + ARM64, OpenMP-parallel kernels) and
  `cpu_scalar` (portable reference). `cpu_x86` is a policy skeleton.
- Quantization: GGUF `Q4_0/Q8_0`, k-quants `Q3_K/Q4_K/Q5_K/Q6_K`, IQ-quants
  `IQ2_S/IQ3_S`, and ternary `TQ2_0` for 1.58-bit models. Zero-dispatch kernel
  binding: every tensor is bound to a specialized kernel at load time.
- Transformer architecture (Gemma 4 family) with RoPE, GQA attention, KV cache,
  and per-session sampler (greedy / top-k / top-p / temperature).
- KV-cache quantization modes (INT8, KIVI), AWQ scale loading, and an n-gram
  speculative-decode path (all `EXPERIMENTAL`).
- Native multimodal: Conformer audio tower (`attach_audio`) and SigLIP vision
  tower for image/video soft-token prefixes (`attach_image` / `attach_video`).
- Build system with per-target/per-mode segregation (`mac`, `mac-omp`, `pi5`,
  `linux`/generic ARM64), `debug`/`asan`/`perf` modes, and on-demand reference
  model fetch (`make fetch-model`).
- Test suite (exit-code contract, `_unit`/`_int`/`_e2e` tiers) and a
  reproducible perf benchmark harness (`make bench-small`).
- `examples/simple_generate` demonstrating the stable text-generation core.

[Unreleased]: https://github.com/geisten/geist/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/geisten/geist/releases/tag/v0.1.0
