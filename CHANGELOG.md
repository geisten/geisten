# Changelog

All notable changes to geist are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
once it reaches 1.0. While in 0.x, `EXPERIMENTAL`-tagged API may change in any
minor release.

## [Unreleased]

## [0.2.0]

### Changed — public API split by audience (source-compatibility break)

`geist.h` is now the **minimal surface to run a model** (backend → model →
session → `set_prompt` → `decode_step` → `token_to_str`). Helpers and advanced
features moved to new headers. **No signatures changed**; declarations moved.

- `geist_util.h` *(new)* — chat / advanced apps: special tokens
  (`geist_model_eos_token` / `_bos_token` / `_token_by_text`),
  `geist_session_tokenize` / `_prefill_tokens`, multimodal `attach_audio/image/video`,
  `pin_prefix`, `peek_logits`, `decode_speculative`, telemetry, and
  `geist_backend_supports_op`.
- `geist_types.h` *(new)* — backend-author territory: low-level tensor / op /
  dtype / buffer / quant types (previously in `geist.h`). Pulled in by
  `geist_backend.h` and `geist_weight.h`.

**Migration:** pure text generation needs no change (`#include <geist.h>` still
compiles; `examples/simple_generate.c` and `tools/geist.c` are untouched). Apps
using special tokens / multimodal / speculative / telemetry add
`#include <geist_util.h>`. Backend/tensor-type code adds `#include <geist_types.h>`.

### Added
- `geist_model_eos_token`, `geist_model_bos_token`, `geist_model_token_by_text`
  (`geist_util.h`): special-token ids from GGUF metadata, so a chat app stops on
  `tok == eos` instead of string-matching decoded output.

## [0.1.3]

### Changed
- Linux release artifact is a fully static **musl** binary (was glibc): ~40 %
  smaller download, portable across any aarch64 Linux with no libc dependency.
  Both release binaries are stripped.

## [0.1.2]

### Changed
- Parallelized the O(n²) int8 SDPA prefill core (bit-exact) — flat Pi 5 prefill
  curve (pp1024 +35 %). Benchmark re-measured honestly (Pi thermal-throttling
  artifact corrected; matched cool-start protocol; llama.cpp leads Pi prefill).

## [0.1.1]

### Added
- `geist_gemm` abstraction + BLAS-free native NEON fp32 + vendored FFT; fully
  dependency-free static ARM build; CI release matrix; int8-kernel tuning; CLI.

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

[Unreleased]: https://github.com/geisten/geist/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/geisten/geist/compare/v0.1.3...v0.2.0
[0.1.3]: https://github.com/geisten/geist/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/geisten/geist/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/geisten/geist/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/geisten/geist/releases/tag/v0.1.0
