# geist 👻

[![CI](https://github.com/geisten/geist/actions/workflows/ci.yml/badge.svg)](https://github.com/geisten/geist/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![C Standard](https://img.shields.io/badge/C-C23-orange.svg)](https://en.wikipedia.org/wiki/C23_(C_standard_revision))
[![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20Linux%20(ARM64)-lightgrey.svg)](#-build--usage)
[![Status](https://img.shields.io/badge/status-experimental%20(v0.1.0)-yellow.svg)](#-status)

**geist** is a high-performance, ultra-lean C23 inference runtime specifically engineered for **Low-Bit models (IQ/TQ)**, **Ternary (1.58-bit) BitNet**, and **Native Multimodal Audio** on edge CPUs.

It is designed for the constraint where llama.cpp is the "universal default" but you need more speed, less RAM, or native multimodal integration on $50-$100 hardware like the **Raspberry Pi 5**.

> **Status: experimental (v0.1.0).** The public API in [`include/geist.h`](include/geist.h)
> carries per-symbol stability tags (`STABLE` / `EXPERIMENTAL`). Expect churn in
> `EXPERIMENTAL` surfaces until 1.0. Issues and PRs welcome — see
> [CONTRIBUTING.md](CONTRIBUTING.md).

---

## ✨ Demo

The whole stable text-generation core is ~70 lines of C
([`examples/simple_generate.c`](examples/simple_generate.c)). Build the library,
build the example, point it at a GGUF:

```console
$ make && make -C examples
$ OMP_WAIT_POLICY=active examples/simple_generate gemma4-e2b-Q4_K_M.gguf \
      "The capital of France is"
loaded gemma4-e2b-Q4_K_M.gguf (arch: transformer)
The capital of France is Paris.

$ OMP_WAIT_POLICY=active examples/simple_generate gemma4-e2b-Q4_K_M.gguf \
      "Write a haiku about the ocean:" 40
Write a haiku about the ocean:

Blue waves crash on sand,
Salt spray kisses the warm air,
Ocean's deep secrets.
```

*Real output from `examples/simple_generate` on Gemma 4 E2B-it (Q4_K_M), greedy
decode. Reproduce with `make fetch-model` then the commands above.*

---

## 🚀 Performance Highlights (Gemma 4 E2B IT)

On Apple Silicon (M1 Max), CPU-only, same `Q4_K_M` GGUF, `geist` leads on
prompt processing and is at near-parity on decode — by pinning to the
performance cores (the efficiency cores stall a static OpenMP schedule) and
binding every tensor to a specialized kernel at load time.

| Engine (M1 Max, CPU, Gemma 4 E2B Q4_K_M) | Prefill pp512 | Decode tg128 |
| :--- | :---: | :---: |
| llama.cpp `-ngl 0` (b9430, BLAS) | 152 t/s | 39 t/s |
| **geist** | **156 t/s** (1.02×) | 32 t/s (0.82×) |

*Measured June 2026 on a quiesced Apple M1 Max (8 P-cores), `llama.cpp` build
`d48a56eff` (9430), both CPU-only on the identical GGUF, each at its best thread
count. geist leads on prompt processing by auto-pinning to the performance cores
(the efficiency cores stall a static OpenMP schedule — this alone moved pp512
from 91 → 156 t/s). Decode is ~0.82× and bounded by the maturity of the Q4_K
decode GEMV (94% of decode time) vs llama.cpp's long-tuned kernel; closing it is
tracked work. See [BENCHMARK.md](BENCHMARK.md) to reproduce on your hardware.*

### Raspberry Pi 5 (Cortex-A76) — the edge target, iso-model & iso-quality

The Pi 5 is the design target, and the hard case (an older ARM core without `i8mm`,
where llama.cpp leans on a decades-tuned OpenBLAS fp32 path). Running the **identical**
Gemma 4 E2B Q4_K_M model with **bit-identical** output, **this first version already
matches llama.cpp** — ahead on short-prompt prefill, at parity on decode, and within
~5 % on long-prompt prefill:

| Engine (Pi 5, 4 threads, Q4_K_M) | Prefill pp128 | Prefill pp256 | Decode |
| :--- | :---: | :---: | :---: |
| llama.cpp (OpenBLAS) | 26.8 t/s | **31.6 t/s** | ~7.0 t/s |
| **geist** | **31.0 t/s** (1.16×) | 30.0 t/s (0.95×) | ~7.0 t/s (parity) |

*Same weights, same quantization, bit-identical results — not a smaller or lossy model.
The remaining long-prompt gap is raw NEON throughput vs OpenBLAS; on `i8mm`/ARMv9 cores
and Apple AMX the int8 path pulls clearly ahead. Benchmarked June 2026; see
[BENCHMARK_PI5.md](BENCHMARK_PI5.md).*

---

## 🛠 Why geist?

### 1. Zero-Dispatch Architecture
Unlike generic engines that use complex layer-dispatch loops, `geist` uses **Kernel Binding**. At load time, every tensor is bound directly to a specialized kernel pointer. This eliminates vtable overhead and management logic during the hot path—critical for single-core-heavy edge CPUs.

### 2. Ternary (1.58-bit) as a First-Class Citizen
We don't treat low-bit formats as an afterthought. Our backend is built for a **multiplication-free future**. `geist` includes native paths for BitNet b1.58, where the CPU only performs additions and subtractions, maximizing performance on hardware without powerful NPUs.

### 3. Native Multimodal Audio
`geist` features a built-in Conformer-based audio tower. Instead of a slow "Whisper → Text → LLM" cascade, we support direct audio-embedding prefixes. The LLM "hears" the audio directly, reducing latency and preserving prosody.


---

## 📦 Build & Usage

### Requirements
- C compiler with C23 support (Apple-clang ≥ 15, gcc ≥ 13).
- `make`.
- **Mac:** Homebrew `libomp` recommended for multi-threading.

### Quick Start
```bash
# Build (target auto-detected: mac-omp / mac / pi5 / linux).
make                       # or: make TARGET=mac-omp | pi5 | linux

# Grab a reference model (Gemma 4 E2B-it Q4_K_M, ~3.1 GB) — optional helper.
make fetch-model

# Run the evaluation REPL against a GGUF. detect-target.sh prints the
# build dir (mac-omp, pi5, ...); the backticks expand it in your shell.
OMP_WAIT_POLICY=active bin/`mk/detect-target.sh`/release/eval_geist gguf_artifacts/gemma4-e2b-Q4_K_M.gguf
```

A minimal C program using the public API lives in
[`examples/`](examples/) — build it with `make -C examples`.

---

## 🗺 Roadmap

- [x] **Close the Pi 5 Gap:** Implement FFN-streaming and lm-head argmax for ARM dominance.
- [ ] **BitNet Optimization:** Reach 1.0x reference parity for 2B-4T ternary models on Pi 5.
- [ ] **Dynamic Quantization:** Release the first mixed-low-bit recipe for Gemma 4.
- [ ] **Realtime Audio Demo:** A standalone VAD-to-Instruction voice assistant on Pi 5.

---

## 🧭 Status

`geist` is **v0.1.0 — experimental**. It runs Gemma 4 (text + vision + audio) end
to end on the CPU backends and has a broad C test suite (`make test`), but the
`EXPERIMENTAL`-tagged parts of the API (KV-cache modes, speculative decode, AWQ,
multimodal attach) may still change between minor versions. The `STABLE` core
(load → session → decode → tokenize) is the part to build on.

## 📜 License & Contribution

`geist` is licensed under the **Apache License 2.0** — permissive, with an
explicit patent grant. See [LICENSE](LICENSE) and [NOTICE](NOTICE) for details.

We welcome technical contributions, especially in the area of **NEON/AMX
microkernels** and **low-bit quantization research**. Start with
[CONTRIBUTING.md](CONTRIBUTING.md).

---

*“The future of AI is local, private, and embedded.”* 👻
