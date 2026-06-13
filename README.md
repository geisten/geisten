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

Build, then point the `geist` CLI at a GGUF:

```console
$ make
$ OMP_WAIT_POLICY=active bin/`mk/detect-target.sh`/release/tools/geist \
      gemma4-e2b-Q4_K_M.gguf "The capital of France is"
loaded gemma4-e2b-Q4_K_M.gguf (arch: transformer)
The capital of France is Paris.

$ OMP_WAIT_POLICY=active bin/`mk/detect-target.sh`/release/tools/geist \
      gemma4-e2b-Q4_K_M.gguf "Write a haiku about the ocean:" -n 40
Write a haiku about the ocean:

Blue waves crash on sand,
Salt spray kisses the warm air,
Ocean's deep secrets.
```

*Real output from the `geist` CLI on Gemma 4 E2B-it (Q4_K_M), greedy decode.
Reproduce with `make fetch-model` then the commands above. The whole stable
text-generation core is ~70 lines of C — see
[`examples/simple_generate.c`](examples/simple_generate.c) to embed it.*

---

## 🚀 Performance — geist vs llama.cpp (Gemma 4 E2B-it, Q4_K_M, CPU-only)

The **identical** GGUF on both engines, quiesced boxes, full prefill sweep from
128 → 1024 tokens plus decode. The story is not "one engine is faster" — it is
**two opposite scaling curves**, and which one wins depends on the machine *and*
the context length.

> **TL;DR** — On **Apple Silicon** geist wins prefill at *every* length and the
> lead **widens** with context (1.44× at 1024 tokens). On the **Pi 5** it is a
> crossover: geist owns short context (1.5× at 128), llama.cpp's OpenBLAS path
> overtakes from ~512 on. **Decode is roughly par on both** (geist slightly ahead).

### Apple M1 Max (8 P-cores) — prefill (tokens/s, higher is better)

| seq_len | llama.cpp `-ngl 0` | geist | winner |
| ---: | :---: | :---: | :--- |
|  128 | 135.2 | **140.2** | geist 1.04× |
|  256 | 136.6 | **137.8** | ~par |
|  512 | 116.7 | **135.4** | **geist 1.16×** |
| 1024 |  88.6 | **127.7** | **geist 1.44×** |

```
prefill t/s   (each █ ≈ 10 t/s)            geist stays flat ·· llama drops off
 geist  128 ██████████████ 140      llama  128 ██████████████ 135
        256 ██████████████ 138             256 ██████████████ 137
        512 ██████████████ 135             512 ████████████ 117
       1024 █████████████ 128             1024 █████████ 89
```

**Decode:** geist **31.5 t/s** vs llama.cpp 30.4 t/s (tg32) — par. geist's decode
eases from 31.5 (128-token context) to 24.6 (1024) as the KV-cache grows.

### Raspberry Pi 5 (Cortex-A76, 4 cores) — prefill (tokens/s, higher is better)

| seq_len | llama.cpp (OpenBLAS) | geist | winner |
| ---: | :---: | :---: | :--- |
|  128 | 22.1 | **32.6** | **geist 1.48×** |
|  256 | 30.0 | **30.4** | ~par |
|  512 | **33.2** | 27.0 | llama 1.23× |
| 1024 | **33.8** | 23.3 | llama 1.45× |

```
prefill t/s   (each █ ≈ 2.4 t/s)           geist fades ·· llama warms up
 geist  128 ██████████████ 33       llama  128 █████████ 22
        256 █████████████ 30               256 █████████████ 30
        512 ███████████ 27                 512 ██████████████ 33
       1024 ██████████ 23                 1024 ██████████████ 34
```

**Decode:** geist **6.9 t/s** vs llama.cpp 6.7 t/s (tg32) — geist's by a hair,
across all context lengths (6.9 → 6.1 as KV grows).

### Reading the numbers — why the curves cross

The two engines reach Q4_K matmuls through fundamentally different paths, and the
crossover falls straight out of that:

- **geist runs prefill on a native int8 (W4A8) kernel** — low fixed overhead, so
  it is fastest the moment work arrives (short context). But its per-token cost
  *grows* with context: at 1024 tokens the O(n²) attention is a much larger share,
  and it does not get cheaper per token the way a big GEMM does. → prefill **fades**
  as seq_len rises (Pi: 33 → 23).
- **llama.cpp dequantizes to fp32 and calls a BLAS sgemm** (OpenBLAS on the Pi,
  Accelerate on the Mac). BLAS carries a large fixed per-call overhead that is
  ruinous on small matrices but *amortizes* over the tall activation matrix of a
  long prompt. → on the Pi its prefill **warms up** (22 → 34) and overtakes geist
  around 512 tokens.
- **On the M1 Max the picture flips in geist's favour** because geist's dense-fp32
  path here is **Accelerate/AMX** (Apple's matrix coprocessor), which scales flat
  to long sequences, while llama.cpp's CPU-only path (`-ngl 0`) *degrades* sharply
  past 256 tokens. Net: geist's lead widens with length (1.04× → 1.44×).
- **Decode is memory-bandwidth-bound** for both (streaming 3 GB of weights per
  token dwarfs the compute), so the kernel differences wash out and the two land
  within a few percent — geist a touch ahead on both boxes.

**How to dig deeper.** To attribute the scaling, profile prefill *phase-by-phase*
(attention vs FFN-matmul vs PLE) at each seq_len rather than as one number:
build with `-DGEIST_PROFILE_QUANT` (per-kernel ns counters, auto-reported at exit)
and compare the attention fraction at 128 vs 1024 — that is the term pulling
geist's prefill down at long context. For llama.cpp, `llama-bench -p <n>` plus
`perf stat` (or Instruments on macOS) isolates where its CPU path loses time past
256 tokens. Reproduce any row with [BENCHMARK.md](BENCHMARK.md) (Mac) and
[BENCHMARK_PI5.md](BENCHMARK_PI5.md) (Pi).

*Measured June 2026 on quiesced hardware, both engines CPU-only on the identical
`Q4_K_M` GGUF, each at its best thread count (Mac auto-pins to the 8 P-cores; Pi
uses 4 threads). llama.cpp build `d05fe1d`. **Each geist figure is the mean of 10
measured repeats taken after a discarded warm-up run** (a throwaway prefill+decode
that pages the weights resident and spins up the OpenMP pool, so timings reflect
steady state, not cold-start); `llama-bench` warms up internally the same way.
**Always measure on a quiesced box** — on the 4-core Pi a single stray process
eating one core inverts the 4-thread numbers. These figures supersede the earlier
single-point table; the full sweep tells a more honest story than any one
sequence length.*

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
- C compiler with `-std=c23` support: gcc ≥ 14, or Apple-clang ≥ 16 (Xcode 16 / macOS 15).
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
OMP_WAIT_POLICY=active bin/`mk/detect-target.sh`/release/tools/eval_geist gguf_artifacts/gemma4-e2b-Q4_K_M.gguf
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
