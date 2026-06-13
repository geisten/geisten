# geist 👻

[![CI](https://github.com/geisten/geist/actions/workflows/ci.yml/badge.svg)](https://github.com/geisten/geist/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![C Standard](https://img.shields.io/badge/C-C23-orange.svg)](https://en.wikipedia.org/wiki/C23_(C_standard_revision))
[![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20Linux%20(ARM64)-lightgrey.svg)](#-build--usage)
[![Status](https://img.shields.io/badge/status-experimental%20(v0.1.0)-yellow.svg)](#-status)

**geist** is a high-performance C23 inference engine that runs LLMs **on the CPU
with zero dependencies** — one small static binary, no BLAS, no Python, no CUDA,
no runtime to install. Copy it to the machine and it runs.

That is the bet, and it is a different one from the universal engines:

- **Dependency-free, CPU-only.** The default ARM build links nothing but libc /
  libm / libgomp and folds them in statically (~860 KB ELF). No GPU, no BLAS
  package, no model server — it runs anywhere the CPU architecture matches.
- **Focused, not universal.** Where llama.cpp aims to run *every* model on *every*
  backend, geist deliberately does **a few models excellently** (Gemma 4 E2B-it
  today; Ternary 1.58-bit BitNet next). That focus is what lets it bind every
  tensor to a hand-picked kernel at load time and beat a generic dispatch loop.
- **Edge-first, Raspberry Pi 5 as the primary optimization target.** The tuning
  effort goes to the $50–$100 Cortex-A76 board first — not the datacenter GPU —
  for **Low-Bit (IQ/TQ)** and **native multimodal audio** inference where
  llama.cpp is the default but you need more speed, less RAM, or no dependency tree.

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

The **identical** GGUF on both engines, full prefill sweep 128 → 1024 tokens. The
story is not "one engine is faster" — it is **two opposite scaling curves**, and
which one wins depends on the machine *and* the context length.

**Apple M1 Max** — prefill t/s (best-of-10, both engines):

| seq_len | 128 | 256 | 512 | 1024 |
| :-- | :---: | :---: | :---: | :---: |
| llama.cpp `-ngl 0` | 141 | 147 | 128 | 97 |
| **geist** | **164** | **161** | **150** | **144** |
| | geist 1.16× | geist 1.10× | geist 1.17× | **geist 1.48×** |

**Raspberry Pi 5** — prefill t/s (mean-of-10, quiesced):

| seq_len | 128 | 256 | 512 | 1024 |
| :-- | :---: | :---: | :---: | :---: |
| llama.cpp (OpenBLAS) | 22.1 | 30.0 | **33.2** | **33.8** |
| **geist** | **32.4** | **30.5** | 27.0 | 23.3 |
| | geist 1.47× | ~par | llama 1.23× | llama 1.45× |

On **Apple Silicon** geist wins prefill at *every* length and the lead **widens**
with context (1.48× at 1024) — geist's dense path uses **Accelerate/AMX**, which
stays flat, while llama.cpp's CPU path drops off. On the **Pi 5** it is a
**crossover**: geist's low-overhead native int8 owns short context (1.47× at 128),
while llama.cpp's BLAS sgemm amortizes its fixed cost over long prompts and
overtakes from ~512 on. **Decode is ~par on both** (Pi: geist 6.9 vs llama 6.7).

📊 **Full sweep, ASCII charts, the "why the curves cross" analysis, and the
methodology (why best-of on the live Mac, mean-of-10 on the quiesced Pi) live in
[`benchmark/`](benchmark/README.md).**

---

## 🛠 Why geist?

### 1. Dependency-Free & Static
The ARM release links only libc / libm / libgomp and folds them in statically — an ~860 KB ELF with **no dynamic dependencies** (`ldd` → *not a dynamic executable*), no BLAS package, no Python, no GPU runtime. A `geist_gemm` abstraction makes BLAS/FFT *optional per platform*: ARM ships fully self-contained (native NEON fp32 + a vendored FFT), macOS keeps Accelerate/AMX because it is always present. Distribution is "copy one file."

### 2. Zero-Dispatch Architecture
Unlike generic engines that use complex layer-dispatch loops, `geist` uses **Kernel Binding**. At load time, every tensor is bound directly to a specialized kernel pointer. This eliminates vtable overhead and management logic during the hot path—critical for single-core-heavy edge CPUs, and it is only practical because geist targets a *focused* set of models rather than every architecture.

### 3. Ternary (1.58-bit) as a First-Class Citizen
We don't treat low-bit formats as an afterthought. Our backend is built for a **multiplication-free future**. `geist` includes native paths for BitNet b1.58, where the CPU only performs additions and subtractions, maximizing performance on hardware without powerful NPUs.

### 4. Native Multimodal Audio
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
