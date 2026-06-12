# geist Benchmarks — Raspberry Pi 5

The Pi 5 (Cortex-A76, 4 cores, no `i8mm`) is geist's design target and the hard
case: an older ARM core where llama.cpp leans on a decades-tuned OpenBLAS fp32
path. The numbers below were measured on a real Pi 5; the procedure reproduces
them.

## Setup

- **Board:** Raspberry Pi 5 Model B Rev 1.1, 4× Cortex-A76, 4 GB RAM,
  64-bit Raspberry Pi OS (kernel 6.18, Debian).
- **geist build:** `make TARGET=pi5 CC=gcc` (gcc 14.2 — gcc-13 also fine;
  OpenBLAS + FFTW3 + OpenMP). Builds clean under `-Werror`.
- **Reference:** llama.cpp `ba1df05`, built with OpenBLAS
  (`cmake -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS -DGGML_NATIVE=ON`). CPU-only
  (the Pi has no GPU).
- **Model:** Gemma 4 E2B-it, Q4_K_M — the **identical** GGUF for both engines.
- **Threads:** 3 is fastest for **both** engines (see below).

```sh
make TARGET=pi5 CC=gcc
M=gguf_artifacts/gemma4-e2b-Q4_K_M.gguf
# geist (auto-pins prefill + decode to cores-1 = 3 on the Pi 5):
GEIST_BENCH_PP=256 GEIST_BENCH_TG=64 OMP_WAIT_POLICY=active \
  bin/pi5/release/tests/bench_session_throughput $M
# llama.cpp reference:
llama-bench -m $M -p 128,256 -n 64 -t 3
```

## Measured results (June 2026)

Each engine at its best thread count (= 3); same weights, same quantization,
CPU-only.

| Engine (Pi 5, 3 threads, Q4_K_M) | Prefill pp128 | Prefill pp256 | Decode tg64 |
| :--- | :---: | :---: | :---: |
| llama.cpp (OpenBLAS, `ba1df05`) | **29.6 t/s** | **30.0 t/s** | 6.78 t/s |
| **geist** | 25.1 t/s (0.85×) | 24.0 t/s (0.80×) | **6.6 t/s** (0.97×) |

**Decode is at parity** (geist 0.97× of llama). **Prefill is ~15–20 % behind:**
the Pi 5's prefill bottleneck is the `m>1` GEMM, where OpenBLAS's hand-tuned
sgemm beats geist's NEON kernels.

geist's `q4k_predecode` fast path (which speeds prefill on Apple) is gated to
`has_accelerate`, i.e. **off on the Pi — correctly.** Forcing it on
(`GEIST_Q4K_PREDECODE=1 GEIST_Q4K_MTILE_PREFILL=1 GEIST_Q4K_NTILE_PREFILL=1`)
makes prefill **slower**, measured pp256 24.0 → 17.4 tps (−28 %): the predecoded
block is ~1.9× the bytes of raw Q4_K, and the Pi 5's LPDDR4X bandwidth makes that
byte-doubling cost more than the saved scale-unpack compute (the same
bandwidth-vs-compute trade-off as the Q8_0 engine on Apple, sharper here). So the
Pi prefill gap is genuine NEON-`m>1`-vs-OpenBLAS-sgemm, not a missing fast path.
Closing it — a quality-safe OpenBLAS sgemm prefill, or a faster NEON `m>1`
kernel — is tracked work. On `i8mm`/ARMv9 cores and Apple AMX the picture differs
(see [BENCHMARK.md](BENCHMARK.md)).

## Thread placement — leave one core

Both engines regress badly when all 4 cores run compute threads: the static
schedule then contends with the OS / OpenMP master. Measured:

| threads | geist prefill pp256 | llama prefill pp256 | llama decode tg64 |
| :---: | :---: | :---: | :---: |
| 3 | **24.0** | **30.0** | **6.78** |
| 4 | 10.9 | 27.1 | 2.26 |

geist now defaults Pi 5 prefill **and** decode to `cores-1` (3); previously
prefill defaulted to all 4 cores, which more than halved it (10.9 vs 24.0).
Override with `GEIST_PREFILL_THREADS` / `GEIST_DECODE_THREADS`.

## Quality

The Pi 5 build is numerically sound: the function-calling / JSON benchmark
(`tools/eval_tooling.py`) scores **28/28**, identical to the Apple build, so
`-ffast-math` + Cortex-A76 NEON do not perturb greedy output on these tasks.

> Measured on one Pi 5 in June 2026, not in CI (CI has no Pi 5 hardware). If you
> reproduce or refute these, please open a PR with your board, OS, thread count,
> and the llama.cpp commit.
