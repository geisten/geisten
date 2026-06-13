# geist Benchmarks — Raspberry Pi 5

The Pi 5 (Cortex-A76, 4 cores, no `i8mm`) is geist's design target and the hard
case: an older ARM core where llama.cpp leans on a decades-tuned OpenBLAS fp32
path. The numbers below were measured on a real, **quiesced** Pi 5.

> ⚠️ **Measure on a quiesced board.** A single stray background process eating
> one core silently halves the 4-thread numbers (4 OMP threads then oversubscribe
> the 3 free cores and the static schedule stalls). Check `uptime` / `top` first;
> an early version of this page reported numbers confounded exactly this way.

## Setup

- **Board:** Raspberry Pi 5 Model B Rev 1.1, 4× Cortex-A76, 4 GB RAM,
  64-bit Raspberry Pi OS (kernel 6.18, Debian).
- **geist build:** `make TARGET=pi5 CC=gcc` (gcc 14.2; gcc ≥ 14 required for `-std=c23`;
  OpenBLAS + FFTW3 + OpenMP). Builds clean under `-Werror`.
- **Reference:** llama.cpp `ba1df05`, built with OpenBLAS
  (`cmake -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS -DGGML_NATIVE=ON`). CPU-only
  (the Pi has no GPU).
- **Model:** Gemma 4 E2B-it, Q4_K_M — the **identical** GGUF for both engines.
- **Weights:** `GEIST_WEIGHT_MMAP=1` (the default) keeps the 1.93 GB PLE table
  mmap-aliased rather than copied resident — important on the 4 GB board.

```sh
make TARGET=pi5 CC=gcc
M=gguf_artifacts/gemma4-e2b-Q4_K_M.gguf
# geist auto-runs prefill on all 4 cores and decode on 3:
GEIST_WEIGHT_MMAP=1 OMP_WAIT_POLICY=active \
  bin/pi5/release/tests/bench_perf_sweep --gguf $M --seq-lens 128,256 --decode-n 16 --warmup 16 --repeats 2
# llama.cpp reference (its prefill is fastest at 4 threads, decode at 3):
llama-bench -m $M -p 128,256 -n 32 -t 4
```

## Measured results (June 2026, quiesced)

Each engine at its best thread count; same weights, same quantization, CPU-only.

| Engine (Pi 5, Q4_K_M, CPU) | Prefill pp128 | Prefill pp256 | Decode |
| :--- | :---: | :---: | :---: |
| llama.cpp (OpenBLAS, `ba1df05`) | **37.5 t/s** | **38.0 t/s** | 6.88 t/s |
| **geist** | 31.5 t/s (0.84×) | 29.7 t/s (0.78×) | **7.1 t/s** (1.03×) |

**geist wins decode** (1.03× of llama — the memory-bound `m=1` GEMV with int8
SDOT edges out OpenBLAS). **Prefill is ~16–22 % behind:** the `m>1` GEMM is where
OpenBLAS's hand-tuned sgemm beats geist's NEON kernels. Closing it — a
quality-safe OpenBLAS sgemm prefill, or a faster NEON `m>1` kernel — is tracked
work. On `i8mm`/ARMv9 cores and Apple AMX the picture differs (see
[BENCHMARK.md](BENCHMARK.md)).

## Thread placement (quiesced)

The two phases want different core counts, and geist sets each automatically:

| threads | geist prefill pp256 | geist decode | llama prefill pp256 | llama decode |
| :---: | :---: | :---: | :---: | :---: |
| 4 | **30.0** | 6.96 | **38.0** | 6.20 |
| 3 | 24.0 | **7.11** | 30.2 | **6.88** |

**Prefill** is compute-bound and scales with all 4 homogeneous A76 cores
(`omp_get_num_procs()`). **Decode** is memory-bandwidth-bound and is fastest at 3
(the 4th thread just adds LPDDR contention) — geist defaults Pi 5 decode to 3.
Both engines follow the same pattern. Override with `GEIST_PREFILL_THREADS` /
`GEIST_DECODE_THREADS`. (Earlier "3 beats 4 for prefill too" numbers here were a
background-load artifact — see the warning at the top.)

## Predecode is correctly off on the Pi

geist's `q4k_predecode` fast path (which speeds prefill on Apple) is gated to
`has_accelerate`, i.e. **off on the Pi.** Forcing it on
(`GEIST_Q4K_PREDECODE=1 GEIST_Q4K_MTILE_PREFILL=1 GEIST_Q4K_NTILE_PREFILL=1`)
makes prefill **slower** — measured clean pp256 30.3 → 21.7 tps (−28 %): the
predecoded block is ~1.9× the bytes of raw Q4_K, and the Pi's LPDDR4X bandwidth
makes that byte-doubling cost more than the saved scale-unpack compute (the same
bandwidth-vs-compute trade-off as the Q8_0 engine on Apple, sharper here). The
gate is correct.

## Quality

The Pi 5 build is numerically sound: the function-calling / JSON benchmark
(`tools/eval_tooling.py`) scores **28/28**, identical to the Apple build, so
`-ffast-math` + Cortex-A76 NEON do not perturb greedy output on these tasks.

> Measured on one Pi 5 in June 2026, not in CI (CI has no Pi 5 hardware). If you
> reproduce or refute these, please open a PR with your board, OS, thread count,
> and the llama.cpp commit.
