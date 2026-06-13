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
- **Reference:** llama.cpp `d05fe1d`, built with OpenBLAS
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
  bin/pi5/release/tests/bench_perf_sweep --gguf $M --seq-lens 128,256,512,1024 --decode-n 16 --warmup 16 --repeats 2
# llama.cpp reference (4 threads, prefill sweep):
llama-bench -m $M -p 128,256,512,1024 -n 32 -t 4
```

## Measured results (June 2026, quiesced)

Each engine at its best thread count; same weights, same quantization, CPU-only.
The full prefill sweep shows a **crossover**, not a flat gap:

| seq_len | llama.cpp (OpenBLAS, `d05fe1d`) | geist | winner |
| ---: | :---: | :---: | :--- |
|  128 | 22.1 | **32.6** | **geist 1.48×** |
|  256 | 30.0 | **30.4** | ~par |
|  512 | **33.2** | 27.0 | llama 1.23× |
| 1024 | **33.8** | 23.3 | llama 1.45× |
| **decode** | 6.7 | **6.9** | **geist 1.03×** |

**geist owns short context and decode; llama overtakes from ~512 on.** The reason
is the matmul path: geist's native int8 (W4A8) kernel has low fixed overhead, so
it leads the moment work arrives, but its per-token cost grows with the O(n²)
attention as the prompt lengthens (33 → 23 t/s). llama dequantizes to fp32 and
calls OpenBLAS sgemm — heavy fixed overhead that is ruinous at 128 tokens (22 t/s)
but amortizes over a long activation matrix (34 t/s at 1024). **Decode** is
memory-bandwidth-bound for both, so they land within a few percent and geist's
int8 `m=1` GEMV edges ahead. On Apple AMX the picture flips entirely in geist's
favour at every length (see [BENCHMARK.md](BENCHMARK.md)).

## Thread placement (quiesced)

The two phases want different core counts, and geist sets each automatically:

| threads | geist prefill pp256 | geist decode | llama prefill pp256 | llama decode |
| :---: | :---: | :---: | :---: | :---: |
| 4 | **30.0** | 6.96 | 29.9 | 6.43 |
| 3 | 24.0 | **7.11** | 29.9 | **7.05** |

**geist prefill** is compute-bound and scales with all 4 homogeneous A76 cores
(`omp_get_num_procs()`) — 24 → 30 from 3 to 4 threads. llama's pp256 is OpenBLAS-
bound and flat across 3/4 threads (~29.9), so it neither gains nor loses the 4th
core here. **Decode** is memory-bandwidth-bound for both and is fastest at 3 (the
4th thread just adds LPDDR contention) — geist defaults Pi 5 decode to 3. Override
with `GEIST_PREFILL_THREADS` / `GEIST_DECODE_THREADS`. (Earlier "3 beats 4 for
prefill too" numbers here were a background-load artifact — see the warning at the
top.)

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
