# geist Benchmarks — Raspberry Pi 5

The Pi 5 (Cortex-A76, 4 cores, no `i8mm`) is geist's design target and the hard
case: an older ARM core where llama.cpp leans on a decades-tuned OpenBLAS fp32
path. These numbers were measured by the maintainer; the procedure below lets
you reproduce them on your own board.

## Setup

- **Board:** Raspberry Pi 5, 4× Cortex-A76, running 64-bit Raspberry Pi OS.
- **Build:** `make TARGET=pi5` (gcc-13, OpenBLAS, FFTW3, OpenMP).
- **Threads:** 4 (`OMP_NUM_THREADS=4 OMP_WAIT_POLICY=active`).
- **Model:** Gemma 4 E2B-it, Q4_K_M — the **identical** GGUF for both engines,
  producing **bit-identical** greedy output. Not a smaller or lossier model.
- **Reference:** llama.cpp built with OpenBLAS. **Record the commit hash** you
  compare against; throughput moves between llama.cpp releases.

```sh
make TARGET=pi5
make fetch-model
OMP_NUM_THREADS=4 OMP_WAIT_POLICY=active BENCH_THREADS=4 make bench-detailed
```

## Maintainer-measured results

Same weights, same quantization, bit-identical results — short-prompt prefill
ahead, decode at parity, long-prompt prefill within ~5 %:

| Engine (Pi 5, 4 threads, Q4_K_M) | Prefill pp128 | Prefill pp256 | Decode |
| :--- | :---: | :---: | :---: |
| llama.cpp (OpenBLAS) | 26.8 t/s | **31.6 t/s** | ~7.0 t/s |
| **geist** | **31.0 t/s** (1.16×) | 30.0 t/s (0.95×) | ~7.0 t/s (parity) |

The remaining long-prompt gap is raw NEON throughput vs OpenBLAS sgemm. On
`i8mm` / ARMv9 cores and Apple AMX, geist's int8 path pulls clearly ahead (see
[BENCHMARK.md](BENCHMARK.md)).

> These are maintainer measurements, not independently reproduced in CI (CI has
> no Pi 5 hardware). If you reproduce or refute them, please open a PR updating
> this table with your board, OS, thread count, and the llama.cpp commit.
