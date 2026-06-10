# geist Benchmarks

This file documents **how** geist is benchmarked and records reproducible
results. The auto-recorded table at the bottom is written by
`make bench-small` / `make bench-detailed`; everything above the
`<!-- BENCH:AUTO -->` marker is hand-maintained prose and is preserved across
runs.

> **Reproducibility over headline numbers.** Every recorded row is tagged with
> host, OS, target, mode, thread count, and model, so results from different
> machines never silently overwrite each other. Throughput varies a lot with
> core count and `OMP_NUM_THREADS`/`OMP_WAIT_POLICY` — always read a row with
> its tags.

## Methodology

Perf is measured by the `bench_session_throughput` binary (built by
`make bench`), driven through `tools/bench_quality_perf.py`:

- **Model:** Gemma 4 E2B-it, Q4_K_M (`make fetch-model`).
- **Warm-up:** 64-token prefill, then reset (excludes cold caches / page-ins).
- **Prefill:** time to prefill 200 tokens → tok/s.
- **Decode:** time to autoregressively decode 50 tokens → tok/s.
- **Best-of-N:** `small` takes the best of 2 runs, `detailed` the best of 5,
  to suppress scheduler noise.

```sh
make fetch-model                        # one-time, ~3.1 GB
OMP_WAIT_POLICY=active make bench-small  # records a row below
# tune threads:
BENCH_THREADS=6 OMP_WAIT_POLICY=active make bench-detailed
```

`OMP_WAIT_POLICY=active` (or `KMP_BLOCKTIME=infinite`) matters on the `mac-omp`
target — passive wait adds large per-matmul thread-pool wake-up overhead. Thread
*placement* is handled automatically: geist pins prefill to the performance
cores and decode to P-cores−1 (see the comparison below). Override the workload
size with `GEIST_BENCH_PP` / `GEIST_BENCH_TG` to match an external reference
(e.g. `llama-bench -p 512 -n 128`).

## Comparison vs llama.cpp (Apple M1 Max, CPU, measured June 2026)

Same machine, same `gemma4-e2b-Q4_K_M.gguf`, both CPU-only. llama.cpp build
`d48a56eff` (9430) run with `-ngl 0` (BLAS/Accelerate, no GPU offload). Each
engine at its best thread count; `pp512` = prompt-processing throughput,
`tg128` = token-generation throughput.

| Engine | pp512 (t/s) | tg128 (t/s) |
| :--- | :---: | :---: |
| llama.cpp `-ngl 0`, best of t∈{6,7,8} | 137 | 35.4 |
| **geist** (P-core pinned) | **143** | 33 |
| ratio (geist / llama.cpp) | **1.04×** | 0.93× |

Reproduce:

```sh
# geist (auto-pins prefill→P-cores, decode→P-cores−1):
GEIST_BENCH_PP=512 GEIST_BENCH_TG=128 OMP_WAIT_POLICY=active \
  bin/$(mk/detect-target.sh)/release/tests/bench_session_throughput model.gguf
# llama.cpp (CPU-only, matched workload):
llama-bench -m model.gguf -ngl 0 -t 8 -p 512 -n 128
```

**Key finding — thread placement dominates on heterogeneous cores.** On Apple
Silicon the efficiency ("E") cores stall a static OpenMP partition: defaulting
to `omp_get_num_procs()` (all 10 cores) gave pp512 ≈ 91 t/s, while pinning to
the 8 performance cores gives ≈ 143. geist now reads `hw.perflevel0.physicalcpu`
and pins prefill to the P-cores and decode to P-cores−1 (decode fires ~210 tiny
matmuls/token and contends when every core is saturated). Override with
`GEIST_PREFILL_THREADS` / `GEIST_DECODE_THREADS`.

The remaining decode gap (~7%) is the 262K-wide lm_head GEMV (66% of decode
time, Q4_K) re-unpacking block scales per row; wiring the predecoded layout into
the decode path is in progress. To reproduce a head-to-head on *your* hardware
with the llama.cpp commit pinned, see
[docs/BENCHMARKING.md](docs/BENCHMARKING.md).

Quality (perplexity / KL-divergence vs the reference, sampled MMLU/GSM8K) is
likewise documented in [docs/BENCHMARKING.md](docs/BENCHMARKING.md); it needs
the HF tokenizer and datasets and is not part of the hermetic `make` flow.

<!-- BENCH:AUTO -->

| Date | Model | Host | OS | Target/Mode | Threads | Prefill tok/s | Decode tok/s |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| 2026-06-10 | gemma4-e2b-Q4_K_M.gguf | MBP-Germar.local/arm64 | Darwin 25.5.0 | mac-omp/release | default | 77.2 | 10.2 |
