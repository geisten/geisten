# geist Benchmarks

geist vs **llama.cpp**, the **identical** `Gemma 4 E2B-it Q4_K_M` GGUF on both
engines, CPU-only, on the two reference machines. The story is not "one engine is
faster everywhere" — it is **two opposite scaling curves**, and which one wins
depends on the machine *and* the context length.

| | [Raspberry Pi 5](BENCHMARK_PI5.md) | [Apple M1 Max](BENCHMARK.md) |
| :-- | :-- | :-- |
| Role | **primary optimization target** (edge) | dev box / Apple-AMX reference |
| Quant matmul path | native int8 (W4A8) | native int8 |
| Dense fp32 path | native NEON (BLAS-free) | Accelerate / **AMX** |
| Measurement | quiesced → **mean of 10** | live desktop → **best of 10** |

> **TL;DR** — On **Apple Silicon** geist wins prefill at *every* length and the
> lead **widens** with context (1.48× at 1024 tokens). On the **Pi 5** geist's
> prefill is now ~flat across the sweep (34 → 31.5 t/s): it wins short/mid context
> (1.55× at 128) and llama.cpp only edges ahead at 1024 (1.07×). **Decode is ~par
> on both** (geist slightly ahead).

---

## Apple M1 Max (8 P-cores) — prefill (tokens/s, higher is better)

| seq_len | llama.cpp `-ngl 0` | geist | winner |
| ---: | :---: | :---: | :--- |
|  128 | 141 | **164** | **geist 1.16×** |
|  256 | 147 | **161** | **geist 1.10×** |
|  512 | 128 | **150** | **geist 1.17×** |
| 1024 |  97 | **144** | **geist 1.48×** |

```
prefill t/s   (each █ ≈ 10 t/s)            geist stays flat ·· llama drops off
 geist  128 ████████████████ 164    llama  128 ██████████████ 141
        256 ████████████████ 161           256 ███████████████ 147
        512 ███████████████ 150            512 █████████████ 128
       1024 ██████████████ 144            1024 ██████████ 97
```

**Decode:** ≈ par (~26 t/s both). On a live desktop the 16-token decode window is
too jitter-prone to rank — see the Pi for the controlled decode result.

geist's dense-fp32 path here is **Accelerate/AMX** (Apple's matrix coprocessor),
which holds ~flat from 128 → 1024 tokens (164 → 144), while llama.cpp's CPU-only
path degrades sharply past 256 (147 → 97). [Full write-up + decode-kernel
investigation →](BENCHMARK.md)

## Raspberry Pi 5 (Cortex-A76, 4 cores) — prefill (tokens/s, higher is better)

| seq_len | llama.cpp (OpenBLAS) | geist | winner |
| ---: | :---: | :---: | :--- |
|  128 | 22.1 | **34.3** | **geist 1.55×** |
|  256 | 30.0 | **34.1** | **geist 1.14×** |
|  512 | 33.2 | **33.0** | ~par |
| 1024 | **33.8** | 31.5 | llama 1.07× |

```
prefill t/s   (each █ ≈ 2.4 t/s)           geist ~flat ·· llama warms up
 geist  128 ██████████████ 34       llama  128 █████████ 22
        256 ██████████████ 34               256 █████████████ 30
        512 ██████████████ 33               512 ██████████████ 33
       1024 █████████████ 32              1024 ██████████████ 34
```

**Decode:** geist **6.9 t/s** vs llama.cpp 6.7 t/s — geist's by a hair, across all
context lengths. geist's prefill curve was flattened by parallelizing the O(n²)
attention core (it used to fade to 23 t/s at 1024 — see the write-up).
[Full write-up + thread placement →](BENCHMARK_PI5.md)

---

## Reading the numbers — why the curves look the way they do

The two engines reach the Q4_K matmuls through fundamentally different paths:

- **geist runs prefill on a native int8 (W4A8) kernel** — low fixed overhead, so
  it is fastest the moment work arrives (short context). Its attention is O(n²),
  but the SDPA core is now **parallelized across cores** (it used to be serial,
  which made the Pi curve fade to 23 t/s at 1024); spread over 4 A76 cores the
  per-token rise is largely absorbed, so the Pi prefill curve stays **~flat**
  (34 → 31.5) and llama only catches it at the very longest length.
- **llama.cpp dequantizes to fp32 and calls a BLAS sgemm** (OpenBLAS on the Pi,
  Accelerate on the Mac). BLAS carries a large fixed per-call overhead — ruinous on
  small matrices but *amortizing* over the tall activation matrix of a long prompt.
  → on the Pi its prefill **warms up** (22 → 34), reaching geist only around 1024.
- **On the M1 Max the picture is geist-favoured at every length** because geist's
  dense-fp32 path is **Accelerate/AMX**, which scales flat to long sequences, while
  llama.cpp's CPU-only path (`-ngl 0`) *degrades* sharply past 256 tokens.
- **Decode is memory-bandwidth-bound** for both (streaming the weights per token
  dwarfs the compute), so the kernel differences wash out and the two land within
  a few percent — geist a touch ahead on both boxes.

> The flat Pi prefill curve is recent: profiling showed the O(n²) attention stage
> climbing 22 %→45 % of prefill with its SDPA **core single-threaded**;
> parallelizing it (bit-exact) lifted pp512 +22 % and pp1024 +35 %. See
> [BENCHMARK_PI5.md](BENCHMARK_PI5.md).

**How to dig deeper.** To attribute the scaling, profile prefill *phase-by-phase*
(attention vs FFN-matmul vs PLE) at each seq_len rather than as one number: build
with `-DGEIST_PROFILE_QUANT` (per-kernel ns counters, auto-reported at exit) and
compare the attention fraction at 128 vs 1024. For llama.cpp, `llama-bench -p <n>`
plus `perf stat` (or Instruments on macOS) isolates where its CPU path loses time
past 256 tokens.

---

## Methodology — and why the two machines use different aggregation

Both machines run the **same model and quantization**, both engines CPU-only, each
at its best thread count, after a **discarded warm-up run** (the runtime pages
weights resident and spins up the OpenMP pool, so timings reflect steady state, not
cold-start). llama.cpp build `d05fe1d`.

- **Raspberry Pi 5 — `mean of 10`.** A dedicated headless box, genuinely quiesced
  (load 0.0). The mean is meaningful: spread is <2 % run-to-run. This is the
  controlled measurement.
- **Apple M1 Max — `best of 10`.** A developer workstation that *cannot* be
  quiesced while in use (WindowServer, browser, IDE all contend for the P-cores).
  On a contended box the **mean** is dominated by interference spikes (±20 %
  run-to-run), so we report the **best** of 10 repeats — the least-interrupted run,
  which approximates the uncontended ceiling and is stable across independent
  campaigns. Both engines use best-of here, so the comparison stays
  apples-to-apples.

**Always measure on a quiesced box.** On the 4-core Pi a single stray process
eating one core inverts the 4-thread numbers. Reproduce with `bench_perf_sweep`
(geist) and `llama-bench` (reference) — see [BENCHMARKING.md](BENCHMARKING.md) for
the exact commands, the correctness gate (bit-identical greedy output on the same
weights), and the quality/PPL caveats.

## Files

- **[BENCHMARK_PI5.md](BENCHMARK_PI5.md)** — Raspberry Pi 5 (Cortex-A76): the edge
  target. Full sweep, thread placement, the int8-vs-OpenBLAS analysis.
- **[BENCHMARK.md](BENCHMARK.md)** — Apple M1 Max: the AMX reference + the
  auto-recorded `make bench-small`/`bench-detailed` results table + the
  decode-kernel investigation.
- **[BENCHMARKING.md](BENCHMARKING.md)** — how to produce trustworthy numbers
  (reproduce, compare-ref, quality/MMLU procedures).
