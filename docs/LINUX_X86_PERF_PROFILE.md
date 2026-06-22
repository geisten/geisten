# Linux/x86_64 perf profile vs llama.cpp

Hard-data comparison of geist's `cpu_x86` backend against llama.cpp on
the reference 9950X (Zen 5, 16C/32T, DDR5-6400). Measured 2026-06-22 on
identical Gemma 4 E2B-it Q4_K_M, single bench run, `OMP_NUM_THREADS=16
OMP_WAIT_POLICY=active OMP_PROC_BIND=close OMP_PLACES=cores`.

The profile data here drives the Phase 3 optimization direction: the
gap to llama.cpp is **not** a thread-scaling problem and **not** a
memory-bandwidth problem — it is a **per-cycle compute-density** problem
in the inner loop.

## `perf stat` headline: IPC collapses 6.4×

For `pp128` only (decode=0), single bench iteration:

| Metric                  |        geist |    llama.cpp |       ratio |
| ----------------------- | -----------: | -----------: | ----------: |
| Wall time               |    7.31 s    |    1.19 s    | 6.1× slower |
| User CPU time           |   82.9 s     |    9.33 s    | 8.9× more   |
| **Instructions**        |    **201 G** |     **95 G** | **2.1× more** |
| **Cycles**              |    **426 G** |    **32 G**  | **13.5× more** |
| **IPC (insn/cycle)**    |   **0.47**   |   **3.01**   | **6.4× lower** |
| Frequency               |  5.08 GHz    |  3.29 GHz    | (we boost higher) |
| Frontend stall          |   0.48 %     |   3.08 %     | similar     |
| Branch miss             |   0.10 %     |   0.32 %     | similar     |
| Page faults             |   2.2 M      |   380 K      | 5.8× more   |

`/usr/bin/time -v`, same run:

| Metric                  |        geist |    llama.cpp |       ratio |
| ----------------------- | -----------: | -----------: | ----------: |
| `%CPU` (parallelism)    |    **1170 %** |   **931 %**  | both ~10 cores effective |
| RSS                     |    7.5 GB    |    4.3 GB    | 1.7× more   |

## Diagnosis

1. **Threading is fine**: `1170 %` CPU means ~11.7 cores actively
   executing — comparable to llama.cpp's 9.3. We were briefly suspicious
   of OMP fork/join overhead, but `OMP_WAIT_POLICY=active` confirmed
   threads are *running*, not stuck on barriers.

2. **We execute 2.1× more instructions for the same work.** That is the
   weight-layout efficiency: every VPMADDUBSW in llama.cpp's inner
   produces partial sums for **8 cells in parallel** (lane-parallel),
   while our VPDPBUSD / VDPBF16PS produces a partial sum for **1 cell**.
   For an equivalent matmul we therefore emit ~8× more cell-level dot
   product instructions, which the inner-loop overhead (loads, scale
   broadcasts, accumulator updates) inflates to a ~2.1× total instruction
   count.

3. **We run those instructions at 6.4× lower IPC.** Zen 5's 6-wide front-
   end issues ~6 instructions per cycle when the dependency graph has
   enough parallelism. llama.cpp's `acc_rows[16]` and `acc_min_rows[16]`
   provide 16 independent fp32 accumulator chains that overlap across
   iterations; the front-end stays full. Our BF16 path writes to ONE
   shared `acc_v[i, jj]` per (i, jj) inner step, and each VFMADD has a
   4-cycle latency. With 4 m-rows and the M_TILE=4 fmadd chain that's a
   **16-cycle critical path** the compiler cannot reorder out of.

4. **Net effect**: 13.5× more cycles total = 2.1× more instructions ×
   6.4× lower IPC. The wall-clock ratio is 6.1× (not 13.5×) because
   geist runs at a higher boost frequency (5.08 vs 3.29 GHz), which
   partially compensates for the cycle deficit.

## Implication for Phase 3

The fix is **architectural in the inner loop**, not tuning the existing
kernel:

- **Independent accumulators** (`acc_rows[16]`) break the dependency
  chain → expected IPC ~3 (Zen 5 peak utilization) → ~6× cycle reduction.
- **Lane-parallel weight layout** (`block_q4_Kx8`) gives 8 cells per
  VPMADDUBSW → ~2× instruction reduction.
- Compound expected lift: ~10–12× on the inner kernel, bringing prefill
  from ~28 t/s to **~150–250 t/s** — within 2–3× of llama.cpp's 514
  rather than the current 18×.

That is exactly what the Phase 3 spec lays out (`docs/LINUX_X86_SPEC.md`
Phase 1b Step 3 + Phase 2 native VDPBF16PS). The kernel port is mechanical
(the scalar reference at `kernel_q4kx8_gemm_scalar.c` is verified correct
and serves as the oracle for any AVX variant), but the shuffle patterns
in llama.cpp's `ggml_gemm_q4_K_8x8_q8_K` (~640 LOC of intrinsics) are
intricate enough that a single-turn port has consistently produced bugs.
The recommended approach is a multi-turn focused debug session where each
intermediate value (d/dmin decode, mins_01 build, iacc_0/iacc_1 lane
assignment) is cross-checked against the scalar reference before moving
on.

## Reproduce

```sh
# geist
make TARGET=linux BACKENDS="cpu_x86 cpu_scalar"
/usr/bin/time -v env OMP_NUM_THREADS=16 OMP_WAIT_POLICY=active \
    OMP_PROC_BIND=close OMP_PLACES=cores GEIST_BENCH_BACKEND=cpu_x86 \
    bin/linux/release/tests/bench_perf_sweep \
    --gguf gguf_artifacts/gemma4-e2b-Q4_K_M.gguf \
    --seq-lens 128 --decode-n 0 --warmup 8 --repeats 1 --threads 16

# llama.cpp (commit 7c082bc)
/usr/bin/time -v llama-bench -m gemma4-e2b-Q4_K_M.gguf -p 128 -n 0 \
    -t 16 -ngl 0 -r 1

# perf stat (needs perf_event_paranoid <= 1):
perf stat -- <above command>
```
