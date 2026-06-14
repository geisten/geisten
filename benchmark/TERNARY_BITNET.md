# Ternary (BitNet b1.58 / TQ2_0) — Pi 5 performance work

**Goal:** geist's ternary decode *and* prefill on a Raspberry Pi 5 (Cortex-A76,
SDOT, **no i8mm**) at or above `MAX(bitnet.cpp, llama.cpp)` on the same model.

**Status:** baseline **not yet measured on the Pi**. This doc records what is
verified, the measurement protocol, and the grounded (but unverified) plan, so
the first Pi run is turnkey.

---

## Verified so far (2026-06)

1. **geist runs a real BitNet ternary model end-to-end.** Previously the TQ2_0 /
   TL1 path was only *synthetically* unit-tested (`tests/test_tl1_parity.c`).
   Confirmed on `gianni-cor/bitnet_b1_58-large-TQ2_0` (0.7 B, 217 MB, real
   ternary weights, `general.architecture = bitnet`): geist loads the arch
   (generic GGUF-driven populator, SubLN + activation detection in
   `arch_family.c`) and the weights, and `bench_perf_sweep` drives the compute
   path to stable numbers.

2. **A76 kernel selection (default).** For TQ2_0 the resolver binds the **SDOT
   `q8a`** path for both decode (`cpu_neon_w_tq2_0_q8a_m1`) and prefill
   (`cpu_neon_w_tq2_0_q8a_mN`). The TL1 LUT path is opt-in: `GEIST_TL1=1`
   (decode) / `GEIST_TL1_PREFILL=1` (prefill). A code comment records that on
   the A76 SDOT prefill already *beats* TL1 (33.6 vs 21.0 t/s seq128, 2B-4T).

3. **The SDOT kernel is already well-tuned** (`kernels/tq2_0.c`): `vdotq_s32`
   with an "unbiased" trick (skips the per-element −1), two accumulators for
   dual-issue, and an `mt4` variant that reuses each weight tile across 4 tokens
   in prefill. No naive low-hanging fruit in the inner loop.

4. **Tokenizer gap for *older* BitNet models.** `1bitLLM/bitnet_b1_58-*` ship a
   llama **SentencePiece *unigram*** tokenizer (`scores` + `token_type`, **no
   `merges`**); geist implements gpt2-BPE and merge-driven SPM only, so it
   refuses unigram and there's no `tokenizer.bin`. This blocks *text* I/O on
   those models (not the compute path — `bench_perf_sweep` uses synthetic IDs).
   The target **2B-4T** model may differ; check its tokenizer KV before relying
   on coherence tests.

### Apple reference numbers (NOT the goal hardware — do not transfer to A76)

M1 Max, `large` model, real weights: SDOT decode ~90 tps vs **TL1 decode ~68
tps** — i.e. on this Apple setup TL1 decode is *slower* than SDOT, contradicting
an older "~2× decode" code comment (measurement was noisy: live desktop). Listed
only to flag that the TL1↔SDOT trade-off must be **measured per platform**; it
inverts between Apple and the A76.

---

## Measurement protocol

Use `benchmark/compare_ternary_pi5.sh` — runs geist (SDOT + TL1), llama.cpp, and
bitnet.cpp on the **same** GGUF / threads, from a **cool** baseline, mean-of-N
after a discarded warm-up, raw outputs saved. See `benchmark/BENCHMARK_PI5.md`
for the thermal/quiesce discipline (a stray process halves 4-thread numbers; a
hot board throttles whichever engine runs second).

```sh
MODEL=~/models/bitnet-2b4t-TQ2_0-v2.gguf \
LLAMA_BENCH=~/llama.cpp/build/bin/llama-bench \
BITNET_BENCH=~/BitNet/build/bin/llama-bench \
./benchmark/compare_ternary_pi5.sh
```

Decode is often fastest at **3 threads** (memory-bandwidth-bound), prefill at 4
(compute-bound) — geist auto-selects; sweep `THREADS=3` vs `4` for the references.

---

## Grounded hypothesis (UNVERIFIED — needs Pi numbers before any code change)

Per the project rule, **no kernel change ships without a quiesced+cooled Pi
measurement.** Working theory of where the A76 gap (if any) lives:

- **Decode (the headline BitNet metric):** memory-bandwidth-bound. TQ2_0 is
  ~2.06 bits/weight, the same order as llama.cpp's TQ2_0 and bitnet.cpp, so all
  three stream ~the same bytes/token → decode is likely **near parity already**,
  set by LPDDR bandwidth, not the kernel. First check: thread count (3),
  activation-quant overhead, no redundant weight passes. Expect little headroom.

- **Prefill (the battleground):** compute-bound. geist SDOT `mt4` vs bitnet.cpp's
  shape-specialized LUT codegen vs llama.cpp's SDOT. On an i8mm-free A76 everyone
  is capped at SDOT throughput (~32 int8 MAC/cyc). Candidate lever, *if* a gap
  shows: **wider token tiling (`mt8`/`mt16`)** to cut weight re-reads per output
  tile (the weight tile is re-loaded once per token-group), and verify prefill
  parallelizes over output rows without false sharing. Only pursue what the
  baseline says is behind.

---

## What unblocks *verified* progress

1. **Pi 5 access** (the only hardware that can validate the goal), or someone to
   run `compare_ternary_pi5.sh` and share the raw outputs.
2. **The 2B-4T TQ2_0 GGUF** (`bitnet-2b4t-TQ2_0-v2.gguf`, referenced in
   `mk/target-pi5.mk`). The official `microsoft/bitnet-b1.58-2B-4T-gguf` is
   `i2_s`, not TQ2_0 — needs a convert/quantize to TQ2_0, or the existing v2 file.
3. **Reference builds on the Pi:** `llama.cpp` (`llama-bench`, OpenBLAS) and
   `bitnet.cpp` (its `llama-bench` fork) for the same model.

With (1)–(3), the loop is: baseline → identify the behind-metric → targeted
A76 kernel change → re-measure cool → keep only if it wins, record in
`BENCHMARK_PI5.md`.
