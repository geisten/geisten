# Metal GPU inference — llama.cpp parity plan

Status: agreed plan (grill-me session, 2026-06-27). Not yet implemented.
Goal: bring the Metal GPU path on par with llama.cpp on Apple Silicon.

## Fixed baseline (the bar)

- **Hardware:** Apple M1 Max, 32-core GPU, 64 GB unified, ~400 GB/s peak
  (tune against *achievable* ~330–380 GB/s, measured with a copy kernel — not 400).
- **Comparison tool:** `llama-bench` at `/opt/homebrew/bin/llama-bench` (installed).
- **Models:**
  - `solar-10.7b-v1.0.Q4_K_M.gguf` — **kernel bar** (vanilla LLaMA arch, full causal,
    GQA n_kv=8). Isolates kernel quality with no arch-divergence. geist supports llama
    arch (`populate_llama`, arch_family.c:79) — confirm it actually loads as step 0.
  - `models/gemma-4-E2B-it-Q4_K_M.gguf` — **product check** (Gemma-3n: PLE, sliding
    window, MQA n_kv=1, soft-cap). Reported alongside, not tuned against directly.
  - `gemma-E4B` — **deferred**; not on disk, must be converted. Add as a third
    confirming number later. Does not gate the start.
- **Metric:** decode (tg) and prefill (pp) reported **separately** (never a single
  blended tok/s). Context regime: **long** (8k+).

## Key facts that shaped the plan

- Metal already: per-sequence command batching (1 commit+wait per token, not per-op),
  capture-replay decode, fused FFN/attn/PLE blocks, simdgroup matvec, device-resident
  weights + KV. It is **not** naive.
- Attention **already respects the sliding window** (backend.c:1392 `slo` bound).
- Gemma decode at long ctx is **weight-bandwidth-bound** (MQA n_kv=1 + 512-window on
  4/5 layers) — flash-attention barely helps Gemma decode.
- Solar is full-causal GQA8 → KV grows with ctx; flash helps decode there and dominates
  prefill (O(seq²)) on both models.
- **No QK soft-cap** in attention (tanh soft-cap is final-logits only, head.c:348) →
  flash kernel keeps a standard online-softmax.
- Gemma head_dim is **512 on full layers, 256 on sliding** (arch_family.c:67) — the
  load-bearing risk for flash tiling vs 32 KB threadgroup memory.
- W4A8 is a **dead end on Apple**: weight reads stay full Q4_K bytes, no dp4a → saves
  dequant ALU not bandwidth; decode is bandwidth-bound. Deprioritize. See
  [[vulkan-experimental-variants]].

## Phase 0 — Measure (do first, no code beyond instrumentation)

1. Confirm geist loads & runs solar-10.7b on the Metal path.
2. Run `bench_session_throughput.c` and `llama-bench` back-to-back on the *same* GGUF
   files, both models, at the long-ctx point. Record pp + tg ratios.
3. Add a copy-kernel achievable-bandwidth probe + roofline-% to `bench_q4k_kernel.c`.

## Phase 1 — Matvec / decode bandwidth (start here)

- **Metric:** % of *achievable* memory bandwidth (not tok/s).
- **Decision rule:** measure best existing decode variant (**NT8**) on solar + Gemma-E2B
  real shapes. If NT8 ≥ ~75% achievable → matvec is done, **stop, go to Phase 2**.
  If < ~75% → rewrite, with llama.cpp `kernel_mul_mv_q4_K_f32` (simdgroup tiling +
  load-scales-once) as the reference target.
- **Consolidation:** if we rewrite, collapse the **decode** matvec zoo
  (`base`/`n4`/`nt4`/`nt8`/`w4a8` + their `GEIST_METAL_*` flags) to **one** unconditional
  kernel. Keep the NT4 repack + `PACK_CACHE_DIR` infra if the winner needs it.
  **Keep the prefill matmul variants** (`m8`/`m16`/`m16_n2`/`mm_sg`) — substrate for
  Phase 2. Delete **after** the shootout locks a winner, never before.
- **Correctness gate (written before the rewrite):**
  1. Numerical parity vs **CPU scalar backend** (rel-err ≤ 1e-3) on real layer shapes,
     via the `buffer_download` readback path, both models.
  2. End-to-end non-regression: `bench_session_throughput.c` decode checksum unchanged,
     tok/s ≥ prior best, both models.

## Phase 2 — Flash-attention (custom Metal kernel) + prefill

- **Custom kernel** (not MPSGraph). A clean SDPA core: scaled/normed/roped Q + K/V cache
  in → context out. QK-norm + RoPE stay as the existing pre-kernel steps.
- **Order:** prefill flash kernel **first**, **f16-KV path first**; drop in replacing
  `attention_rows` **only in the prefill path**, keep naive `attention_rows` for decode.
  f32-KV variant second.
- **Must support:** GQA/MQA, causal + sliding-window bounds, f16 & f32 KV. **No** QK
  soft-cap needed.
- **Up-front spike (~1 day):** prove a tiling scheme that fits **head_dim 512** in 32 KB
  threadgroup memory (tile over head_dim / shrink Q-row tile / KV-in-registers per
  simdgroup à la `flash_attn_ext`). If 512 can't be made efficient, it reshapes the kernel.
- **Gate:** parity vs CPU reference (≤ 1e-3) across **both** regimes — Gemma
  (windowed + MQA + hd 256/512) and solar (full causal + GQA8 + uniform hd); then prefill
  `pp` within target of `llama-bench` at the long-ctx point.
- **Flash-decoding (split-KV):** separate later item, scoped to **solar** long-ctx decode.
  Gemma decode stays weight-bound — do not over-invest.

## Phase 3 — Auto-adaptation (deferred, productizes proven wins)

- Only after Phases 1–2 prove which kernels win.
- (a) Static device table: query MTLGPUFamily + core count at load, select known-good
  variants per device class. Cheap, deterministic.
- (b) Runtime micro-benchmark autotune (cache to disk like `PACK_CACHE_DIR`) only if the
  static table proves too coarse for untested hardware.
- Detailed deferred-refactor notes: [[vulkan-simplify-deferred]] (Vulkan side mirrors this).
