/*
 * src/backends/cpu_x86/kernel_w4a8.h — W4A8 dot/GEMV inner kernel.
 *
 * Layer: BACKEND (cpu_x86, internal).
 *
 * --- Contract ----------------------------------------------------------------
 *
 * Inner kernel: W4A8 row-dot, designed to map cleanly onto VPDPBUSD
 * (u8 × s8 → s32) and onto the GGUF Q4_K_M weight layout.
 *
 *   for each block b in [0, n_blocks):
 *       d_b = sum over i in [0, 32) of u_w[b, i] * a[b, i]
 *   y = scale_x * sum_b ( w_scales[b] * d_b - w_offsets[b] * sum_a[b] )
 *
 * where:
 *   u_w[b, i] is the i-th UNSIGNED 4-bit weight value in block b (∈ [0, 15]),
 *             packed two-per-byte in `weights`:
 *               low nibble  (bits 0..3) → element 2k
 *               high nibble (bits 4..7) → element 2k+1
 *   a[b, i]   is acts[b * 32 + i], an int8 activation in [-127, 127]
 *             (pre-quantized once per row by the caller).
 *   w_scales[b] is the per-block dequant scale of the weights (fp32, finite).
 *   w_offsets[b] is the per-block additive offset (fp32, finite). Maps the
 *             Q4_K "min" term directly: y_decoded[i] = w_scale * u_w[i]
 *             - w_offset. For quants without an offset (e.g. Q8_0 → W4A8),
 *             pass all zeros; the resulting term drops out exactly.
 *   sum_a[b]  is sum over i in [0, 32) of a[b, i] (int32). Must be computed
 *             once per activation row by the caller — typically alongside
 *             the int8 quantization. Allocation-free, single pass.
 *   scale_x   is the per-row activation scale (fp32, finite).
 *
 * The hot path is allocation-free. The caller owns every buffer. Activations,
 * sum_a, and scale_x are computed once per row of the input matrix and
 * reused across every output row.
 *
 * --- Alignment ---------------------------------------------------------------
 *
 * Scalar reference: no alignment requirement.
 * AVX2 / AVX-512 variants: weights and acts SHOULD be 64-byte aligned for
 *   peak throughput, but kernels use unaligned loads so misaligned inputs
 *   are correct, only slower. w_scales / w_offsets / sum_a have no alignment
 *   requirement.
 *
 * --- Numerical contract ------------------------------------------------------
 *
 * Per-block d_b is computed as exact int32 (no overflow: |sum| ≤ 32·15·127 =
 * 60960 per block). The fp32 accumulation across blocks uses single rounding
 * per block. Cross-ISA differences are bounded by fp32 reorder ε.
 *
 * Win-criterion γ tolerance (see docs/LINUX_X86_SPEC.md §Quality gates):
 *   |y_isa - y_scalar| ≤ 1e-3 abs/rel for n_blocks ≤ 1024.
 *
 * --- Error model -------------------------------------------------------------
 *
 * Inner kernel is float and assumes a validated contract — the engine
 * validates at backend.linear() entry. The dispatcher init returns the
 * chosen ISA tier (informational, never SIGILL: cpuid-gated).
 */
#ifndef GEIST_INTERNAL_BACKEND_CPU_X86_KERNEL_W4A8_H
#define GEIST_INTERNAL_BACKEND_CPU_X86_KERNEL_W4A8_H

#ifndef GEIST_INTERNAL_BACKEND_LAYER
#error "cpu_x86/kernel_w4a8.h is internal to the backend layer."
#endif

#include <stddef.h>
#include <stdint.h>

/* ISA tiers the dispatcher can select. Ordered by preference. */
enum w4a8_isa {
    W4A8_ISA_SCALAR      = 0,
    W4A8_ISA_AVX2        = 1,
    W4A8_ISA_AVX512      = 2,
    W4A8_ISA_AVX512_VNNI = 3,
    W4A8_ISA_AVX512_BF16 = 4, /* alias of VNNI for now; reserved for Phase 2 */
};

/* Width of one block; not a runtime parameter — required by the layout. */
constexpr size_t W4A8_BLOCK_ELEMS         = 32;
constexpr size_t W4A8_BLOCK_BYTES_WEIGHTS = W4A8_BLOCK_ELEMS / 2;

/* Scalar reference. Always available, used as cross-ISA-consistency oracle. */
[[nodiscard]] float w4a8_dot_scalar(
        size_t        n_blocks,
        const uint8_t weights[static n_blocks * W4A8_BLOCK_BYTES_WEIGHTS],
        const float   w_scales[static n_blocks],
        const float   w_offsets[static n_blocks],
        const int8_t  acts[static n_blocks * W4A8_BLOCK_ELEMS],
        const int32_t sum_a_per_block[static n_blocks],
        float         scale_x);

/* Dispatched entry point. Picks the best variant for the current host at
 * init time (see w4a8_dispatcher_init). Output matches the scalar reference
 * within the numerical contract above. */
[[nodiscard]] float w4a8_dot(
        size_t        n_blocks,
        const uint8_t weights[static n_blocks * W4A8_BLOCK_BYTES_WEIGHTS],
        const float   w_scales[static n_blocks],
        const float   w_offsets[static n_blocks],
        const int8_t  acts[static n_blocks * W4A8_BLOCK_ELEMS],
        const int32_t sum_a_per_block[static n_blocks],
        float         scale_x);

/* Initialize the dispatcher. Idempotent, thread-safe (first-call wins).
 * Returns the ISA tier the dispatcher will use on this host.
 *
 * Override via env GEIST_FORCE_ISA = {scalar,avx2,avx512,avx512_vnni,
 * avx512_bf16}; an override that the host does not support is silently
 * clamped down to the best available tier. */
enum w4a8_isa w4a8_dispatcher_init(void);

/* Inspect the dispatcher choice (post-init). Returns SCALAR if uninitialized. */
enum w4a8_isa w4a8_dispatcher_current(void);

/* Human-readable name for an ISA tier. */
const char *w4a8_isa_name(enum w4a8_isa isa);

#endif /* GEIST_INTERNAL_BACKEND_CPU_X86_KERNEL_W4A8_H */
