/*
 * src/backends/cpu_x86/linear_q4k.c — cpu_x86 Q4_K M=1 (decode) wiring.
 *
 * Layer: BACKEND (cpu_x86).
 *
 * Per Q4_K weight:
 *   1. Allocate one heap-aligned blob for the W4A8 SoA: packed nibbles
 *      (n_in/2 bytes), per-block scales (n_in/32 fp32), per-block offsets
 *      (n_in/32 fp32), in that order; each row contributes n_blocks of
 *      each. The blob is owned by the weight (GEIST_W_AUX_HEAP_OWNED) so
 *      the engine frees it at model destroy.
 *   2. Predecode via q4k_to_w4a8_row, row-major over n_out rows.
 *   3. Grow the per-backend activation scratch (int8 acts + sum_a) to
 *      cover n_in if needed — at model-load, never in the hot path.
 *   4. Install cpu_x86_linear_q4k_m1 into w->linear_m1.
 *
 * The hot-path kernel reconstructs the SoA pointers from w->aux_fp32 +
 * w->n_in + w->n_out arithmetic; no per-call allocation, no branching.
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "linear_q4k.h"

#include "backend_state.h"
#include "kernel_w4a8.h"
#include "q4k_to_w4a8.h"

#include "heap.h"
#include "quant.h" /* Q4_K_BLOCK_ELEMS / Q4_K_BLOCK_BYTES */

#include <geist_backend.h>

#include <stdalign.h>
#include <stddef.h>
#include <stdint.h>

/* Layout sizes for one weight (row-major SoA). Per-row blocks = n_in/32. */
static inline size_t weights_bytes_per_row(size_t n_in) {
    return (n_in / W4A8_BLOCK_ELEMS) * W4A8_BLOCK_BYTES_WEIGHTS;
}
static inline size_t scales_count_per_row(size_t n_in) {
    return n_in / W4A8_BLOCK_ELEMS;
}

/* SoA pointer reconstruction. The blob layout is:
 *   [weights : n_out * weights_bytes_per_row(n_in)]
 *   [w_scales: n_out * scales_count_per_row(n_in) fp32]
 *   [w_offsets:n_out * scales_count_per_row(n_in) fp32]
 * The fp32 arrays are placed after the byte array, aligned to alignof(float)
 * by construction (heap_alloc_aligned is 64-byte-aligned, weights size is
 * a multiple of 32 — so the fp32 start is 32+-aligned). */
static void blob_pointers(const uint8_t *blob,
                          size_t         n_in,
                          size_t         n_out,
                          const uint8_t **weights_out,
                          const float  **scales_out,
                          const float  **offsets_out) {
    const size_t weights_bytes = n_out * weights_bytes_per_row(n_in);
    const size_t scales_count  = n_out * scales_count_per_row(n_in);

    *weights_out = blob;
    *scales_out  = (const float *) (blob + weights_bytes);
    *offsets_out = *scales_out + scales_count;
}

/* Grow the backend's activation scratch buffers to cover at least n_in
 * elements. Called only at resolve_weight time. Returns OK or E_OOM. */
static enum geist_status grow_scratch(struct cpu_x86_state *st, size_t n_in) {
    if (n_in <= st->scratch_cap) {
        return GEIST_OK;
    }
    int8_t *new_acts = heap_alloc_aligned(n_in * sizeof(int8_t), OPTIMAL_ALIGNMENT);
    if (new_acts == nullptr) {
        return GEIST_E_OOM;
    }
    const size_t n_blocks = n_in / W4A8_BLOCK_ELEMS;
    int32_t *new_sum_a    = heap_alloc_aligned(n_blocks * sizeof(int32_t), OPTIMAL_ALIGNMENT);
    if (new_sum_a == nullptr) {
        safe_free((void **) &new_acts);
        return GEIST_E_OOM;
    }
    safe_free((void **) &st->acts_scratch);
    safe_free((void **) &st->sum_a_scratch);
    st->acts_scratch  = new_acts;
    st->sum_a_scratch = new_sum_a;
    st->scratch_cap   = n_in;
    return GEIST_OK;
}

[[nodiscard]] enum geist_status
cpu_x86_linear_q4k_resolve(struct cpu_x86_state *st, struct geist_weight *w) {
    if (st == nullptr || w == nullptr || w->n_in <= 0 || w->n_out <= 0) {
        return GEIST_E_INVALID_ARG;
    }
    const size_t n_in  = (size_t) w->n_in;
    const size_t n_out = (size_t) w->n_out;
    if (n_in % Q4_K_BLOCK_ELEMS != 0) {
        return GEIST_E_INVALID_ARG;
    }

    /* SoA blob: weights bytes + n_out * 2 * fp32 per W4A8 block. */
    const size_t weights_total_bytes = n_out * weights_bytes_per_row(n_in);
    const size_t scales_total_bytes  = n_out * scales_count_per_row(n_in) * sizeof(float);
    const size_t blob_bytes          = weights_total_bytes + 2 * scales_total_bytes;

    uint8_t *blob = heap_alloc_aligned(blob_bytes, OPTIMAL_ALIGNMENT);
    if (blob == nullptr) {
        return GEIST_E_OOM;
    }
    const uint8_t *blob_w_const;
    const float   *blob_s_const;
    const float   *blob_o_const;
    blob_pointers(blob, n_in, n_out, &blob_w_const, &blob_s_const, &blob_o_const);
    /* Cast away const so we can write the freshly-allocated blob. */
    uint8_t *blob_w = (uint8_t *) blob_w_const;
    float   *blob_s = (float *) blob_s_const;
    float   *blob_o = (float *) blob_o_const;

    /* Predecode row-major. Q4_K row stride is (n_in / Q4_K_BLOCK_ELEMS)
     * super-blocks, each Q4_K_BLOCK_BYTES bytes. */
    const size_t q4k_row_bytes = (n_in / Q4_K_BLOCK_ELEMS) * Q4_K_BLOCK_BYTES;
    const size_t w_row_bytes   = weights_bytes_per_row(n_in);
    const size_t s_row_count   = scales_count_per_row(n_in);
    const uint8_t *q4k_raw     = (const uint8_t *) w->raw;
    for (size_t m = 0; m < n_out; m++) {
        q4k_to_w4a8_row(n_in,
                        q4k_raw + m * q4k_row_bytes,
                        blob_w + m * w_row_bytes,
                        blob_s + m * s_row_count,
                        blob_o + m * s_row_count);
    }

    /* Grow scratch to cover this n_in. */
    enum geist_status scratch_st = grow_scratch(st, n_in);
    if (scratch_st != GEIST_OK) {
        safe_free((void **) &blob);
        return scratch_st;
    }

    /* aux_fp32 reinterpreted as the blob pointer; engine frees it on
     * model destroy via heap_free / safe_free (GEIST_W_AUX_HEAP_OWNED). */
    w->aux_fp32  = (const float *) blob;
    w->aux_n     = (int32_t) blob_bytes;
    w->flags    |= GEIST_W_AUX_HEAP_OWNED | GEIST_W_AUX_BACKEND_REPACK;
    w->linear_m1 = cpu_x86_linear_q4k_m1;
    w->linear_mN = cpu_x86_linear_q4k_mN;
    return GEIST_OK;
}

void cpu_x86_linear_q4k_m1(const float               *x,
                           const struct geist_weight *w,
                           struct geist_backend      *be,
                           float                     *y) {
    struct cpu_x86_state *st    = (struct cpu_x86_state *) be->state;
    const size_t          n_in  = (size_t) w->n_in;
    const size_t          n_out = (size_t) w->n_out;
    const size_t          n_blocks_per_row = n_in / W4A8_BLOCK_ELEMS;

    const uint8_t *weights;
    const float   *w_scales;
    const float   *w_offsets;
    blob_pointers((const uint8_t *) w->aux_fp32, n_in, n_out,
                  &weights, &w_scales, &w_offsets);

    /* Per-row activation quantization → int8 acts + per-block sum_a. */
    const float scale_x = w4a8_quantize_acts_row(
            n_in, x, st->acts_scratch, st->sum_a_scratch);

    /* Multi-row GEMV. OMP-parallel internally. */
    w4a8_gemv(n_out, n_blocks_per_row,
              weights, w_scales, w_offsets,
              st->acts_scratch, st->sum_a_scratch, scale_x, y);
}

void cpu_x86_linear_q4k_mN(const float               *x,
                           const struct geist_weight *w,
                           size_t                     m,
                           struct geist_backend      *be,
                           float                     *y) {
    /* ponytail: M-times M=1. Phase 1b later fuses the m-tile so each
     * weight block is streamed once per m tokens; today this still
     * beats cpu_scalar's per-row Q4_K dequant. */
    const size_t n_in  = (size_t) w->n_in;
    const size_t n_out = (size_t) w->n_out;
    for (size_t row = 0; row < m; row++) {
        cpu_x86_linear_q4k_m1(x + row * n_in, w, be, y + row * n_out);
    }
}
