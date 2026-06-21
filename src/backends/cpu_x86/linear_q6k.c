/*
 * src/backends/cpu_x86/linear_q6k.c — cpu_x86 Q6_K M=1 (decode) path.
 *
 * Layer: BACKEND (cpu_x86).
 *
 * Q6_K predecoded to W8A8 (1 byte / weight + per-16-elem-block scale +
 * offset). Decode hot path: per-row int8 act-quant + w8a8_gemv via
 * VPDPBUSD. Memory: ~1.25 bytes/weight vs ~0.81 on-disk Q6_K and ~4
 * bytes/weight in the previous fp32 path; the 4× shrink vs fp32 lifts
 * the bandwidth-bound Gemma 4 lm_head decode close to the W4A8 ceiling.
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "linear_q6k.h"

#include "backend_state.h"
#include "kernel_w4a8.h" /* w4a8_quantize_acts_row */
#include "kernel_w8a8.h"
#include "q6k_to_w8a8.h"

#include "heap.h"
#include "quant.h" /* Q6_K_BLOCK_ELEMS / Q6_K_BLOCK_BYTES */

#include <geist_backend.h>

#include <stddef.h>
#include <stdint.h>

/* SoA layout in the heap blob (allocated once per Q6_K weight at
 * resolve_weight): weights | w_scales | w_offsets. Sizes derive from n_in,
 * n_out, W8A8_BLOCK_ELEMS. The fp32 arrays follow the byte array; since
 * heap_alloc_aligned gives 64-byte alignment and the byte array size is a
 * multiple of W8A8_BLOCK_ELEMS (= 16), the float starts are 16-byte
 * aligned which is sufficient for fp32 loads. */
static inline size_t blocks_per_row(size_t n_in) {
    return n_in / W8A8_BLOCK_ELEMS;
}
static inline size_t weights_bytes_per_row(size_t n_in) {
    return blocks_per_row(n_in) * W8A8_BLOCK_ELEMS;
}

static void blob_pointers(const uint8_t *blob,
                          size_t         n_in,
                          size_t         n_out,
                          const uint8_t **weights_out,
                          const float  **scales_out,
                          const float  **offsets_out) {
    const size_t weights_bytes = n_out * weights_bytes_per_row(n_in);
    const size_t scales_count  = n_out * blocks_per_row(n_in);

    *weights_out = blob;
    *scales_out  = (const float *) (blob + weights_bytes);
    *offsets_out = *scales_out + scales_count;
}

[[nodiscard]] enum geist_status cpu_x86_linear_q6k_resolve(struct geist_weight *w) {
    if (w == nullptr || w->raw == nullptr || w->n_in <= 0 || w->n_out <= 0) {
        return GEIST_E_INVALID_ARG;
    }
    const size_t n_in  = (size_t) w->n_in;
    const size_t n_out = (size_t) w->n_out;
    if (n_in % Q6_K_BLOCK_ELEMS != 0) {
        return GEIST_E_INVALID_ARG;
    }

    const size_t weights_total = n_out * weights_bytes_per_row(n_in);
    const size_t scales_total  = n_out * blocks_per_row(n_in) * sizeof(float);
    const size_t blob_bytes    = weights_total + 2 * scales_total;

    uint8_t *blob = heap_alloc_aligned(blob_bytes, OPTIMAL_ALIGNMENT);
    if (blob == nullptr) {
        return GEIST_E_OOM;
    }
    const uint8_t *bw;
    const float   *bs;
    const float   *bo;
    blob_pointers(blob, n_in, n_out, &bw, &bs, &bo);
    uint8_t *blob_w = (uint8_t *) bw;
    float   *blob_s = (float *) bs;
    float   *blob_o = (float *) bo;

    const size_t q6k_row_bytes = (n_in / Q6_K_BLOCK_ELEMS) * Q6_K_BLOCK_BYTES;
    const size_t w_row_bytes   = weights_bytes_per_row(n_in);
    const size_t s_row_count   = blocks_per_row(n_in);
    const uint8_t *q6k_raw     = (const uint8_t *) w->raw;
    for (size_t m = 0; m < n_out; m++) {
        q6k_to_w8a8_row(n_in,
                        q6k_raw + m * q6k_row_bytes,
                        blob_w + m * w_row_bytes,
                        blob_s + m * s_row_count,
                        blob_o + m * s_row_count);
    }

    w->aux_fp32  = (const float *) blob;
    w->aux_n     = (int32_t) blob_bytes;
    w->flags    |= GEIST_W_AUX_HEAP_OWNED | GEIST_W_AUX_BACKEND_REPACK;
    w->linear_m1 = cpu_x86_linear_q6k_m1;
    return GEIST_OK;
}

/* The activation pipeline (int8 quant + per-block sum_a) is shared with
 * W4A8 — block sizes differ (W4A8: 32, W8A8: 16) but the activation
 * buffer is a row of int8 values and sum_a is summed in W8A8_BLOCK_ELEMS
 * chunks here. We re-quantize and re-sum locally; w4a8_quantize_acts_row
 * sums in W4A8_BLOCK_ELEMS chunks (32), which is wrong for W8A8, so do
 * a small inline sum at W8A8_BLOCK_ELEMS granularity. */
static void quantize_acts_w8a8(size_t n_in,
                                struct cpu_x86_state *st,
                                const float          *x,
                                float                *scale_x_out) {
    /* Use quantize_x_int8_sym directly via the W4A8 wrapper, which
     * stores int8 acts into st->acts_scratch. Discard the W4A8 sum_a
     * (block_elems=32) — recompute at W8A8 granularity below. */
    *scale_x_out = w4a8_quantize_acts_row(
            n_in, x, st->acts_scratch, st->sum_a_scratch);

    /* Re-sum acts at 16-element granularity for W8A8. Overwrites the
     * 32-elem sum_a since they're not used by Q6_K decode. */
    const size_t n_blocks = n_in / W8A8_BLOCK_ELEMS;
    const int8_t *acts    = st->acts_scratch;
    for (size_t b = 0; b < n_blocks; b++) {
        int32_t s = 0;
        for (size_t i = 0; i < W8A8_BLOCK_ELEMS; i++) {
            s += (int32_t) acts[b * W8A8_BLOCK_ELEMS + i];
        }
        st->sum_a_scratch[b] = s;
    }
}

void cpu_x86_linear_q6k_m1(const float               *x,
                           const struct geist_weight *w,
                           struct geist_backend      *be,
                           float                     *y) {
    struct cpu_x86_state *st               = (struct cpu_x86_state *) be->state;
    const size_t          n_in             = (size_t) w->n_in;
    const size_t          n_out            = (size_t) w->n_out;
    const size_t          n_blocks_per_row = blocks_per_row(n_in);

    const uint8_t *weights;
    const float   *w_scales;
    const float   *w_offsets;
    blob_pointers((const uint8_t *) w->aux_fp32, n_in, n_out,
                  &weights, &w_scales, &w_offsets);

    float scale_x;
    quantize_acts_w8a8(n_in, st, x, &scale_x);

    w8a8_gemv(n_out, n_blocks_per_row,
              weights, w_scales, w_offsets,
              st->acts_scratch, st->sum_a_scratch, scale_x, y);
}
