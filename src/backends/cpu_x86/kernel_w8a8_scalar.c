/*
 * src/backends/cpu_x86/kernel_w8a8_scalar.c — W8A8 scalar reference.
 *
 * Pure C23. Correctness oracle for the AVX-512+VNNI variant and the
 * fallback on hosts without VPDPBUSD. Contract: kernel_w8a8.h.
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "kernel_w8a8.h"

#include <stddef.h>
#include <stdint.h>

[[nodiscard]] float w8a8_dot_scalar(
        size_t        n_blocks,
        const uint8_t weights[static n_blocks * W8A8_BLOCK_ELEMS],
        const float   w_scales[static n_blocks],
        const float   w_offsets[static n_blocks],
        const int8_t  acts[static n_blocks * W8A8_BLOCK_ELEMS],
        const int32_t sum_a_per_block[static n_blocks],
        float         scale_x) {
    float acc = 0.0f;
    for (size_t b = 0; b < n_blocks; b++) {
        const uint8_t *w_block = weights + b * W8A8_BLOCK_ELEMS;
        const int8_t  *a_block = acts + b * W8A8_BLOCK_ELEMS;
        int32_t        d_b     = 0;
        for (size_t i = 0; i < W8A8_BLOCK_ELEMS; i++) {
            d_b += (int32_t) w_block[i] * (int32_t) a_block[i];
        }
        const float block_term =
                w_scales[b] * (float) d_b -
                w_offsets[b] * (float) sum_a_per_block[b];
        acc += block_term;
    }
    return scale_x * acc;
}

void w8x8_repack(
        size_t        n_out,
        size_t        n_in,
        const uint8_t weights[static n_out * n_in],
        const float   w_scales[static n_out * (n_in / W8A8_BLOCK_ELEMS)],
        const float   w_offsets[static n_out * (n_in / W8A8_BLOCK_ELEMS)],
        uint8_t       qs_out[static n_out * n_in],
        float         scales_out[static n_out * (n_in / W8A8_BLOCK_ELEMS)],
        float         offsets_out[static n_out * (n_in / W8A8_BLOCK_ELEMS)]) {
    const size_t NB     = n_in / W8A8_BLOCK_ELEMS;
    const size_t n_grp  = n_in / 4;             /* 4-element stripes */
    const size_t NG     = n_out / W8X8_NROWS;
    const size_t qs_per_group = n_in * W8X8_NROWS;
    const size_t sc_per_group = NB * W8X8_NROWS;

    for (size_t g = 0; g < NG; g++) {
        uint8_t *qs_g = qs_out + g * qs_per_group;
        float   *sc_g = scales_out + g * sc_per_group;
        float   *of_g = offsets_out + g * sc_per_group;
        for (size_t r = 0; r < W8X8_NROWS; r++) {
            const size_t   row = g * W8X8_NROWS + r;
            const uint8_t *wr  = weights + row * n_in;
            for (size_t grp = 0; grp < n_grp; grp++) {
                for (size_t e = 0; e < 4; e++) {
                    qs_g[grp * (4 * W8X8_NROWS) + r * 4 + e] = wr[grp * 4 + e];
                }
            }
            const float *sr = w_scales + row * NB;
            const float *or_ = w_offsets + row * NB;
            for (size_t b = 0; b < NB; b++) {
                sc_g[b * W8X8_NROWS + r] = sr[b];
                of_g[b * W8X8_NROWS + r] = or_[b];
            }
        }
    }
}
