/*
 * src/backends/cpu_x86/kernel_w8a8_scalar.c — W8A8 scalar reference.
 *
 * Pure C23. Correctness oracle for the AVX-512+VNNI variant and the
 * fallback on hosts without VPDPBUSD. Contract: kernel_w8a8.h.
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "kernel_w8a8.h"

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
