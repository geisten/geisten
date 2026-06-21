/*
 * src/backends/cpu_x86/linear_q6k.c — cpu_x86 Q6_K M=1 path.
 *
 * Layer: BACKEND (cpu_x86).
 *
 * Predecode Q6_K → fp32 row-major (n_out × n_in) at model-load, then run
 * a single cblas_sgemv (via geist_sgemv) per decode token. Memory cost:
 * 1.5× the on-disk Q6_K size for the typical Gemma 4 tied lm_head — the
 * fp32 dequant inflation. Phase 2 will pack as BF16 instead.
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "linear_q6k.h"

#include "geist_gemm.h" /* geist_sgemv */
#include "heap.h"
#include "quant.h" /* dequant_q6_K_row, Q6_K_BLOCK_BYTES */

#include <geist_backend.h>

#include <stddef.h>
#include <stdint.h>

[[nodiscard]] enum geist_status cpu_x86_linear_q6k_resolve(struct geist_weight *w) {
    if (w == nullptr || w->raw == nullptr || w->n_in <= 0 || w->n_out <= 0) {
        return GEIST_E_INVALID_ARG;
    }
    const size_t n_in  = (size_t) w->n_in;
    const size_t n_out = (size_t) w->n_out;
    if (n_in % Q6_K_BLOCK_ELEMS != 0) {
        return GEIST_E_INVALID_ARG;
    }

    const size_t fp32_bytes = n_out * n_in * sizeof(float);
    float       *blob       = heap_alloc_aligned(fp32_bytes, OPTIMAL_ALIGNMENT);
    if (blob == nullptr) {
        return GEIST_E_OOM;
    }

    const size_t   q6k_row_bytes = (n_in / Q6_K_BLOCK_ELEMS) * Q6_K_BLOCK_BYTES;
    const uint8_t *q6k_raw       = (const uint8_t *) w->raw;
    for (size_t m = 0; m < n_out; m++) {
        dequant_q6_K_row(q6k_raw + m * q6k_row_bytes, blob + m * n_in, n_in);
    }

    w->aux_fp32  = blob;
    w->aux_n     = (int32_t) (n_out * n_in);
    w->flags    |= GEIST_W_AUX_HEAP_OWNED;
    w->linear_m1 = cpu_x86_linear_q6k_m1;
    return GEIST_OK;
}

void cpu_x86_linear_q6k_m1(const float               *x,
                           const struct geist_weight *w,
                           struct geist_backend      *be,
                           float                     *y) {
    (void) be;
    const size_t n_in  = (size_t) w->n_in;
    const size_t n_out = (size_t) w->n_out;
    /* A is [n_out, n_in] row-major; y = A * x, length n_out. */
    geist_sgemv(GEIST_OP_N,
                (int) n_out,
                (int) n_in,
                1.0f,
                w->aux_fp32,
                (int) n_in,
                x,
                1,
                0.0f,
                y,
                1);
}
