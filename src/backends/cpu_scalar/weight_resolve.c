/*
 * src/backends/cpu_scalar/weight_resolve.c — resolver for cpu_scalar (P2.e).
 *
 * Layer: BACKEND.
 *
 * cpu_scalar is the pure-C reference backend. After P2.e the legacy
 * v->linear() vtable slot is gone from geist_backend_vtbl; every
 * backend must install kernel pointers via resolve_weight. This
 * file gives cpu_scalar a (slow, correct) resolver that wraps the
 * existing dequant helpers from gguf_quant.c into pre-resolved
 * function pointers.
 *
 * Performance characteristics:
 *   - F32 dense: naive triple loop with double accumulator. ~10× slower
 *     than cpu_neon + cblas; intentional, this is the reference.
 *   - Q3_K / Q4_K / Q5_K / Q6_K / Q8_0 / IQ2_S / IQ3_S: dequant one
 *     weight row at a time into a heap row-buffer, naive dot. Same as
 *     the old cpu_scalar_linear_quant body, just exposed via the
 *     resolver pattern.
 *
 * No SIMD, no BLAS — that's what cpu_neon is for.
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "internal.h"

#include "gguf_quant.h"
#include "heap.h"

#include <geist.h>
#include <geist_backend.h>
#include <geist_weight.h>

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* Dispatch helper: dequant one row of weight `w` row index `j` into
 * `row` (which is `n_in` floats). Returns false on unsupported dtype. */
static bool dequant_one_row_for(const struct geist_weight *w, size_t j,
                                 float *row) {
    const uint8_t *base = (const uint8_t *) w->raw;
    const size_t n_in   = (size_t) w->n_in;
    switch ((enum geist_dtype) w->dtype) {
        case GEIST_DTYPE_Q3_K:
            dequant_q3_K_row(base + j * n_in / Q3_K_BLOCK_ELEMS * Q3_K_BLOCK_BYTES,
                              row, n_in); return true;
        case GEIST_DTYPE_Q4_K:
            dequant_q4_K_row(base + j * n_in / Q4_K_BLOCK_ELEMS * Q4_K_BLOCK_BYTES,
                              row, n_in); return true;
        case GEIST_DTYPE_Q5_K:
            dequant_q5_K_row(base + j * n_in / Q5_K_BLOCK_ELEMS * Q5_K_BLOCK_BYTES,
                              row, n_in); return true;
        case GEIST_DTYPE_Q6_K:
            dequant_q6_K_row(base + j * n_in / Q6_K_BLOCK_ELEMS * Q6_K_BLOCK_BYTES,
                              row, n_in); return true;
        case GEIST_DTYPE_Q8_0:
            dequant_q8_0_row(base + j * n_in / Q8_0_BLOCK_ELEMS * Q8_0_BLOCK_BYTES,
                              row, n_in); return true;
        case GEIST_DTYPE_IQ2_S:
            dequant_iq2_s_row(base + j * n_in / IQ2_S_BLOCK_ELEMS * IQ2_S_BLOCK_BYTES,
                               row, n_in); return true;
        case GEIST_DTYPE_IQ3_S:
            dequant_iq3_s_row(base + j * n_in / IQ3_S_BLOCK_ELEMS * IQ3_S_BLOCK_BYTES,
                               row, n_in); return true;
        case GEIST_DTYPE_TQ2_0:
            dequant_tq2_0_row(base + j * n_in / TQ2_0_BLOCK_ELEMS * TQ2_0_BLOCK_BYTES,
                               row, n_in); return true;
        case GEIST_DTYPE_Q4_0:
            dequant_q4_0_row(base + j * n_in / Q4_0_BLOCK_ELEMS * Q4_0_BLOCK_BYTES,
                               row, n_in); return true;
        default:
            return false;
    }
}

/* ---- F32 dense ---- */

static void cpu_scalar_w_f32_m1(const float *x, const struct geist_weight *w,
                                  struct geist_backend *be, float *y) {
    (void) be;
    const float *wp = (const float *) w->raw;
    const size_t n_in  = (size_t) w->n_in;
    const size_t n_out = (size_t) w->n_out;
    for (size_t j = 0; j < n_out; j++) {
        double acc = 0.0;
        const float *row = wp + j * n_in;
        for (size_t k = 0; k < n_in; k++) acc += (double) x[k] * (double) row[k];
        y[j] = (float) acc;
    }
}

static void cpu_scalar_w_f32_mN(const float *x, const struct geist_weight *w,
                                  size_t m, struct geist_backend *be, float *y) {
    (void) be;
    const float *wp = (const float *) w->raw;
    const size_t n_in  = (size_t) w->n_in;
    const size_t n_out = (size_t) w->n_out;
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n_out; j++) {
            double acc = 0.0;
            const float *row = wp + j * n_in;
            for (size_t k = 0; k < n_in; k++) {
                acc += (double) x[i * n_in + k] * (double) row[k];
            }
            y[i * n_out + j] = (float) acc;
        }
    }
}

/* ---- Quantized: per-row dequant + naive dot/matmul ---- */

static void cpu_scalar_w_quant_m1(const float *x, const struct geist_weight *w,
                                    struct geist_backend *be, float *y) {
    (void) be;
    const size_t n_in  = (size_t) w->n_in;
    const size_t n_out = (size_t) w->n_out;
    float *row = heap_alloc_aligned(n_in * sizeof(float), OPTIMAL_ALIGNMENT);
    if (row == nullptr) return;
    for (size_t j = 0; j < n_out; j++) {
        if (!dequant_one_row_for(w, j, row)) { y[j] = 0; continue; }
        double acc = 0.0;
        for (size_t k = 0; k < n_in; k++) acc += (double) x[k] * (double) row[k];
        y[j] = (float) acc;
    }
    safe_free((void **) &row);
}

static void cpu_scalar_w_quant_mN(const float *x, const struct geist_weight *w,
                                    size_t m, struct geist_backend *be, float *y) {
    (void) be;
    const size_t n_in  = (size_t) w->n_in;
    const size_t n_out = (size_t) w->n_out;
    float *row = heap_alloc_aligned(n_in * sizeof(float), OPTIMAL_ALIGNMENT);
    if (row == nullptr) return;
    for (size_t j = 0; j < n_out; j++) {
        if (!dequant_one_row_for(w, j, row)) {
            for (size_t i = 0; i < m; i++) y[i * n_out + j] = 0;
            continue;
        }
        for (size_t i = 0; i < m; i++) {
            double acc = 0.0;
            for (size_t k = 0; k < n_in; k++) {
                acc += (double) x[i * n_in + k] * (double) row[k];
            }
            y[i * n_out + j] = (float) acc;
        }
    }
    safe_free((void **) &row);
}

[[nodiscard]] enum geist_status
cpu_scalar_resolve_weight(struct geist_backend *be, struct geist_weight *w) {
    (void) be;
    if (w == nullptr || w->raw == nullptr || w->n_in <= 0 || w->n_out <= 0) {
        return GEIST_E_INVALID_ARG;
    }
    switch ((enum geist_dtype) w->dtype) {
        case GEIST_DTYPE_F32:
            w->linear_m1 = cpu_scalar_w_f32_m1;
            w->linear_mN = cpu_scalar_w_f32_mN;
            return GEIST_OK;
        case GEIST_DTYPE_Q4_0:
        case GEIST_DTYPE_Q3_K:
        case GEIST_DTYPE_Q4_K:
        case GEIST_DTYPE_Q5_K:
        case GEIST_DTYPE_Q6_K:
        case GEIST_DTYPE_Q8_0:
        case GEIST_DTYPE_IQ2_S:
        case GEIST_DTYPE_IQ3_S:
        case GEIST_DTYPE_TQ2_0:
            w->linear_m1 = cpu_scalar_w_quant_m1;
            w->linear_mN = cpu_scalar_w_quant_mN;
            return GEIST_OK;
        default:
            return GEIST_E_UNSUPPORTED;
    }
}
