/*
 * test_q4kx8_gemm_unit — end-to-end Q4_Kx8 GEMM correctness.
 *
 * Builds synthetic Q4_K weights, repacks to Q4_Kx8, quantizes fp32 acts
 * to Q8_Kx4, runs the scalar GEMM, and compares to the reference path:
 * dequant_q4_K_row each weight row + fp32 sgemm against fp32 acts.
 *
 * Tolerance: dominated by int8 act quant (per-row, ~127 levels).
 *
 * Deterministic; runs in <500 ms.
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "../src/backends/cpu_x86/kernel_q4kx8_gemm.h"
#include "../src/backends/cpu_x86/q4k_to_q4kx8.h"
#include "../src/backends/cpu_x86/q8_kx4.h"
#include "quant.h"
#include "quant_blocks.h"
#include "test_helpers.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint16_t fp32_to_fp16(float f) {
    union {
        float    f;
        uint32_t u;
    } bits  = {.f = f};
    uint32_t x      = bits.u;
    uint32_t sign   = (x >> 31) & 0x1u;
    int32_t  exp32  = (int32_t) ((x >> 23) & 0xFFu) - 127;
    uint32_t mant32 = x & 0x7FFFFFu;
    uint16_t out_sign = (uint16_t) (sign << 15);
    if (exp32 == 128) {
        return (uint16_t) (out_sign | 0x7C00u | (mant32 != 0));
    }
    if (exp32 < -14) {
        return out_sign;
    }
    if (exp32 > 15) {
        return (uint16_t) (out_sign | 0x7C00u);
    }
    uint16_t exp16     = (uint16_t) ((exp32 + 15) & 0x1Fu);
    uint32_t mant16    = mant32 >> 13;
    uint32_t round_bit = (mant32 >> 12) & 1u;
    uint32_t sticky    = mant32 & 0xFFFu;
    if (round_bit != 0 && (sticky != 0 || (mant16 & 1u) != 0)) {
        mant16 += 1;
        if (mant16 == 0x400u) {
            mant16 = 0;
            exp16  = (uint16_t) (exp16 + 1u);
        }
    }
    return (uint16_t) (out_sign | (uint16_t) (exp16 << 10) | (uint16_t) mant16);
}

static uint32_t prng_next(uint32_t *state) {
    uint32_t z = (*state += 0x9E3779B9u);
    z          = (z ^ (z >> 16)) * 0x85EBCA6Bu;
    z          = (z ^ (z >> 13)) * 0xC2B2AE35u;
    return z ^ (z >> 16);
}

static float prng_uniform_pm1(uint32_t bits) {
    return 2.0f * (float) (bits & 0xFFFFFFu) / (float) (1u << 24) - 1.0f;
}

static void set_scale_min_k4(int j, uint8_t scales[12], uint8_t d_in, uint8_t m_in) {
    if (j < 4) {
        scales[j]     = (uint8_t) ((scales[j] & 0xC0u) | (d_in & 0x3Fu));
        scales[j + 4] = (uint8_t) ((scales[j + 4] & 0xC0u) | (m_in & 0x3Fu));
    } else {
        const uint8_t d_lo4 = d_in & 0x0Fu;
        const uint8_t d_hi2 = (d_in >> 4) & 0x03u;
        const uint8_t m_lo4 = m_in & 0x0Fu;
        const uint8_t m_hi2 = (m_in >> 4) & 0x03u;
        scales[j + 4]       = (uint8_t) (d_lo4 | (m_lo4 << 4));
        scales[j - 4]       = (uint8_t) ((scales[j - 4] & 0x3Fu) | (d_hi2 << 6));
        scales[j]           = (uint8_t) ((scales[j] & 0x3Fu) | (m_hi2 << 6));
    }
}

static int scenario_endtoend(void) {
    /* 8 weight rows × 1 super-block (K=256). M=4 activation rows. */
    constexpr size_t N      = 8;
    constexpr size_t M      = 4;
    constexpr size_t K      = 256;
    constexpr size_t N_SUPER = K / 256;

    /* Synthesize Q4_K weights. */
    struct block_q4_K_t W_q4k[N * N_SUPER];
    uint32_t            s = 0xFEEDFACEu;
    for (size_t r = 0; r < N * N_SUPER; r++) {
        memset(&W_q4k[r], 0, sizeof(W_q4k[r]));
        const float df    = 0.002f + 0.005f * ((prng_next(&s) & 0xFFFFu) / 65536.0f);
        const float mf    = 0.001f + 0.003f * ((prng_next(&s) & 0xFFFFu) / 65536.0f);
        W_q4k[r].d        = fp32_to_fp16(df);
        W_q4k[r].dmin     = fp32_to_fp16(mf);
        for (int sb = 0; sb < 8; sb++) {
            const uint8_t sc = (uint8_t) (prng_next(&s) & 0x3Fu);
            const uint8_t mn = (uint8_t) (prng_next(&s) & 0x3Fu);
            set_scale_min_k4(sb, W_q4k[r].scales, sc, mn);
        }
        for (size_t k = 0; k < 128; k++) {
            W_q4k[r].qs[k] = (uint8_t) (prng_next(&s) & 0xFFu);
        }
    }

    /* Repack to Q4_Kx8. */
    struct block_q4_Kx8 W_q4kx8[N_SUPER];
    q4k_to_q4kx8_octet(N_SUPER, (const uint8_t *) W_q4k, W_q4kx8);

    /* Synthesize fp32 activations. */
    float X_fp32[M * K];
    for (size_t i = 0; i < M * K; i++) {
        X_fp32[i] = 1.0f * prng_uniform_pm1(prng_next(&s));
    }

    /* Quantize to Q8_Kx4. */
    struct block_q8_Kx4 X_q8kx4[N_SUPER];
    quantize_q8_Kx4(K, X_fp32, X_q8kx4);

    /* Run the Q4_Kx8 GEMM. */
    float Y_test[M * N];
    q4kx8_gemm_scalar(M, N, K, X_q8kx4, W_q4kx8, Y_test);

    /* Reference: dequant each weight row → fp32 sgemm against fp32 acts. */
    float W_fp32[N * K];
    for (size_t r = 0; r < N; r++) {
        dequant_q4_K_row(&W_q4k[r * N_SUPER], W_fp32 + r * K, K);
    }
    float Y_ref[M * N];
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            double acc = 0.0;
            for (size_t k = 0; k < K; k++) {
                acc += (double) X_fp32[i * K + k] * (double) W_fp32[j * K + k];
            }
            Y_ref[i * N + j] = (float) acc;
        }
    }

    /* Compare. RMS tolerance ~ scale_x * sqrt(K) * |w_rms|. */
    double rms_sq = 0.0;
    for (size_t i = 0; i < M * N; i++) {
        rms_sq += (double) Y_ref[i] * (double) Y_ref[i];
    }
    const float rms = (float) sqrt(rms_sq / (M * N));
    const float tol = 0.10f + 0.20f * rms;

    int   fails    = 0;
    float max_diff = 0.0f;
    for (size_t i = 0; i < M * N; i++) {
        const float d = fabsf(Y_test[i] - Y_ref[i]);
        if (d > tol) {
            fprintf(stderr, "i=%zu ref=%g test=%g diff=%g tol=%g\n",
                    i, (double) Y_ref[i], (double) Y_test[i], (double) d, (double) tol);
            fails++;
            if (fails > 4) {
                break;
            }
        }
        if (d > max_diff) {
            max_diff = d;
        }
    }
    fprintf(stdout, "[q4kx8_gemm] max |Δ| = %g, rms = %g, tol = %g\n",
            (double) max_diff, (double) rms, (double) tol);
    return fails;
}

int main(void) {
    int fails = 0;
    if (scenario_endtoend() != 0) {
        fputs("scenario_endtoend FAILED\n", stderr);
        fails++;
    }
    return fails == 0 ? GEIST_TEST_PASS : GEIST_TEST_FAIL;
}
