/*
 * src/backends/cpu_x86/kernel_w8a8_avx512_vnni.c — AVX-512+VNNI W8A8 dot.
 *
 * Compiled with -mavx512f -mavx512bw -mavx512dq -mavx512vl -mavx512vnni;
 * the dispatcher only calls this when cpuid confirms VNNI support.
 *
 * Two 16-element blocks per 256-bit VPDPBUSD (32 bytes weights + 32
 * acts). 8 int32 lanes; lanes 0-3 = block 0, lanes 4-7 = block 1.
 * Q6_K's 16-element sub-block granularity guarantees n_blocks is even
 * (16 sub-blocks per super-block × n_super); a 16-byte single-block
 * tail handles other callers.
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "kernel_w8a8.h"

#include <immintrin.h>
#include <stdint.h>

[[nodiscard]] float w8a8_dot_avx512_vnni(
        size_t        n_blocks,
        const uint8_t weights[static n_blocks * W8A8_BLOCK_ELEMS],
        const float   w_scales[static n_blocks],
        const float   w_offsets[static n_blocks],
        const int8_t  acts[static n_blocks * W8A8_BLOCK_ELEMS],
        const int32_t sum_a_per_block[static n_blocks],
        float         scale_x) {
    float  acc = 0.0f;
    size_t b   = 0;

    /* 2 blocks (32 elements) per VPDPBUSD. */
    for (; b + 2 <= n_blocks; b += 2) {
        const __m256i u_w = _mm256_loadu_si256(
                (const __m256i *) (weights + b * W8A8_BLOCK_ELEMS));
        const __m256i s_a = _mm256_loadu_si256(
                (const __m256i *) (acts + b * W8A8_BLOCK_ELEMS));
        const __m256i dot8 =
                _mm256_dpbusd_epi32(_mm256_setzero_si256(), u_w, s_a);

        /* lanes 0-3: block 0; lanes 4-7: block 1. Reduce each half. */
        const __m128i d_lo = _mm256_castsi256_si128(dot8);
        const __m128i d_hi = _mm256_extracti128_si256(dot8, 1);
        __m128i       sum0 = _mm_hadd_epi32(d_lo, d_lo);
        sum0               = _mm_hadd_epi32(sum0, sum0);
        __m128i       sum1 = _mm_hadd_epi32(d_hi, d_hi);
        sum1               = _mm_hadd_epi32(sum1, sum1);
        const int32_t d_b0 = _mm_cvtsi128_si32(sum0);
        const int32_t d_b1 = _mm_cvtsi128_si32(sum1);

        acc += w_scales[b]     * (float) d_b0 -
               w_offsets[b]    * (float) sum_a_per_block[b];
        acc += w_scales[b + 1] * (float) d_b1 -
               w_offsets[b + 1] * (float) sum_a_per_block[b + 1];
    }

    /* Tail: one 16-elem block. */
    if (b < n_blocks) {
        const __m128i u_w128 =
                _mm_loadu_si128((const __m128i *) (weights + b * W8A8_BLOCK_ELEMS));
        const __m128i s_a128 =
                _mm_loadu_si128((const __m128i *) (acts + b * W8A8_BLOCK_ELEMS));
        const __m128i dot4 =
                _mm_dpbusd_epi32(_mm_setzero_si128(), u_w128, s_a128);
        __m128i       s = _mm_hadd_epi32(dot4, dot4);
        s               = _mm_hadd_epi32(s, s);
        const int32_t d_b = _mm_cvtsi128_si32(s);
        acc += w_scales[b] * (float) d_b -
               w_offsets[b] * (float) sum_a_per_block[b];
    }

    return scale_x * acc;
}
