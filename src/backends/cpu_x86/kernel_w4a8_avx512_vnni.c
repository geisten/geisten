/*
 * src/backends/cpu_x86/kernel_w4a8_avx512_vnni.c — AVX-512 + AVX-512_VNNI
 * variant of the W4A8 dot kernel.
 *
 * Compiled with -mavx512f -mavx512bw -mavx512dq -mavx512vl -mavx512vnni; the
 * dispatcher (kernel_w4a8.c, baseline -march) only ever calls this on a host
 * whose runtime cpuid reports the corresponding feature bits, so there is no
 * SIGILL risk on AVX2-only or older CPUs.
 *
 * --- Kernel sketch ---------------------------------------------------------
 *
 * The unsigned-4-bit weights map directly onto VPDPBUSD's (u8, s8) signature
 * — no bias trick required. Inner loop per block:
 *   1. Load 16 packed weight bytes.
 *   2. Extract low+high nibbles as unsigned bytes in [0, 15].
 *   3. Interleave to get 32 unsigned bytes in original element order.
 *   4. Load 32 int8 activations.
 *   5. VPDPBUSD accumulates u_w × s_a into 8 int32 lanes.
 *   6. Horizontal-reduce to one int32 d_b.
 *   7. acc += w_scales[b] * d_b - w_offsets[b] * sum_a_per_block[b].
 *
 * Caller pre-computes sum_a_per_block once per activation row, so the
 * inner loop is pure VPDPBUSD + load/store. Multi-block fusion using the
 * full _mm512_dpbusd_epi32 width is a Phase 1a Step 3 optimization.
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "kernel_w4a8.h"

#include <immintrin.h>
#include <stdint.h>

[[nodiscard]] float w4a8_dot_avx512_vnni(
        size_t        n_blocks,
        const uint8_t weights[static n_blocks * W4A8_BLOCK_BYTES_WEIGHTS],
        const float   w_scales[static n_blocks],
        const float   w_offsets[static n_blocks],
        const int8_t  acts[static n_blocks * W4A8_BLOCK_ELEMS],
        const int32_t sum_a_per_block[static n_blocks],
        float         scale_x) {
    const __m256i lo_mask = _mm256_set1_epi8(0x0F);

    float acc = 0.0f;
    for (size_t b = 0; b < n_blocks; b++) {
        const uint8_t *w_block = weights + b * W4A8_BLOCK_BYTES_WEIGHTS;
        const int8_t  *a_block = acts + b * W4A8_BLOCK_ELEMS;

        /* Load 16 packed weight bytes into the low half of a 256-bit
         * register, then broadcast into both halves so lo/hi nibble
         * extraction runs on the full 256-bit width. */
        const __m128i packed_128 = _mm_loadu_si128((const __m128i *) w_block);
        const __m256i packed_256 = _mm256_set_m128i(packed_128, packed_128);

        /* lo_nibs = byte & 0x0F (unsigned 4-bit at even positions).
         * hi_nibs = (byte >> 4) & 0x0F (unsigned 4-bit at odd positions). */
        const __m256i lo_nibs = _mm256_and_si256(packed_256, lo_mask);
        const __m256i hi_nibs = _mm256_and_si256(_mm256_srli_epi16(packed_256, 4), lo_mask);

        /* Interleave so element k of u_w is the k-th nibble in original
         * order [lo0, hi0, lo1, hi1, ...]. _mm256_unpacklo/hi work
         * lane-locally; permute2x128 stitches the lanes together. */
        const __m256i nibs_lo_lane = _mm256_unpacklo_epi8(lo_nibs, hi_nibs);
        const __m256i nibs_hi_lane = _mm256_unpackhi_epi8(lo_nibs, hi_nibs);
        const __m256i u_w =
                _mm256_permute2x128_si256(nibs_lo_lane, nibs_hi_lane, 0x20);

        /* 32 signed-int8 activations. */
        const __m256i s_a = _mm256_loadu_si256((const __m256i *) a_block);

        /* VPDPBUSD: int32_lane[k] += sum_{j∈0..3} u_w[4k+j] * s_a[4k+j].
         * Eight int32 lanes total. */
        const __m256i dot32 =
                _mm256_dpbusd_epi32(_mm256_setzero_si256(), u_w, s_a);

        /* Horizontal reduce 8 int32 lanes → one int32. */
        const __m128i dot_lo = _mm256_castsi256_si128(dot32);
        const __m128i dot_hi = _mm256_extracti128_si256(dot32, 1);
        __m128i       dot4   = _mm_add_epi32(dot_lo, dot_hi);
        dot4                 = _mm_hadd_epi32(dot4, dot4);
        dot4                 = _mm_hadd_epi32(dot4, dot4);
        const int32_t d_b    = _mm_cvtsi128_si32(dot4);

        const float block_term =
                w_scales[b] * (float) d_b -
                w_offsets[b] * (float) sum_a_per_block[b];
        acc += block_term;
    }
    return scale_x * acc;
}
