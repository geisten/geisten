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
#include "kernel_w8a8.h" /* sum_a sized for W8A8 to also cover Q6_K */
#include "q4k_to_w4a8.h"

#include "heap.h"
#include "quant.h" /* Q4_K_BLOCK_ELEMS / Q4_K_BLOCK_BYTES */

#include <geist_backend.h>

#include <immintrin.h>
#include <stdalign.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

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

/* Grow the per-call mtile scratch to cover at least (m, n_in). Called
 * from linear_mN on first use and whenever m or n_in exceeds the cached
 * capacity. Not in the inner hot path. */
static enum geist_status grow_mtile(struct cpu_x86_state *st, size_t m, size_t n_in) {
    if (m <= st->mtile_m_cap && n_in <= st->mtile_n_cap) {
        return GEIST_OK;
    }
    const size_t n_blocks   = n_in / W4A8_BLOCK_ELEMS;
    const size_t acts_bytes = m * n_in * sizeof(int8_t);
    const size_t suma_bytes = m * n_blocks * sizeof(int32_t);
    const size_t sx_bytes   = m * sizeof(float);

    int8_t  *new_acts = heap_alloc_aligned(acts_bytes, OPTIMAL_ALIGNMENT);
    if (new_acts == nullptr) {
        return GEIST_E_OOM;
    }
    int32_t *new_suma = heap_alloc_aligned(suma_bytes, OPTIMAL_ALIGNMENT);
    if (new_suma == nullptr) {
        safe_free((void **) &new_acts);
        return GEIST_E_OOM;
    }
    float   *new_sx   = heap_alloc_aligned(sx_bytes, OPTIMAL_ALIGNMENT);
    if (new_sx == nullptr) {
        safe_free((void **) &new_acts);
        safe_free((void **) &new_suma);
        return GEIST_E_OOM;
    }
    safe_free((void **) &st->acts_mtile);
    safe_free((void **) &st->sum_a_mtile);
    safe_free((void **) &st->scale_x_mtile);
    st->acts_mtile    = new_acts;
    st->sum_a_mtile   = new_suma;
    st->scale_x_mtile = new_sx;
    st->mtile_m_cap   = m;
    st->mtile_n_cap   = n_in;
    return GEIST_OK;
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
    /* Size sum_a for the SMALLEST block granularity — W8A8 (16) — so the
     * buffer covers both Q4_K (W4A8, 32-elem blocks) and Q6_K (W8A8) callers
     * sharing this scratch. */
    const size_t n_blocks = n_in / W8A8_BLOCK_ELEMS;
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

/* These helpers + the tiled mN below use AVX-512+VNNI intrinsics. The
 * `target` attribute tells gcc to allow them in this TU even though
 * the file is compiled at baseline -march=x86-64-v3; the dispatcher
 * only calls cpu_x86_linear_q4k_mN on hosts whose cpuid confirms VNNI
 * support, so there is no SIGILL risk on AVX2-only CPUs. */
#define VNNI_TARGET "avx2,avx512f,avx512bw,avx512dq,avx512vl,avx512vnni"

/* Horizontal sum of 8 int32 lanes → one int32. */
__attribute__((target(VNNI_TARGET)))
static inline int32_t hsum_i32_avx2(__m256i v) {
    const __m128i lo = _mm256_castsi256_si128(v);
    const __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i       s  = _mm_add_epi32(lo, hi);
    s                = _mm_hadd_epi32(s, s);
    s                = _mm_hadd_epi32(s, s);
    return _mm_cvtsi128_si32(s);
}

/* Unpack 16 packed Q4_K bytes (= one 32-element block) into 32 unsigned
 * int8 values in element order, returned in a 256-bit register. The
 * caller will feed the result to VPDPBUSD against int8 activations. */
__attribute__((target(VNNI_TARGET)))
static inline __m256i unpack_w4a8_block_to_u8(const uint8_t weight_packed[16]) {
    const __m128i packed_128 = _mm_loadu_si128((const __m128i *) weight_packed);
    const __m256i packed_256 = _mm256_set_m128i(packed_128, packed_128);
    const __m256i lo_mask    = _mm256_set1_epi8(0x0F);
    const __m256i lo_nibs    = _mm256_and_si256(packed_256, lo_mask);
    const __m256i hi_nibs    = _mm256_and_si256(_mm256_srli_epi16(packed_256, 4), lo_mask);
    const __m256i nibs_lo    = _mm256_unpacklo_epi8(lo_nibs, hi_nibs);
    const __m256i nibs_hi    = _mm256_unpackhi_epi8(lo_nibs, hi_nibs);
    return _mm256_permute2x128_si256(nibs_lo, nibs_hi, 0x20);
}

/* M-tile width for the tiled prefill kernel. 4 output rows held in
 * registers across the K-block sweep — one VPDPBUSD per (i, j, b) tuple
 * with per-block scale/offset applied immediately so the kernel never
 * needs to spill the int32 partial sums. */
constexpr size_t Q4K_MTILE      = 4;
/* Max m supported by the stack-resident accumulator tile. m > this
 * falls back to the per-row M=1 loop (still correct, just slower). */
constexpr size_t Q4K_MTILE_MMAX = 2048;

/* Phase 1b Step 3 (c): tiled int8 GEMM for Q4_K prefill (M>1).
 *
 * Outer OMP parallelization over row-tiles. Per tile, hold M_TILE=4
 * fp32 accumulators for each of m output cells (in a stack-resident
 * acc[]), then sweep all K-blocks: per block, load 4 unpacked weight
 * blocks + 4 scale/offset pairs, iterate j∈[0,m) loading one activation
 * block + one sum_a value, and do M_TILE VPDPBUSDs + scale/offset
 * applications per j. Each weight block is read once per row-tile (vs
 * once per m tokens in the per-row gemv path), and each activation
 * block is read M_TILE times instead of n_out times — the standard
 * tiled-GEMM amortization. */
__attribute__((target(VNNI_TARGET)))
void cpu_x86_linear_q4k_mN(const float               *x,
                           const struct geist_weight *w,
                           size_t                     m,
                           struct geist_backend      *be,
                           float                     *y) {
    struct cpu_x86_state *st               = (struct cpu_x86_state *) be->state;
    const size_t          n_in             = (size_t) w->n_in;
    const size_t          n_out            = (size_t) w->n_out;
    const size_t          n_blocks_per_row = n_in / W4A8_BLOCK_ELEMS;

    /* Fallback to the per-row M=1 path when:
     *   - the mtile scratch can't grow (OOM),
     *   - or m > Q4K_MTILE_MMAX (stack acc bound).
     * Both are slow but correct. */
    if (m > Q4K_MTILE_MMAX || grow_mtile(st, m, n_in) != GEIST_OK) {
        for (size_t row = 0; row < m; row++) {
            cpu_x86_linear_q4k_m1(x + row * n_in, w, be, y + row * n_out);
        }
        return;
    }

    /* Quantize all m activation rows up front into the mtile scratch.
     * Layout: acts_mtile[r * n_in + b * 32 + l],
     *         sum_a_mtile[r * n_blocks_per_row + b]. */
    for (size_t r = 0; r < m; r++) {
        st->scale_x_mtile[r] = w4a8_quantize_acts_row(
                n_in,
                x + r * n_in,
                st->acts_mtile + r * n_in,
                st->sum_a_mtile + r * n_blocks_per_row);
    }

    /* Resolve SoA pointers once. */
    const uint8_t *weights;
    const float   *w_scales;
    const float   *w_offsets;
    blob_pointers((const uint8_t *) w->aux_fp32, n_in, n_out,
                  &weights, &w_scales, &w_offsets);

    const size_t bytes_per_row  = n_blocks_per_row * W4A8_BLOCK_BYTES_WEIGHTS;
    const size_t scales_per_row = n_blocks_per_row;

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (size_t row_tile = 0; row_tile < n_out; row_tile += Q4K_MTILE) {
        const size_t this_M = (row_tile + Q4K_MTILE <= n_out)
                                      ? Q4K_MTILE
                                      : (n_out - row_tile);

        /* Per-thread fp32 accumulator tile [this_M × m]. Stack-resident,
         * fits in L1 (4 × 2048 × 4 = 32 KiB worst case). */
        float acc[Q4K_MTILE * Q4K_MTILE_MMAX];
        memset(acc, 0, this_M * m * sizeof(float));

        for (size_t b = 0; b < n_blocks_per_row; b++) {
            /* Load + unpack 4 weight blocks; cache scales/offsets. */
            __m256i u_w[Q4K_MTILE];
            float   ws[Q4K_MTILE];
            float   wo[Q4K_MTILE];
            for (size_t i = 0; i < this_M; i++) {
                const uint8_t *w_row = weights + (row_tile + i) * bytes_per_row;
                u_w[i] = unpack_w4a8_block_to_u8(w_row + b * W4A8_BLOCK_BYTES_WEIGHTS);
                ws[i]  = w_scales[(row_tile + i) * scales_per_row + b];
                wo[i]  = w_offsets[(row_tile + i) * scales_per_row + b];
            }

            for (size_t j = 0; j < m; j++) {
                const __m256i s_a = _mm256_loadu_si256(
                        (const __m256i *) (st->acts_mtile + j * n_in + b * W4A8_BLOCK_ELEMS));
                const float sa_j = (float) st->sum_a_mtile[j * n_blocks_per_row + b];
                for (size_t i = 0; i < this_M; i++) {
                    const __m256i dot = _mm256_dpbusd_epi32(_mm256_setzero_si256(),
                                                            u_w[i], s_a);
                    const int32_t d   = hsum_i32_avx2(dot);
                    acc[i * m + j] += ws[i] * (float) d - wo[i] * sa_j;
                }
            }
        }

        /* Write tile to y, applying per-row scale_x. */
        for (size_t i = 0; i < this_M; i++) {
            for (size_t j = 0; j < m; j++) {
                y[j * n_out + row_tile + i] = st->scale_x_mtile[j] * acc[i * m + j];
            }
        }
    }
}
