/*
 * src/formats/gguf/iq3_s.c — IQ3_S block dequantization + flat-decode helper.
 *
 * Pure file-format decoders:
 *   dequant_iq3_s_row        : block → fp32
 *   iq3s_decode_to_int8_row  : block → flat int8 (cpu_neon IQ flat-cache).
 *
 * W3A8 NEON kernels (standard + flat-cache variant) live in
 * src/backends/cpu_neon/kernels/iq3_s.c. The shared
 * iq3s_subblock_to_int8 helper is `static inline` in both files.
 */
#include "quant_blocks.h"
#include "quant.h"
#include "iq_grids.h"

#include <stddef.h>
#include <stdint.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

static void dequant_iq3_s_block(const struct block_iq3_s_t *blk, float *y) {
    const float    d     = fp16_to_fp32(blk->d);
    const uint8_t *qs    = blk->qs;
    const uint8_t *qh    = blk->qh;
    const uint8_t *signs = blk->signs;
    /* Each ib32 covers 32 elements. Two halves of 4 super-blocks each share scale[ib]. */
    for (int ib32 = 0; ib32 < 8; ib32 += 2) {
        const int   ib  = ib32 / 2; /* scale index 0..3 */
        const float db1 = d * (1.0f + 2 * ((blk->scales[ib] >> 0) & 0xf));
        const float db2 = d * (1.0f + 2 * ((blk->scales[ib] >> 4) & 0xf));
        for (int l = 0; l < 4; l++) {
            const uint8_t *grid1 =
                    (const uint8_t *) (iq3s_grid +
                                       ((uint32_t) qs[2 * l + 0] |
                                        ((uint32_t) (qh[ib32 + 0] << (8 - 2 * l)) & 256)));
            const uint8_t *grid2 =
                    (const uint8_t *) (iq3s_grid +
                                       ((uint32_t) qs[2 * l + 1] |
                                        ((uint32_t) (qh[ib32 + 0] << (7 - 2 * l)) & 256)));
            for (int j = 0; j < 4; j++) {
                y[j + 0] =
                        db1 * (float) grid1[j] * ((signs[l] & kmask_iq2xs[j + 0]) ? -1.0f : 1.0f);
                y[j + 4] =
                        db1 * (float) grid2[j] * ((signs[l] & kmask_iq2xs[j + 4]) ? -1.0f : 1.0f);
            }
            y += 8;
        }
        qs += 8;
        signs += 4;
        for (int l = 0; l < 4; l++) {
            const uint8_t *grid1 =
                    (const uint8_t *) (iq3s_grid +
                                       ((uint32_t) qs[2 * l + 0] |
                                        ((uint32_t) (qh[ib32 + 1] << (8 - 2 * l)) & 256)));
            const uint8_t *grid2 =
                    (const uint8_t *) (iq3s_grid +
                                       ((uint32_t) qs[2 * l + 1] |
                                        ((uint32_t) (qh[ib32 + 1] << (7 - 2 * l)) & 256)));
            for (int j = 0; j < 4; j++) {
                y[j + 0] =
                        db2 * (float) grid1[j] * ((signs[l] & kmask_iq2xs[j + 0]) ? -1.0f : 1.0f);
                y[j + 4] =
                        db2 * (float) grid2[j] * ((signs[l] & kmask_iq2xs[j + 4]) ? -1.0f : 1.0f);
            }
            y += 8;
        }
        qs += 8;
        signs += 4;
    }
}

void dequant_iq3_s_row(const void *blocks, float *out, size_t n_elems) {
    const struct block_iq3_s_t *b  = (const struct block_iq3_s_t *) blocks;
    const size_t                nb = n_elems / 256;
    for (size_t i = 0; i < nb; i++)
        dequant_iq3_s_block(&b[i], out + i * 256);
}

/* Shared with backends/cpu_neon/kernels/iq3_s.c. */
static inline void iq3s_subblock_to_int8(int8_t         out[32],
                                         const uint8_t *qs_lo,  /* 8 bytes */
                                         const uint8_t *sign_b, /* 4 bytes */
                                         uint8_t        qh) {
#if defined(__ARM_NEON)
    const uint8x8_t kmask_v = vld1_u8(kmask_iq2xs);
    for (int l = 0; l < 4; l++) {
        const uint32_t idx1 = (uint32_t) qs_lo[2 * l + 0] | ((uint32_t) (qh << (8 - 2 * l)) & 256);
        const uint32_t idx2 = (uint32_t) qs_lo[2 * l + 1] | ((uint32_t) (qh << (7 - 2 * l)) & 256);
        /* Pack grid1[0..3] || grid2[0..3] into one 8-byte NEON vector. */
        uint32x2_t pair          = vdup_n_u32(0);
        pair                     = vld1_lane_u32((const uint32_t *) (iq3s_grid + idx1), pair, 0);
        pair                     = vld1_lane_u32((const uint32_t *) (iq3s_grid + idx2), pair, 1);
        const int8x8_t  grid_v   = vreinterpret_s8_u32(pair);
        const uint8x8_t sig_dup  = vdup_n_u8(sign_b[l]);
        const uint8x8_t neg_msk  = vtst_u8(sig_dup, kmask_v);
        const int8x8_t  signed_v = vbsl_s8(neg_msk, vneg_s8(grid_v), grid_v);
        vst1_s8(out + l * 8, signed_v);
    }
#else
    for (int l = 0; l < 4; l++) {
        const uint32_t idx1  = (uint32_t) qs_lo[2 * l + 0] | ((uint32_t) (qh << (8 - 2 * l)) & 256);
        const uint32_t idx2  = (uint32_t) qs_lo[2 * l + 1] | ((uint32_t) (qh << (7 - 2 * l)) & 256);
        const uint8_t *grid1 = (const uint8_t *) (iq3s_grid + idx1);
        const uint8_t *grid2 = (const uint8_t *) (iq3s_grid + idx2);
        const uint8_t  sig   = sign_b[l];
        for (int j = 0; j < 4; j++) {
            const int8_t v1    = (int8_t) grid1[j];
            const int8_t v2    = (int8_t) grid2[j];
            out[l * 8 + j + 0] = (sig & kmask_iq2xs[j + 0]) ? (int8_t) -v1 : v1;
            out[l * 8 + j + 4] = (sig & kmask_iq2xs[j + 4]) ? (int8_t) -v2 : v2;
        }
    }
#endif
}

void iq3s_decode_to_int8_row(const void *w_iq3s_row, int8_t *flat, size_t n_in) {
    const struct block_iq3_s_t *row              = (const struct block_iq3_s_t *) w_iq3s_row;
    const size_t                n_blocks_per_row = n_in / IQ3_S_BLOCK_ELEMS;
    for (size_t b = 0; b < n_blocks_per_row; b++) {
        const struct block_iq3_s_t *blk  = &row[b];
        int8_t                     *base = flat + b * IQ3_S_BLOCK_ELEMS;
        for (int ib = 0; ib < 4; ib++) {
            iq3s_subblock_to_int8(base + ib * 64 + 0,
                                  &blk->qs[ib * 16 + 0],
                                  &blk->signs[ib * 8 + 0],
                                  blk->qh[ib * 2 + 0]);
            iq3s_subblock_to_int8(base + ib * 64 + 32,
                                  &blk->qs[ib * 16 + 8],
                                  &blk->signs[ib * 8 + 4],
                                  blk->qh[ib * 2 + 1]);
        }
    }
}
