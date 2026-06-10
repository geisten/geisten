/*
 * src/formats/gguf/iq2_s.c — IQ2_S block dequantization + flat-decode helper.
 *
 * Pure file-format decoders:
 *   dequant_iq2_s_row    : block → fp32
 *   iq2s_decode_to_int8_row : block → flat int8 (used by cpu_neon's IQ
 *     flat-cache; the lookup decides WHEN to call this, the decode is
 *     format-side).
 *
 * The W2A8 NEON kernels (standard + flat-cache variant) live in
 * src/backends/cpu_neon/kernels/iq2_s.c. The shared helper
 * iq2s_subblock_to_int8 is `static inline` here AND there.
 */
#include "internal.h"
#include "gguf_quant.h"
#include "gguf_iq_grids.h"

#include <stddef.h>
#include <stdint.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

static void dequant_iq2_s_block(const struct block_iq2_s_t* blk, float* y) {
    const float d = fp16_to_fp32(blk->d);
    const uint8_t* qs = blk->qs;
    const uint8_t* qh = blk->qh;
    const uint8_t* signs = qs + 32;   /* signs occupy second half of qs region */
    for (int ib32 = 0; ib32 < 8; ib32++) {
        float db[2];
        db[0] = d * (0.5f + (blk->scales[ib32] & 0xf)) * 0.25f;
        db[1] = d * (0.5f + (blk->scales[ib32] >>  4)) * 0.25f;
        for (int l = 0; l < 4; l++) {
            const float dl = db[l/2];
            const uint16_t idx = (uint16_t)qs[l] | ((uint16_t)(qh[ib32] << (8 - 2*l)) & 0x300);
            const uint8_t* grid = (const uint8_t*)(iq2s_grid + idx);
            for (int j = 0; j < 8; j++) {
                y[j] = dl * (float)grid[j] * ((signs[l] & kmask_iq2xs[j]) ? -1.0f : 1.0f);
            }
            y += 8;
        }
        qs += 4;
        signs += 4;
    }
}

void dequant_iq2_s_row(const void* blocks, float* out, size_t n_elems) {
    const struct block_iq2_s_t* b = (const struct block_iq2_s_t*)blocks;
    const size_t nb = n_elems / 256;
    for (size_t i = 0; i < nb; i++) dequant_iq2_s_block(&b[i], out + i * 256);
}

/* Shared with backends/cpu_neon/kernels/iq2_s.c. */
static inline void iq2s_subblock_to_int8(int8_t out[32],
                                          const uint8_t* qs_lo,    /* 4 bytes */
                                          const uint8_t* sign_b,   /* 4 bytes */
                                          uint8_t qh) {
#if defined(__ARM_NEON)
    const uint8x8_t kmask_v = vld1_u8(kmask_iq2xs);
    for (int l = 0; l < 4; l++) {
        const uint16_t idx = (uint16_t)qs_lo[l] |
                              ((uint16_t)(qh << (8 - 2*l)) & 0x300);
        const int8x8_t  grid_v  = vld1_s8((const int8_t*)(iq2s_grid + idx));
        const uint8x8_t sig_dup = vdup_n_u8(sign_b[l]);
        const uint8x8_t neg_msk = vtst_u8(sig_dup, kmask_v);
        const int8x8_t  signed_v = vbsl_s8(neg_msk, vneg_s8(grid_v), grid_v);
        vst1_s8(out + l*8, signed_v);
    }
#else
    for (int l = 0; l < 4; l++) {
        const uint16_t idx = (uint16_t)qs_lo[l] |
                              ((uint16_t)(qh << (8 - 2*l)) & 0x300);
        const uint8_t* grid = (const uint8_t*)(iq2s_grid + idx);
        const uint8_t  sig  = sign_b[l];
        for (int j = 0; j < 8; j++) {
            const int8_t val = (int8_t)grid[j];
            out[l*8 + j] = (sig & kmask_iq2xs[j]) ? (int8_t)-val : val;
        }
    }
#endif
}

void iq2s_decode_to_int8_row(const void* w_iq2s_row, int8_t* flat, size_t n_in) {
    const struct block_iq2_s_t* row = (const struct block_iq2_s_t*)w_iq2s_row;
    const size_t n_blocks_per_row = n_in / IQ2_S_BLOCK_ELEMS;
    for (size_t b = 0; b < n_blocks_per_row; b++) {
        const struct block_iq2_s_t* blk = &row[b];
        int8_t* base = flat + b * IQ2_S_BLOCK_ELEMS;
        for (int ib32 = 0; ib32 < 8; ib32++) {
            iq2s_subblock_to_int8(base + ib32 * 32,
                                   &blk->qs[ib32 * 4],
                                   &blk->qs[32 + ib32 * 4],
                                   blk->qh[ib32]);
        }
    }
}
