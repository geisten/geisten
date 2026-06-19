/*
 * ptqtp_kernel.h — PTQTP (Post-Training Quantization to Trit-Planes) kernels.
 *
 * Each weight is approximated as
 *
 *     w_ij  ≈  Σ_{k=0..K-1}  α_k[i, g(j)] · t_k[i, j]      with t_k ∈ {-1, 0, +1}
 *
 * where K ∈ {2, 3} is the number of trit-planes, g(j) = j / group_size, and
 * α is shared per (output row, input group). Group size is fixed per tensor
 * and recoverable as n_in / n_groups.
 *
 * Storage:
 *   K = 2:  trits packed 2 weights per byte (idx ∈ {0..8} = (T1+1)*3 + (T2+1))
 *   K = 3:  trits packed 1 weight per byte  (idx ∈ {0..26} = (T1+1)*9 + (T2+1)*3 + (T3+1))
 *   alpha:  K · fp16 per (row, group); a parallel fp32 arena is built at load
 *
 * Hot path is allocation-free. Caller provides x_q8 (pre-quantized activations
 * via quantize_x_int8_sym) and a y output buffer of n_out floats.
 */
#ifndef GEIST_PTQTP_KERNEL_H
#define GEIST_PTQTP_KERNEL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Compile-time invariants checked once for all kernels. */
static_assert(sizeof(float) == 4, "PTQTP assumes IEEE-754 binary32");
static_assert(sizeof(uint16_t) == 2, "PTQTP assumes 16-bit fp16 storage");

/* 2-plane GEMV with FP32-precomputed alpha (loader-side conversion).
 * Use this when the loader has already promoted alpha to fp32 (one-time
 * cost at startup; hot path avoids the conversion). */
void ptqtp_gemv_2plane_fp32alpha(size_t        n_in,
                                 size_t        n_out,
                                 size_t        group_size,
                                 const int8_t *x_q8,    /* [n_in]; pre-quantized symmetric int8 */
                                 float         scale_x, /* x quant scale; output multiplied by it */
                                 const uint8_t *trits,  /* [n_out * n_in / 2]; joint nibble */
                                 const float   *alpha_fp32, /* [n_out * n_groups * 2] */
                                 float         *y           /* [n_out]; output, overwritten */
);

/* 2-plane GEMV with FP16 alpha (NEON vcvt_f32_f16 inline). Slightly less
 * bandwidth on the alpha side; identical math. */
void ptqtp_gemv_2plane_fp16alpha(size_t          n_in,
                                 size_t          n_out,
                                 size_t          group_size,
                                 const int8_t   *x_q8,
                                 float           scale_x,
                                 const uint8_t  *trits,
                                 const uint16_t *alpha_fp16, /* [n_out * n_groups * 2] */
                                 float          *y);

/* 2-plane batched GEMM. M tokens × n_in input → M × n_out output. Each
 * weight row is read once and accumulated over all M tokens. */
void ptqtp_gemm_2plane_fp32alpha(size_t         M,
                                 size_t         n_in,
                                 size_t         n_out,
                                 size_t         group_size,
                                 const int8_t  *x_q8,    /* [M * n_in] row-major */
                                 const float   *scale_x, /* [M] */
                                 const uint8_t *trits,
                                 const float   *alpha_fp32,
                                 float         *y /* [M * n_out] row-major */
);

/* 3-plane GEMV with FP32-precomputed alpha. Trits are 1 byte per weight,
 * idx ∈ [0, 27). Decoder uses 3× vqtbl2q (32-entry LUT) per chunk. */
void ptqtp_gemv_3plane_fp32alpha(size_t         n_in,
                                 size_t         n_out,
                                 size_t         group_size,
                                 const int8_t  *x_q8,
                                 float          scale_x,
                                 const uint8_t *trits,      /* [n_out * n_in] */
                                 const float   *alpha_fp32, /* [n_out * n_groups * 3] */
                                 float         *y);

/* 3-plane PACKED-5-bit GEMV. Same 3-plane semantics as above, but trits
 * stored at 5 bpw (vs 8 bpw): per row, low nibbles of every weight come
 * first (n_in/2 bytes), then a high-bit stream (n_in/8 bytes, 8 weights
 * packed per byte). Constraint: n_in % 8 == 0.
 *
 * Reconstruction per weight i:
 *     nibble  = (low_stream[i/2] >> ((i & 1) * 4)) & 0x0F
 *     hi_bit  = (high_stream[i/8] >> (i & 7)) & 0x01
 *     idx     = nibble | (hi_bit << 4)              ∈ [0, 27)
 * then look up T1/T2/T3 in the 32-entry LUTs.
 *
 * The high-bit stream is decoded via a 256-byte expansion LUT
 * (uint8x8 per source byte). NEON path: ~16 ops/16 weights vs the
 * standard 3-plane kernel's ~8 ops/16 weights (~2× compute) with
 * 5/8 = 62.5% of the bandwidth. Wins when memory-bound. */
void ptqtp_gemv_3plane_packed5_fp32alpha(size_t         n_in,
                                         size_t         n_out,
                                         size_t         group_size,
                                         const int8_t  *x_q8,
                                         float          scale_x,
                                         const uint8_t *trits, /* per row: n_in/2 + n_in/8 bytes */
                                         const float   *alpha_fp32,
                                         float         *y);

#ifdef __cplusplus
}
#endif

#endif /* GEIST_PTQTP_KERNEL_H */
