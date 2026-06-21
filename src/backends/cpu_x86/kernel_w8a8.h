/*
 * src/backends/cpu_x86/kernel_w8a8.h — W8A8 dot/GEMV inner kernel.
 *
 * Layer: BACKEND (cpu_x86, internal).
 *
 * --- Contract ----------------------------------------------------------------
 *
 * Inner kernel: W8A8 row-dot, designed to map cleanly onto VPDPBUSD
 * (u8 × s8 → s32) and onto the GGUF Q6_K weight layout. Same algebraic
 * shape as kernel_w4a8.h but with 16-element blocks and 1-byte-per-weight
 * storage (Q6_K's sub-block granularity is 16, so the per-block scale and
 * offset map directly).
 *
 *   for each block b in [0, n_blocks):
 *       d_b = sum over i in [0, 16) of u_w[b, i] * a[b, i]
 *   y = scale_x * sum_b ( w_scales[b] * d_b - w_offsets[b] * sum_a[b] )
 *
 * where:
 *   u_w[b, i] ∈ [0, 255] unsigned 8-bit (Q6_K predecoder writes [0, 63]).
 *   a[b, i]   ∈ [-127, 127] signed int8, pre-quantized per row.
 *   w_scales[b], w_offsets[b] fp32 per-block.
 *   sum_a[b]  int32 per-block sum of a[b, .].
 *   scale_x   per-row fp32 activation scale.
 *
 * The hot path is allocation-free. The caller owns every buffer.
 */
#ifndef GEIST_INTERNAL_BACKEND_CPU_X86_KERNEL_W8A8_H
#define GEIST_INTERNAL_BACKEND_CPU_X86_KERNEL_W8A8_H

#ifndef GEIST_INTERNAL_BACKEND_LAYER
#error "cpu_x86/kernel_w8a8.h is internal to the backend layer."
#endif

#include <stddef.h>
#include <stdint.h>

constexpr size_t W8A8_BLOCK_ELEMS = 16;

[[nodiscard]] float w8a8_dot_scalar(
        size_t        n_blocks,
        const uint8_t weights[static n_blocks * W8A8_BLOCK_ELEMS],
        const float   w_scales[static n_blocks],
        const float   w_offsets[static n_blocks],
        const int8_t  acts[static n_blocks * W8A8_BLOCK_ELEMS],
        const int32_t sum_a_per_block[static n_blocks],
        float         scale_x);

[[nodiscard]] float w8a8_dot(
        size_t        n_blocks,
        const uint8_t weights[static n_blocks * W8A8_BLOCK_ELEMS],
        const float   w_scales[static n_blocks],
        const float   w_offsets[static n_blocks],
        const int8_t  acts[static n_blocks * W8A8_BLOCK_ELEMS],
        const int32_t sum_a_per_block[static n_blocks],
        float         scale_x);

/* Multi-row GEMV: n_rows independent dots, OMP-parallel internally. */
void w8a8_gemv(
        size_t        n_rows,
        size_t        n_blocks_per_row,
        const uint8_t weights[static n_rows * n_blocks_per_row * W8A8_BLOCK_ELEMS],
        const float   w_scales[static n_rows * n_blocks_per_row],
        const float   w_offsets[static n_rows * n_blocks_per_row],
        const int8_t  acts[static n_blocks_per_row * W8A8_BLOCK_ELEMS],
        const int32_t sum_a_per_block[static n_blocks_per_row],
        float         scale_x,
        float         out[static n_rows]);

/* Reuses the per-row int8 quantization + sum_a logic from kernel_w4a8.h
 * (W8A8 and W4A8 share the activation pipeline). The caller is expected
 * to use that path; nothing new is needed here. */

#endif /* GEIST_INTERNAL_BACKEND_CPU_X86_KERNEL_W8A8_H */
