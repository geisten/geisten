/*
 * src/backends/cpu_x86/kernel_q4kx8_gemm.h — Q4_K × Q8_K GEMM kernel.
 *
 * Layer: BACKEND (cpu_x86, internal).
 *
 * Computes Y[M, N] = X_q8kx4 @ (W_q4kx8)^T where:
 *   W is stored as block_q4_Kx8 (8 output rows interleaved per super-block).
 *   X is pre-quantized to block_q8_Kx4 (4 m-rows interleaved per super-block).
 *   Y is fp32 row-major [M, N].
 *
 * Constraints:
 *   - M must be a multiple of 4 (Q8_Kx4 row group).
 *   - N must be a multiple of 8 (Q4_Kx8 row group).
 *   - K (= n_in) must be a multiple of QK_K (= 256).
 *
 * The scalar reference walks both layouts step-by-step and computes the
 * same arithmetic an AVX-512BW lane-parallel kernel would: per super-block
 * sum (acc_main - acc_min), with acc_min using the bsums precomputed at
 * quantize time.
 */
#ifndef GEIST_INTERNAL_BACKEND_CPU_X86_KERNEL_Q4KX8_GEMM_H
#define GEIST_INTERNAL_BACKEND_CPU_X86_KERNEL_Q4KX8_GEMM_H

#ifndef GEIST_INTERNAL_BACKEND_LAYER
#error "cpu_x86/kernel_q4kx8_gemm.h is internal to the backend layer."
#endif

#include "q4k_to_q4kx8.h"
#include "q8_kx4.h"

#include <stddef.h>

/* Scalar reference. Correct-oracle for the AVX-512 variant. Allocation-
 * free; caller owns Y. */
void q4kx8_gemm_scalar(size_t                     M,
                       size_t                     N,
                       size_t                     K,
                       const struct block_q8_Kx4 *X,
                       const struct block_q4_Kx8 *W,
                       float                      Y[static M * N]);

#endif /* GEIST_INTERNAL_BACKEND_CPU_X86_KERNEL_Q4KX8_GEMM_H */
