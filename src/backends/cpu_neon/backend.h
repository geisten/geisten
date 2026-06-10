/*
 * src/backends/cpu_neon/backend.h — ARM64 NEON-optimized backend.
 *
 * Layer: BACKEND. ARMv8-A AArch64 NEON intrinsics for hot kernels
 * (linear with quantized weights, fused attention, etc.). Falls back
 * to scalar paths inside the backend for ops that are not yet
 * NEON-accelerated; never reaches across to cpu_scalar's symbols.
 *
 * Defined in (Phase B-3):
 *   src/backends/cpu_neon/backend.c       — descriptor, lifecycle, capability
 *   src/backends/cpu_neon/linear_q3k.c    — W3A8 vdotq_s32 kernels
 *   src/backends/cpu_neon/linear_q4k.c    — W4A8 kernels
 *   src/backends/cpu_neon/linear_q8.c     — W8A8 kernels
 *   src/backends/cpu_neon/linear_f16.c    — F16 dot kernels
 *   src/backends/cpu_neon/attention.c     — INT8 QK NEON
 *   src/backends/cpu_neon/rmsnorm.c
 *   src/backends/cpu_neon/neon_helpers.h  — shared intrinsic wrappers
 */
#ifndef GEIST_INTERNAL_BACKEND_CPU_NEON_H
#define GEIST_INTERNAL_BACKEND_CPU_NEON_H

#ifndef GEIST_INTERNAL_BACKEND_LAYER
#error "cpu_neon/backend.h is internal to the backend layer."
#endif

#include <geist.h>

struct geist_backend_vtbl;

/* Same descriptor shape as cpu_scalar — engine treats them uniformly. */
struct geist_backend_descriptor;

extern const struct geist_backend_descriptor geist_backend_cpu_neon;

#endif /* GEIST_INTERNAL_BACKEND_CPU_NEON_H */
