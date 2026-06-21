/*
 * src/backends/cpu_x86/backend_state.h — per-instance state held in
 * geist_backend->state.
 *
 * Layer: BACKEND (cpu_x86, internal).
 *
 * Lifecycle:
 *   - cpu_x86_create (in backend.c) allocates struct cpu_x86_state via
 *     geist_backend_alloc; scratch buffers start at null/zero.
 *   - cpu_x86_resolve_weight grows the scratch (via heap.h) the first
 *     time it sees a weight with n_in larger than scratch_cap.
 *   - cpu_x86_destroy frees both the scratch buffers and the state.
 *
 * The scratch holds the per-row int8 activation buffer and the per-block
 * sum_a integer buffer. Both are written by w4a8_quantize_acts_row and
 * read by w4a8_gemv inside one linear_m1 invocation; not shared across
 * threads, not racy under multi-session because each backend instance
 * owns its own state.
 */
#ifndef GEIST_INTERNAL_BACKEND_CPU_X86_BACKEND_STATE_H
#define GEIST_INTERNAL_BACKEND_CPU_X86_BACKEND_STATE_H

#ifndef GEIST_INTERNAL_BACKEND_LAYER
#error "cpu_x86/backend_state.h is internal to the backend layer."
#endif

#include <stddef.h>
#include <stdint.h>

struct cpu_x86_state {
    int8_t  *acts_scratch;  /* int8 activation buffer; heap_alloc_aligned. */
    int32_t *sum_a_scratch; /* per-block sum_a int32 buffer; heap-aligned. */
    size_t   scratch_cap;   /* max n_in (in fp32 elements) the scratch covers. */
};

#endif /* GEIST_INTERNAL_BACKEND_CPU_X86_BACKEND_STATE_H */
