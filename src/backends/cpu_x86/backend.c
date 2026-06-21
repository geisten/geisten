/*
 * src/backends/cpu_x86/backend.c — x86_64 backend, Phase 1a Step 5.
 *
 * Layer: BACKEND.
 *
 * cpu_x86's vtbl reuses cpu_scalar's slots for everything except create,
 * destroy, and resolve_weight: those three are overridden to thread the
 * per-instance state needed by the W4A8 hot path (acts + sum_a scratch),
 * and to install the cpu_x86 Q4_K M=1 kernel via cpu_x86_linear_q4k_resolve.
 *
 * The vtbl is initialized at module load via __attribute__((constructor)):
 *   1. Struct-copy cpu_scalar_vtbl into cpu_x86_vtbl.
 *   2. Override .create / .destroy / .resolve_weight.
 * Constructor runs before main, so the descriptor's vtbl pointer is
 * always valid by the time the engine calls geist_backend_create.
 *
 * Non-Q4_K dtypes fall through to cpu_scalar's resolver via the same
 * cpu_scalar_resolve_weight function — the descriptor remains drop-in
 * compatible with every Gemma 4 weight, and only the Q4_K decode path
 * routes through W4A8 + VPDPBUSD today.
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "backend.h"

#include "backend_state.h"
#include "linear_q4k.h"
#include "linear_q6k.h"

#include "heap.h"

#include <geist_backend.h>
#include <geist_weight.h>

#include <stddef.h>

/* Borrowed from cpu_scalar/backend.c (exported there). */
extern const struct geist_backend_vtbl cpu_scalar_vtbl;
extern enum geist_status               cpu_scalar_resolve_weight(struct geist_backend *be,
                                                                 struct geist_weight  *w);

/* Mutable vtbl filled in at module-init time. */
static struct geist_backend_vtbl cpu_x86_vtbl;

/* ---------- Lifecycle ---------- */

[[nodiscard]] static enum geist_status cpu_x86_create(struct geist_backend            *be,
                                                      const struct geist_backend_opts *opts) {
    (void) opts;
    struct cpu_x86_state *st = geist_backend_alloc(
            be, sizeof(*st), alignof(struct cpu_x86_state));
    if (st == nullptr) {
        geist_backend_set_error(be, GEIST_E_OOM,
                                "cpu_x86: failed to allocate %zu-byte state",
                                sizeof(*st));
        return GEIST_E_OOM;
    }
    *st       = (struct cpu_x86_state) {0};
    be->state = st;
    return GEIST_OK;
}

static void cpu_x86_destroy(struct geist_backend *be) {
    if (be == nullptr || be->state == nullptr) {
        return;
    }
    struct cpu_x86_state *st = (struct cpu_x86_state *) be->state;
    safe_free((void **) &st->acts_scratch);
    safe_free((void **) &st->sum_a_scratch);
    geist_backend_free(be, st);
    be->state = nullptr;
}

/* ---------- Resolver ---------- */

[[nodiscard]] static enum geist_status cpu_x86_resolve_weight(struct geist_backend *be,
                                                              struct geist_weight  *w) {
    /* Start from the cpu_scalar mapping: covers every dtype + sets m1/_mN
     * to the cpu_scalar kernels. Bail out on any error there. */
    enum geist_status base = cpu_scalar_resolve_weight(be, w);
    if (base != GEIST_OK) {
        return base;
    }
    /* Override the M=1 path per dtype. M>1 stays on cpu_scalar's slow
     * path for now — Phase 1b wires the W4A8 prefill kernel (see
     * docs/LINUX_X86_SPEC.md §"Prefill kernel topology").
     *
     * Q4_K → W4A8 + VPDPBUSD. Q6_K → fp32 predecode + cblas_sgemv (the
     * typical Gemma 4 tied lm_head). Other dtypes stay on cpu_scalar. */
    switch ((enum geist_dtype) w->dtype) {
    case GEIST_DTYPE_Q4_K: {
        struct cpu_x86_state *st = (struct cpu_x86_state *) be->state;
        (void) cpu_x86_linear_q4k_resolve(st, w); /* OOM → keep scalar m1 */
        break;
    }
    case GEIST_DTYPE_Q6_K:
        (void) cpu_x86_linear_q6k_resolve(w); /* OOM → keep scalar m1 */
        break;
    default:
        break;
    }
    return GEIST_OK;
}

/* ---------- Vtbl init ---------- */

__attribute__((constructor)) static void cpu_x86_init_vtbl(void) {
    cpu_x86_vtbl                  = cpu_scalar_vtbl;
    cpu_x86_vtbl.create           = cpu_x86_create;
    cpu_x86_vtbl.destroy          = cpu_x86_destroy;
    cpu_x86_vtbl.resolve_weight   = cpu_x86_resolve_weight;
}

const struct geist_backend_descriptor geist_backend_cpu_x86 = {
        .name   = "cpu_x86",
        .vtbl   = &cpu_x86_vtbl,
        .caps   = nullptr,
        .n_caps = 0,
};
