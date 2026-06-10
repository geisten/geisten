/*
 * src/engine/session.h — internal definition of struct geist_session.
 *
 * Layer: ENGINE. Includes <geist.h> for public types only. Cannot include
 * Architecture or Backend internals.
 *
 * Defined in (Phase B-4):
 *   src/engine/session.c — lifecycle, decode-step orchestration
 *   src/engine/sampler.c — sampler state mutators
 *
 * Holds:
 *   - Reference to model + backend
 *   - Architecture-specific recurrent state (KV cache or SSM state)
 *   - Sampler state
 *   - Multimodal context (injected soft-token range, ordering)
 *   - Profiling counters
 */
#ifndef GEIST_INTERNAL_SESSION_H
#define GEIST_INTERNAL_SESSION_H

#ifndef GEIST_INTERNAL_ENGINE_LAYER
#error "session.h is internal to the engine layer. Use <geist.h> from outside."
#endif

#include <geist.h>

/* TODO Phase B-4: full struct definition.
 * Stub only — Phase B-1 establishes the file shape. */
struct geist_session {
    struct geist_model   *model;
    struct geist_backend *backend;

    /* Backend-owned per-session state (opaque). */
    void *backend_state;

    /* Architecture-specific recurrent state (KV cache or SSM state). */
    void *arch_state;

    /* Engine-only state (sampler, multimodal context, profiling) — populated
     * in Phase B-4 when lm.c splits. */
    char placeholder_;
};

#endif /* GEIST_INTERNAL_SESSION_H */
