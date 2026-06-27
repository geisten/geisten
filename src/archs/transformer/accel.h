/*
 * src/archs/transformer/accel.h - optional transformer accelerator shim.
 *
 * Layer: ARCHITECTURE. Internal to the transformer arch.
 *
 * The accelerator owns model-level device execution state; each transformer
 * session may own a matching accelerator session. The current increment is a
 * no-op shim so the CPU path remains authoritative while later Vulkan work
 * can attach device-resident graphs and logits without changing public APIs.
 */
#ifndef GEIST_INTERNAL_ARCH_TRANSFORMER_ACCEL_H
#define GEIST_INTERNAL_ARCH_TRANSFORMER_ACCEL_H

#ifndef GEIST_INTERNAL_ARCH_LAYER
#error "transformer/accel.h is internal to the architecture layer."
#endif

#include <geist.h>

struct transformer_arch_state;
struct transformer_arch_session;
struct transformer_accel;
struct transformer_accel_session;

[[nodiscard]] enum geist_status
transformer_accel_try_create(struct transformer_arch_state *state,
                             struct transformer_accel    **out);

void transformer_accel_destroy(struct transformer_accel *accel);

[[nodiscard]] enum geist_status
transformer_accel_session_create(struct transformer_accel          *accel,
                                 struct transformer_arch_session   *session,
                                 struct transformer_accel_session **out);

void transformer_accel_session_destroy(struct transformer_accel         *accel,
                                       struct transformer_accel_session *session);

[[nodiscard]] bool transformer_accel_session_decode_greedy_enabled(
    const struct transformer_accel_session *session);

[[nodiscard]] bool transformer_accel_session_prefill_text_enabled(
    const struct transformer_accel_session *session);

[[nodiscard]] bool transformer_accel_session_verify_greedy_enabled(
    const struct transformer_accel_session *session);

#endif /* GEIST_INTERNAL_ARCH_TRANSFORMER_ACCEL_H */
