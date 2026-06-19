/*
 * src/archs/transformer/arch_ops.h — internal arch-ops surface.
 *
 * Layer: ARCHITECTURE (internal). Owned by arch_ops.c. The public-
 * internal entry points (prefill_text_batch, prefill_audio_batch,
 * verify_forward, kv_truncate, pin_prefix) live in arch_state.h —
 * those are the symbols the arch.c vtable thunks dispatch into. This
 * header is for AWQ which is private to arch_state.c → arch_ops.c.
 */
#ifndef GEIST_INTERNAL_ARCH_TRANSFORMER_ARCH_OPS_H
#define GEIST_INTERNAL_ARCH_TRANSFORMER_ARCH_OPS_H

#ifndef GEIST_INTERNAL_ARCH_LAYER
#error "transformer/arch_ops.h is internal to the architecture layer."
#endif

#include "arch_state.h"

/* Apply AWQ (Activation-aware Weight Quantization) scales from the
 * given file at load time: folds attn_norm + ffn_norm gamma; stashes
 * per-layer 1/scale arrays for o_proj and down_proj. nullptr path or
 * missing file → no-op return GEIST_OK. */
enum geist_status apply_awq_to_state(struct transformer_arch_state *st, const char *awq_path);

#endif /* GEIST_INTERNAL_ARCH_TRANSFORMER_ARCH_OPS_H */
