/*
 * src/archs/transformer/forward.h — internal forward-pass surface.
 *
 * Layer: ARCHITECTURE (internal). Implementations live in forward.c;
 * orchestration entry points (prefill_text_batch, prefill_audio_batch,
 * verify_forward, decode_step, kv_truncate, pin_prefix) live in
 * arch_state.c and call into the helpers declared here.
 *
 * NOT part of the public ABI. Only included by transformer/{forward,
 * arch_state}.c.
 */
#ifndef GEIST_INTERNAL_ARCH_TRANSFORMER_FORWARD_H
#define GEIST_INTERNAL_ARCH_TRANSFORMER_FORWARD_H

#ifndef GEIST_INTERNAL_ARCH_LAYER
#error "transformer/forward.h is internal to the architecture layer."
#endif

#include "arch_state.h"
#include <geist_backend.h>
#include <geist_types.h>

/* Layer loop: feed `seq` token rows through all GEIST_GEMMA4_NUM_LAYERS
 * layers. Writes into out_h_buf (residual stream). KV slot is at
 * q_position; advance_kv inside transformer_forward_one_layer is
 * the caller's job, this helper iterates and orchestrates. */
[[nodiscard]] enum geist_status
transformer_run_all_layers(struct transformer_arch_state *st,
                               size_t q_position, size_t seq,
                               struct geist_buffer *initial_h_buf,
                               struct geist_buffer *per_layer_input_buf,
                               struct geist_buffer *out_h_buf);

/* Drain the per-session KIVI residual ring across all non-shared layers,
 * if KIVI mode is enabled and residual_count >= R. No-op otherwise. */
void transformer_kivi_drain_full(struct transformer_arch_state *st);

/* Batched PLE precompute: dequant n PLE rows + model_proj(h) + rmsnorm.
 * Output: out_buf [n, PLE_OUT]. Caller ensures n <= st->m_max. */
[[nodiscard]] enum geist_status
compute_per_layer_inputs_batch(struct transformer_arch_state *st,
                                size_t n, const geist_token_t *ple_ids,
                                struct geist_buffer *h_buf,
                                struct geist_buffer *out_buf);

/* Output head — softcap'd lm_head on a single row of the residual
 * stream. Writes scratch_logits and sets next_token_pending +
 * logits_valid. row_idx selects which row of scratch_h_a/h_b to read
 * from (decode hot path passes 0; prefill last-row variants pass
 * seq-1). */
[[nodiscard]] enum geist_status
finalize_logits_one_row(struct transformer_arch_state *st,
                         size_t row_idx, geist_token_t *out_token);

[[nodiscard]] enum geist_status
finalize_logits_one_row_to_token_slot(struct transformer_arch_state *st,
                                      size_t row_idx,
                                      size_t token_output_offset,
                                      geist_token_t *out_token);

/* Speculative i8-sketch output head (GEIST_SPEC_HEAD=1). On a large tied F16
 * lm_head it rough-ranks the vocab via an int8 sketch, then computes exact
 * f16 logits for the top-K candidates only — writing scratch_logits and the
 * greedy argmax into *out_token. Returns true if it handled the projection
 * (caller skips the dense lm_head); false to fall back to the exact path
 * (disabled, ineligible weight, non-greedy sampling, or first-build OOM).
 * Reads the normalized hidden from scratch_h_a. */
bool transformer_spec_head_try(struct transformer_arch_state *st,
                               geist_token_t *out_token);

/* Copy one or more contiguous residual-stream rows into the output-head
 * scratch input. Uses backend buffer_copy when available so device-local
 * activation buffers do not require host mapping. */
[[nodiscard]] enum geist_status
transformer_head_copy_rows(const struct geist_backend_vtbl *v,
                           struct geist_buffer *dst,
                           const struct geist_buffer *src,
                           size_t row_idx,
                           size_t row_count,
                           size_t row_bytes);

/* Batched variant for verify_forward — runs lm_head on k rows in one
 * batched call, writes k per-position argmaxes into out_tokens. */
[[nodiscard]] enum geist_status
finalize_logits_batch(struct transformer_arch_state *st, size_t k,
                       geist_token_t *out_tokens);

/* After a batched prefill of `seq` rows, materialize logits for the
 * LAST row only (the one a subsequent decode_step will consume). */
[[nodiscard]] enum geist_status
finalize_logits_last_row(struct transformer_arch_state *st, size_t seq);

/* Dequant ONE row of an arbitrary-dtype tensor into a host fp32 row.
 * Used by the PLE single + batched paths and by the embedding lookup. */
[[nodiscard]] enum geist_status
dequant_one_row(struct geist_backend *be, const struct geist_tensor *t,
                 size_t row_idx, float *dst);

#endif /* GEIST_INTERNAL_ARCH_TRANSFORMER_FORWARD_H */
