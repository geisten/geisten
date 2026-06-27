/*
 * src/archs/transformer/forward/head.c — output-head finalize_logits
 * family (single-row, batched, last-row).
 *
 * Layer: ARCHITECTURE.
 *
 * Extracted from forward.c during R4 of the C23/AGENT.md cleanup.
 * The three finalize_* routines share the same shape: take the
 * residual stream at the post-layer point, optionally apply the embed
 * scale, project through the lm_head linear, and write logits into the
 * session scratch. They differ in batch shape and where the next-token
 * argmax goes.
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "internal.h"
#include "profile.h"
#include "../arch_state.h"
#include "../forward.h"

#include "gguf_quant.h"
#include "gemma4_kernels.h"

#include <geist.h>
#include <geist_backend.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum head_profile_stage {
    HEAD_PROFILE_COPY = 0,
    HEAD_PROFILE_NORM,
    HEAD_PROFILE_LM_HEAD,
    HEAD_PROFILE_SOFTCAP,
    HEAD_PROFILE_SAMPLE,
    HEAD_PROFILE_COUNT,
};

static uint64_t g_head_profile_ns[HEAD_PROFILE_COUNT];
static uint64_t g_head_profile_calls[HEAD_PROFILE_COUNT];
static const char * const g_head_profile_names[HEAD_PROFILE_COUNT] = {
    "copy", "norm", "lm_head", "softcap", "sample",
};
static struct transformer_forward_profile g_head_profile = {
    .title = "transformer head",
    .stage_names = g_head_profile_names,
    .stage_count = HEAD_PROFILE_COUNT,
    .ns = g_head_profile_ns,
    .calls = g_head_profile_calls,
};

static void transformer_mark_pending_logits(struct transformer_arch_state *st,
                                            geist_token_t token,
                                            bool host_valid) {
    st->sess->next_token_pending = token;
    st->sess->logits_valid = true;
    st->sess->logits_host_valid = host_valid;
    st->sess->logits_on_device = !host_valid;
}

static bool transformer_logits_buffer_is_host_visible(
    const struct geist_backend_vtbl *v,
    struct geist_buffer *buf) {

    if (v == nullptr || v->buffer_map == nullptr || buf == nullptr) {
        return false;
    }
    void *p = v->buffer_map(buf);
    if (p == nullptr) {
        return false;
    }
    v->buffer_unmap(buf);
    return true;
}

[[nodiscard]] static enum geist_status transformer_try_greedy_head_fastpath(
    struct transformer_arch_state *st,
    size_t row_idx,
    size_t token_output_offset,
    geist_token_t *out_token,
    bool *out_handled) {

    if (out_handled == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    *out_handled = false;
    if (st == nullptr || out_token == nullptr ||
        st->backend == nullptr || st->backend->desc == nullptr ||
        st->backend->desc->vtbl == nullptr) {
        return GEIST_E_INVALID_ARG;
    }

    struct geist_backend *be = st->backend;
    const struct geist_backend_vtbl *v = be->desc->vtbl;
    if (st->sess->temperature != 0.0f || v->greedy_head == nullptr) {
        return GEIST_OK;
    }
    const bool deferred_token =
        st->sess->backend_command_sequence_active &&
        v->command_sequence_read_token != nullptr;
    if (st->d_model <= 0 || st->vocab_size <= 0 ||
        row_idx > SIZE_MAX / ((size_t) st->d_model * sizeof(float))) {
        return GEIST_E_INVALID_ARG;
    }

    struct geist_tensor hidden =
        view_1d(st->sess->scratch_h_b, st->d_model);
    hidden.offset = row_idx * (size_t) st->d_model * sizeof(float);
    struct geist_tensor norm_weight =
        view_1d(st->output_norm.buffer, st->d_model);
    struct geist_tensor normed =
        view_1d(st->sess->scratch_h_a, st->d_model);
    struct geist_tensor logits =
        view_1d(st->sess->scratch_logits, st->vocab_size);

    const struct geist_backend_greedy_head head = {
        .struct_size = sizeof(head),
        .d_model = (size_t) st->d_model,
        .vocab_size = (size_t) st->vocab_size,
        .token_output_offset = token_output_offset,
        .eps = st->config.rms_eps,
        .hidden = &hidden,
        .norm_weight = &norm_weight,
        .lm_head_weight = &st->embed_table,
        .normed_scratch = &normed,
        .logits = &logits,
    };
    geist_token_t token = -1;
    enum geist_status s = v->greedy_head(be, &head, &token);
    if (s == GEIST_E_UNSUPPORTED) {
        if (st->sess->backend_command_sequence_active) {
            return s;
        }
        return GEIST_OK;
    }
    if (s != GEIST_OK) {
        return s;
    }

    if (deferred_token) {
        *out_token = -1;
        *out_handled = true;
        return GEIST_OK;
    }

    *out_token = token;
    const bool host_valid =
        transformer_logits_buffer_is_host_visible(v, st->sess->scratch_logits);
    transformer_mark_pending_logits(st, token, host_valid);
    *out_handled = true;
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status transformer_try_greedy_head_batch_fastpath(
    struct transformer_arch_state *st,
    size_t row_count,
    geist_token_t *out_tokens,
    bool *out_handled) {

    if (out_handled == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    *out_handled = false;
    if (st == nullptr || out_tokens == nullptr ||
        st->backend == nullptr || st->backend->desc == nullptr ||
        st->backend->desc->vtbl == nullptr) {
        return GEIST_E_INVALID_ARG;
    }

    struct geist_backend *be = st->backend;
    const struct geist_backend_vtbl *v = be->desc->vtbl;
    if (st->sess->temperature != 0.0f || v->greedy_head_batch == nullptr) {
        return GEIST_OK;
    }
    if (st->d_model <= 0 || st->vocab_size <= 0 || row_count == 0 ||
        row_count > (size_t) INT64_MAX ||
        row_count > SIZE_MAX / ((size_t) st->d_model * sizeof(float)) ||
        row_count > SIZE_MAX / ((size_t) st->vocab_size * sizeof(float))) {
        return GEIST_E_INVALID_ARG;
    }

    struct geist_tensor hidden =
        view_2d(st->sess->scratch_h_b, (int64_t) row_count, st->d_model);
    struct geist_tensor norm_weight =
        view_1d(st->output_norm.buffer, st->d_model);
    struct geist_tensor normed =
        view_2d(st->sess->scratch_h_a, (int64_t) row_count, st->d_model);
    struct geist_tensor logits =
        view_2d(st->sess->scratch_logits, (int64_t) row_count,
                st->vocab_size);

    const struct geist_backend_greedy_head_batch head = {
        .struct_size = sizeof(head),
        .d_model = (size_t) st->d_model,
        .vocab_size = (size_t) st->vocab_size,
        .row_count = row_count,
        .token_output_offset = 0,
        .eps = st->config.rms_eps,
        .hidden = &hidden,
        .norm_weight = &norm_weight,
        .lm_head_weight = &st->embed_table,
        .normed_scratch = &normed,
        .logits = &logits,
    };
    enum geist_status s = v->greedy_head_batch(be, &head, out_tokens);
    if (s == GEIST_E_UNSUPPORTED) {
        if (st->sess->backend_command_sequence_active) {
            return s;
        }
        return GEIST_OK;
    }
    if (s != GEIST_OK) {
        return s;
    }

    if (st->sess->backend_command_sequence_active &&
        v->command_sequence_read_tokens != nullptr) {
        for (size_t row = 0; row < row_count; row++) {
            out_tokens[row] = -1;
        }
        *out_handled = true;
        return GEIST_OK;
    }

    const bool host_valid =
        transformer_logits_buffer_is_host_visible(v, st->sess->scratch_logits);
    transformer_mark_pending_logits(st, out_tokens[row_count - 1], host_valid);
    *out_handled = true;
    return GEIST_OK;
}

[[nodiscard]] enum geist_status
transformer_head_copy_rows(const struct geist_backend_vtbl *v,
                           struct geist_buffer *dst,
                           const struct geist_buffer *src,
                           size_t row_idx,
                           size_t row_count,
                           size_t row_bytes) {

    if (v == nullptr || dst == nullptr || src == nullptr ||
        row_count == 0 || row_bytes == 0) {
        return GEIST_E_INVALID_ARG;
    }
    if (row_idx > SIZE_MAX / row_bytes ||
        row_count > SIZE_MAX / row_bytes) {
        return GEIST_E_INVALID_ARG;
    }
    const size_t src_offset = row_idx * row_bytes;
    const size_t n_bytes = row_count * row_bytes;

    if (v->buffer_copy != nullptr) {
        return v->buffer_copy(dst, 0, src, src_offset, n_bytes);
    }

    if (v->buffer_map == nullptr || v->buffer_unmap == nullptr ||
        src_offset > SIZE_MAX - n_bytes) {
        return GEIST_E_INVALID_ARG;
    }
    const uint8_t *src_bytes =
        (const uint8_t *) v->buffer_map((struct geist_buffer *) src);
    uint8_t *dst_bytes = (uint8_t *) v->buffer_map(dst);
    if (src_bytes == nullptr || dst_bytes == nullptr) {
        if (src_bytes != nullptr) {
            v->buffer_unmap((struct geist_buffer *) src);
        }
        if (dst_bytes != nullptr) {
            v->buffer_unmap(dst);
        }
        return GEIST_E_BACKEND;
    }

    memcpy(dst_bytes, src_bytes + src_offset, n_bytes);
    v->buffer_unmap((struct geist_buffer *) src);
    v->buffer_unmap(dst);
    return GEIST_OK;
}

[[nodiscard]] enum geist_status finalize_logits_one_row_to_token_slot(
    struct transformer_arch_state *st, size_t row_idx,
    size_t token_output_offset, geist_token_t *out_token) {

    struct geist_backend *be = st->backend;
    const struct geist_backend_vtbl *v = be->desc->vtbl;
    enum geist_status s;
    const bool profile = transformer_profile_enabled(&g_head_profile);
    uint64_t t0 = profile ? transformer_profile_now_ns() : 0;

    {
        bool handled = false;
        s = transformer_try_greedy_head_fastpath(st, row_idx,
                                                 token_output_offset,
                                                 out_token, &handled);
        transformer_profile_add(&g_head_profile, HEAD_PROFILE_LM_HEAD, t0);
        if (s != GEIST_OK) { return s; }
        if (handled) { return GEIST_OK; }
    }
    if (transformer_state_command_sequence_active(st)) {
        geist_backend_set_error(be, GEIST_E_UNSUPPORTED,
                                "transformer head: command sequence requires "
                                "greedy_head fastpath");
        return GEIST_E_UNSUPPORTED;
    }

    /* Copy chosen row of scratch_h_b into scratch_h_a (reuse as a clean
     * [1, HIDDEN] buffer for the output head). */
    {
        const size_t bytes = st->d_model * sizeof(float);
        s = transformer_head_copy_rows(v, st->sess->scratch_h_a,
                                       st->sess->scratch_h_b,
                                       row_idx, 1, bytes);
        if (s != GEIST_OK) { return s; }
    }
    transformer_profile_add(&g_head_profile, HEAD_PROFILE_COPY, t0);

    struct geist_tensor t_h_1d = view_1d(st->sess->scratch_h_a, st->d_model);
    struct geist_tensor t_w_out_norm = view_1d(st->output_norm.buffer, st->d_model);
    t0 = profile ? transformer_profile_now_ns() : 0;
    s = v->rmsnorm(be, &t_h_1d, &t_w_out_norm, st->config.rms_eps, &t_h_1d);
    transformer_profile_add(&g_head_profile, HEAD_PROFILE_NORM, t0);
    if (s != GEIST_OK) { return s; }

    /* Speculative i8-sketch fast path (GEIST_SPEC_HEAD=1). Handles the whole
     * projection + greedy argmax when eligible; otherwise falls through to the
     * exact dense lm_head below. Reads the normalized hidden from scratch_h_a. */
    t0 = profile ? transformer_profile_now_ns() : 0;
    if (transformer_spec_head_try(st, out_token)) {
        transformer_profile_add(&g_head_profile, HEAD_PROFILE_LM_HEAD, t0);
        transformer_mark_pending_logits(st, *out_token, true);
        return GEIST_OK;
    }

    struct geist_tensor t_h_2d = view_2d(st->sess->scratch_h_a, 1, st->d_model);
    struct geist_tensor t_logits_2d = view_2d(st->sess->scratch_logits, 1, st->vocab_size);
    t0 = profile ? transformer_profile_now_ns() : 0;
    s = linear_w_or_legacy(be, v, st->sess->scratch_h_a, st->sess->scratch_logits,
                            &st->embed_table_w, /* seq = */ 1,
                            &t_h_2d, &st->embed_table, &t_logits_2d);
    transformer_profile_add(&g_head_profile, HEAD_PROFILE_LM_HEAD, t0);
    if (s != GEIST_OK) { return s; }

    /* Softcap. P1.5: family-conditional. H1: skip in greedy mode — tanh is
     * monotonic so argmax is identical with or without softcap. Saves
     * ~262 144 × tanhf calls per token (~5% of decode on Gemma 4). */
    const bool sampler_needs_softcap = st->sess->temperature > 0.0f;
    if (st->config.logit_softcap > 0.0f && sampler_needs_softcap) {
        t0 = profile ? transformer_profile_now_ns() : 0;
        float *p = (float *) v->buffer_map(st->sess->scratch_logits);
        const float c = st->config.logit_softcap;
        for (size_t i = 0; i < (size_t) st->vocab_size; i++) {
            p[i] = tanhf(p[i] / c) * c;
        }
        v->buffer_unmap(st->sess->scratch_logits);
        transformer_profile_add(&g_head_profile, HEAD_PROFILE_SOFTCAP, t0);
    }

    /* Sampler dispatch. scratch_logits already holds the softcapped
     * row; on CPU backends buffer_map returns the host pointer directly,
     * so the sampler reads it without a copy. P0.1 (2026-05-15): no more
     * per-call 1 MB heap_alloc_aligned. */
    geist_token_t best_id;
    if (st->sess->temperature == 0.0f && v->argmax_f32 != nullptr) {
        struct geist_tensor t_logits_1d =
            view_1d(st->sess->scratch_logits, st->vocab_size);
        t0 = profile ? transformer_profile_now_ns() : 0;
        s = v->argmax_f32(be, &t_logits_1d, &best_id);
        transformer_profile_add(&g_head_profile, HEAD_PROFILE_SAMPLE, t0);
        if (s == GEIST_OK) {
            *out_token = best_id;
            const bool host_valid =
                transformer_logits_buffer_is_host_visible(v, st->sess->scratch_logits);
            transformer_mark_pending_logits(st, best_id, host_valid);
            return GEIST_OK;
        }
        if (s != GEIST_E_UNSUPPORTED) {
            return s;
        }
    }
    {
        t0 = profile ? transformer_profile_now_ns() : 0;
        const float *logits = (const float *) v->buffer_map(st->sess->scratch_logits);
        if (logits == nullptr) { return GEIST_E_BACKEND; }
        if (st->sess->temperature == 0.0f) {
            best_id = geist_sampler_argmax((size_t) st->vocab_size, logits);
        } else if (st->sess->top_k > 1) {
            best_id = geist_sampler_top_k_ws(&st->sess->sampler_ws, logits,
                                              st->sess->top_k, st->sess->temperature, &st->sess->rng);
        } else if (st->sess->top_p > 0.0f && st->sess->top_p < 1.0f) {
            best_id = geist_sampler_top_p_ws(&st->sess->sampler_ws, logits,
                                              st->sess->top_p, st->sess->temperature, &st->sess->rng);
        } else {
            best_id = geist_sampler_temperature((size_t) st->vocab_size,
                                                 logits, st->sess->temperature, &st->sess->rng);
        }
        v->buffer_unmap(st->sess->scratch_logits);
        transformer_profile_add(&g_head_profile, HEAD_PROFILE_SAMPLE, t0);
    }
    *out_token = best_id;
    transformer_mark_pending_logits(st, best_id, true);
    return GEIST_OK;
}

[[nodiscard]] enum geist_status finalize_logits_one_row(
    struct transformer_arch_state *st, size_t row_idx,
    geist_token_t *out_token) {

    return finalize_logits_one_row_to_token_slot(st, row_idx, 0, out_token);
}

/* Batched variant of finalize_logits_one_row for verify_forward: runs
 * the lm_head linear ONCE over all k rows of scratch_h_b, then does
 * per-row softcap + argmax/sampler. Lights up the M>1 NEON IQ-format
 * prefill kernels for the 262K-wide projection — vs k separate M=1
 * SGEMV calls that re-stream the embed_table weight rows. Writes
 * out_tokens[0..k-1] with the per-position sampled token. */
[[nodiscard]] enum geist_status finalize_logits_batch(
    struct transformer_arch_state *st, size_t k,
    geist_token_t *out_tokens) {

    struct geist_backend *be = st->backend;
    const struct geist_backend_vtbl *v = be->desc->vtbl;
    enum geist_status s;

    {
        bool handled = false;
        s = transformer_try_greedy_head_batch_fastpath(st, k, out_tokens,
                                                       &handled);
        if (s != GEIST_OK) { return s; }
        if (handled) { return GEIST_OK; }
    }
    if (transformer_state_command_sequence_active(st)) {
        geist_backend_set_error(be, GEIST_E_UNSUPPORTED,
                                "transformer head: command sequence requires "
                                "greedy_head_batch fastpath");
        return GEIST_E_UNSUPPORTED;
    }

    /* Source: scratch_h_b [k, HIDDEN]. Apply output_norm row-wise
     * (rmsnorm batches naturally over the leading dim). Reuse
     * scratch_h_a as the normed buffer so we don't trash h_b which
     * the caller may still want. */
    {
        const size_t row_bytes = st->d_model * sizeof(float);
        s = transformer_head_copy_rows(v, st->sess->scratch_h_a,
                                       st->sess->scratch_h_b,
                                       0, k, row_bytes);
        if (s != GEIST_OK) { return s; }
    }
    struct geist_tensor t_h_2d = view_2d(st->sess->scratch_h_a, (int64_t) k, st->d_model);
    struct geist_tensor t_w_out_norm = view_1d(st->output_norm.buffer, st->d_model);
    s = v->rmsnorm(be, &t_h_2d, &t_w_out_norm, st->config.rms_eps, &t_h_2d);
    if (s != GEIST_OK) { return s; }

    /* Single batched linear: [k, HIDDEN] @ embed_table^T → [k, VOCAB]. */
    struct geist_tensor t_logits_2d =
        view_2d(st->sess->scratch_logits, (int64_t) k, st->vocab_size);
    s = linear_w_or_legacy(be, v, st->sess->scratch_h_a, st->sess->scratch_logits,
                            &st->embed_table_w, k,
                            &t_h_2d, &st->embed_table, &t_logits_2d);
    if (s != GEIST_OK) { return s; }

    /* Per-row softcap + sampler. Softcap is monotonic, so for greedy
     * (temperature=0) argmax is identical with or without it — skip
     * the ~262k tanhf calls per row. Stochastic modes still need it
     * to preserve the correct logit distribution. P0.1 (2026-05-15):
     * the row is softcapped in place inside scratch_logits and the
     * sampler reads the same row directly — no per-row 1 MB
     * heap_alloc_aligned, no per-row memcpy. */
    {
        float *all = (float *) v->buffer_map(st->sess->scratch_logits);
        if (all == nullptr) { return GEIST_E_BACKEND; }
        const float c = st->config.logit_softcap;
        const bool  sampler_needs_softcap = st->sess->temperature > 0.0f;
        const bool  do_softcap = c > 0.0f && sampler_needs_softcap;
        for (size_t row = 0; row < k; row++) {
            float *p = all + row * (size_t) st->vocab_size;
            if (do_softcap) {
                for (size_t i = 0; i < (size_t) st->vocab_size; i++) {
                    p[i] = tanhf(p[i] / c) * c;
                }
            }

            geist_token_t best_id;
            if (st->sess->temperature == 0.0f) {
                best_id = geist_sampler_argmax((size_t) st->vocab_size, p);
            } else if (st->sess->top_k > 1) {
                best_id = geist_sampler_top_k_ws(&st->sess->sampler_ws, p,
                                                  st->sess->top_k, st->sess->temperature, &st->sess->rng);
            } else if (st->sess->top_p > 0.0f && st->sess->top_p < 1.0f) {
                best_id = geist_sampler_top_p_ws(&st->sess->sampler_ws, p,
                                                  st->sess->top_p, st->sess->temperature, &st->sess->rng);
            } else {
                best_id = geist_sampler_temperature((size_t) st->vocab_size,
                                                     p, st->sess->temperature, &st->sess->rng);
            }
            out_tokens[row] = best_id;
        }
        v->buffer_unmap(st->sess->scratch_logits);
    }
    return GEIST_OK;
}

[[nodiscard]] enum geist_status finalize_logits_last_row(
    struct transformer_arch_state *st, size_t seq) {
    geist_token_t tok = -1;
    enum geist_status s = finalize_logits_one_row(st, seq - 1, &tok);
    if (s != GEIST_OK) { return s; }
    if (transformer_state_command_sequence_active(st) &&
        st->backend != nullptr &&
        st->backend->desc != nullptr &&
        st->backend->desc->vtbl != nullptr &&
        st->backend->desc->vtbl->command_sequence_read_token != nullptr) {
        return GEIST_OK;
    }
    if (!st->sess->logits_valid || tok < 0) {
        geist_backend_set_error(st->backend, GEIST_E_INTERNAL,
                                "transformer head: final row produced no pending token");
        return GEIST_E_INTERNAL;
    }
    return GEIST_OK;
}
