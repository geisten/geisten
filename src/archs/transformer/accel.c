/*
 * src/archs/transformer/accel.c - optional transformer accelerator shim.
 *
 * Layer: ARCHITECTURE.
 *
 * This is the lifecycle seam for the Vulkan fastpath. It only enables the
 * narrow device-resident decode path after backend capability, model geometry,
 * and loaded weight-layout checks have all passed.
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "accel.h"
#include "arch_state.h"

#include <geist.h>
#include <geist_backend.h>

#include "heap.h"

#include <inttypes.h>
#include <stdalign.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct transformer_accel {
    struct geist_backend *backend;
    struct geist_backend_accel_caps caps;
    bool decode_greedy_path;
    bool prefill_text_path;
};

struct transformer_accel_session {
    bool greedy_device_path;
    bool prefill_text_device_path;
    bool verify_greedy_device_path;
};

static bool transformer_accel_debug_enabled(void) {
    const char *value = getenv("GEIST_ACCEL_DEBUG");
    return value != nullptr && value[0] != '\0' && strcmp(value, "0") != 0;
}

static void transformer_accel_debug(const char *reason) {
    if (transformer_accel_debug_enabled()) {
        fprintf(stderr, "transformer_accel: %s\n", reason);
    }
}

static void transformer_accel_debug_tensor(const char *name,
                                           const struct geist_tensor *t) {
    if (!transformer_accel_debug_enabled()) {
        return;
    }
    if (t == nullptr) {
        fprintf(stderr, "transformer_accel: unsupported %s: null tensor\n",
                name);
        return;
    }
    fprintf(stderr,
            "transformer_accel: unsupported %s: dtype=%d layout=%d ndim=%d shape=[%" PRId64 ",%" PRId64 "]\n",
            name, (int) t->dtype, (int) t->layout, t->ndim,
            t->shape[0], t->shape[1]);
}

static bool transformer_accel_model_is_target(
    const struct transformer_arch_state *state) {

    return state != nullptr &&
           state->config.family != nullptr &&
           strcmp(state->config.family, "gemma4") == 0 &&
           state->config.has_ple &&
           state->config.has_gemma_attn_norms &&
           !state->config.has_sub_ln &&
           state->config.ffn_activation == GEIST_FFN_GEGLU &&
           state->n_layers == GEIST_GEMMA4_NUM_LAYERS &&
           state->d_model == GEIST_GEMMA4_HIDDEN &&
           state->vocab_size == GEIST_GEMMA4_VOCAB &&
           state->n_q_heads == GEIST_GEMMA4_N_Q_HEADS &&
           state->n_kv_heads == GEIST_GEMMA4_N_KV_HEADS &&
           state->hidden_per_layer == GEIST_GEMMA4_HIDDEN_PER_LAYER &&
           state->ple_out == GEIST_GEMMA4_PLE_OUT;
}

static bool transformer_accel_tensor_is_f32_dense(
    const struct geist_tensor *t,
    int                       ndim) {

    return t != nullptr &&
           t->dtype == GEIST_DTYPE_F32 &&
           t->layout == GEIST_LAYOUT_DENSE &&
           t->ndim == ndim;
}

static bool transformer_accel_tensor_is_q4_or_q6(
    const struct geist_tensor *t) {

    return t != nullptr &&
           t->layout == GEIST_LAYOUT_BLOCK_QUANTIZED &&
           (t->dtype == GEIST_DTYPE_Q4_K || t->dtype == GEIST_DTYPE_Q6_K) &&
           t->ndim == 2;
}

static bool transformer_accel_norm_is_supported(
    const struct geist_tensor *t,
    int64_t                   n) {

    return transformer_accel_tensor_is_f32_dense(t, 1) &&
           t->shape[0] == n;
}

static bool transformer_accel_weight_is_supported(
    const struct geist_tensor *t,
    int64_t                   rows,
    int64_t                   cols) {

    return transformer_accel_tensor_is_q4_or_q6(t) &&
           t->shape[0] == rows &&
           t->shape[1] == cols;
}

static bool transformer_accel_embedding_weight_is_supported(
    const struct geist_tensor *t,
    int64_t                   rows,
    int64_t                   cols) {

    return t != nullptr &&
           t->layout == GEIST_LAYOUT_BLOCK_QUANTIZED &&
           (t->dtype == GEIST_DTYPE_Q4_K ||
            t->dtype == GEIST_DTYPE_Q5_K ||
            t->dtype == GEIST_DTYPE_Q6_K) &&
           t->ndim == 2 &&
           t->shape[0] == rows &&
           t->shape[1] == cols;
}

static bool transformer_accel_ple_weight_is_supported(
    const struct geist_tensor *t,
    int64_t                   rows,
    int64_t                   cols) {

    if (transformer_accel_tensor_is_f32_dense(t, 2) &&
        t->shape[0] == rows &&
        t->shape[1] == cols) {
        return true;
    }
    return transformer_accel_weight_is_supported(t, rows, cols);
}

static bool transformer_accel_layer_weights_are_supported(
    const struct transformer_arch_state    *state,
    const struct transformer_layer_weights *L) {

    if (state == nullptr || L == nullptr) {
        return false;
    }
    if (L->o_awq_inv_scale != nullptr || L->down_awq_inv_scale != nullptr) {
        return false;
    }
    if (L->attn_sub_norm.ndim != 0 || L->ffn_sub_norm.ndim != 0) {
        return false;
    }

    const int64_t hidden = (int64_t) state->d_model;
    const int64_t head_dim = (int64_t) L->head_dim;
    const int64_t q_out = (int64_t) L->q_out;
    const int64_t kv_out = (int64_t) L->kv_out;
    const int64_t intermediate = (int64_t) L->intermediate;
    const int64_t hidden_per_layer = (int64_t) state->hidden_per_layer;

    if (!transformer_accel_norm_is_supported(&L->attn_norm, hidden) ||
        !transformer_accel_norm_is_supported(&L->q_norm, head_dim) ||
        !transformer_accel_norm_is_supported(&L->post_attn_norm, hidden) ||
        !transformer_accel_norm_is_supported(&L->ffn_norm, hidden) ||
        !transformer_accel_norm_is_supported(&L->post_ffw_norm, hidden) ||
        !transformer_accel_norm_is_supported(&L->post_per_layer_norm, hidden)) {
        transformer_accel_debug("layer norm tensor unsupported");
        return false;
    }
    if (!L->is_kv_shared &&
        !transformer_accel_norm_is_supported(&L->k_norm, head_dim)) {
        transformer_accel_debug_tensor("layer.k_norm", &L->k_norm);
        return false;
    }

    if (!transformer_accel_weight_is_supported(&L->q_proj, q_out, hidden)) {
        transformer_accel_debug_tensor("layer.q_proj", &L->q_proj);
        return false;
    }
    if (!transformer_accel_weight_is_supported(&L->o_proj, hidden, q_out)) {
        transformer_accel_debug_tensor("layer.o_proj", &L->o_proj);
        return false;
    }
    if (!transformer_accel_weight_is_supported(&L->gate_proj,
                                               intermediate, hidden)) {
        transformer_accel_debug_tensor("layer.gate_proj", &L->gate_proj);
        return false;
    }
    if (!transformer_accel_weight_is_supported(&L->up_proj,
                                               intermediate, hidden)) {
        transformer_accel_debug_tensor("layer.up_proj", &L->up_proj);
        return false;
    }
    if (!transformer_accel_weight_is_supported(&L->down_proj,
                                               hidden, intermediate)) {
        transformer_accel_debug_tensor("layer.down_proj", &L->down_proj);
        return false;
    }
    if (!transformer_accel_ple_weight_is_supported(&L->per_layer_gate,
                                                   hidden_per_layer, hidden)) {
        transformer_accel_debug_tensor("layer.per_layer_gate",
                                       &L->per_layer_gate);
        return false;
    }
    if (!transformer_accel_ple_weight_is_supported(&L->per_layer_proj,
                                                   hidden, hidden_per_layer)) {
        transformer_accel_debug_tensor("layer.per_layer_proj",
                                       &L->per_layer_proj);
        return false;
    }
    if (!L->is_kv_shared &&
        (!transformer_accel_weight_is_supported(&L->k_proj, kv_out, hidden) ||
         !transformer_accel_weight_is_supported(&L->v_proj, kv_out, hidden))) {
        transformer_accel_debug("layer k/v projection unsupported");
        return false;
    }
    if (L->is_kv_shared &&
        (L->k_norm.ndim != 0 || L->k_proj.ndim != 0 || L->v_proj.ndim != 0)) {
        return false;
    }
    return true;
}

static bool transformer_accel_model_weights_are_supported(
    const struct transformer_arch_state *state) {

    if (state == nullptr || state->layers == nullptr) {
        return false;
    }
    const int64_t vocab = (int64_t) state->vocab_size;
    const int64_t hidden = (int64_t) state->d_model;
    const int64_t ple_out = (int64_t) state->ple_out;
    const int64_t hidden_per_layer = (int64_t) state->hidden_per_layer;

    if (!transformer_accel_embedding_weight_is_supported(&state->embed_table,
                                                         vocab, hidden)) {
        transformer_accel_debug_tensor("embed_table", &state->embed_table);
        return false;
    }
    if (!transformer_accel_embedding_weight_is_supported(&state->ple_table,
                                                         vocab, ple_out)) {
        transformer_accel_debug_tensor("ple_table", &state->ple_table);
        return false;
    }
    if (!transformer_accel_tensor_is_f32_dense(&state->model_proj, 2) ||
        state->model_proj.shape[0] != ple_out ||
        state->model_proj.shape[1] != hidden) {
        transformer_accel_debug_tensor("model_proj", &state->model_proj);
        return false;
    }
    if (!transformer_accel_norm_is_supported(&state->model_proj_norm,
                                             hidden_per_layer)) {
        transformer_accel_debug_tensor("model_proj_norm",
                                       &state->model_proj_norm);
        return false;
    }
    if (!transformer_accel_norm_is_supported(&state->output_norm, hidden)) {
        transformer_accel_debug_tensor("output_norm", &state->output_norm);
        return false;
    }

    for (size_t i = 0; i < state->n_layers; i++) {
        if (!transformer_accel_layer_weights_are_supported(state,
                                                           &state->layers[i])) {
            if (transformer_accel_debug_enabled()) {
                fprintf(stderr, "transformer_accel: unsupported layer %zu\n",
                        i);
            }
            return false;
        }
    }
    return true;
}

[[nodiscard]] static enum geist_status transformer_accel_prepare_weight(
    struct geist_backend *be,
    const struct geist_backend_vtbl *v,
    const struct geist_tensor *w) {

    if (v->prepare_weight_layout == nullptr || w == nullptr ||
        w->buffer == nullptr) {
        return GEIST_OK;
    }
    return v->prepare_weight_layout(be, w);
}

[[nodiscard]] static enum geist_status transformer_accel_prepare_layer_weights(
    struct geist_backend *be,
    const struct geist_backend_vtbl *v,
    const struct transformer_layer_weights *L) {

    enum geist_status s =
        transformer_accel_prepare_weight(be, v, &L->q_proj);
    if (s == GEIST_OK && !L->is_kv_shared) {
        s = transformer_accel_prepare_weight(be, v, &L->k_proj);
    }
    if (s == GEIST_OK && !L->is_kv_shared) {
        s = transformer_accel_prepare_weight(be, v, &L->v_proj);
    }
    if (s == GEIST_OK) {
        s = transformer_accel_prepare_weight(be, v, &L->o_proj);
    }
    if (s == GEIST_OK) {
        s = transformer_accel_prepare_weight(be, v, &L->gate_proj);
    }
    if (s == GEIST_OK) {
        s = transformer_accel_prepare_weight(be, v, &L->up_proj);
    }
    if (s == GEIST_OK) {
        s = transformer_accel_prepare_weight(be, v, &L->down_proj);
    }
    if (s == GEIST_OK) {
        s = transformer_accel_prepare_weight(be, v, &L->per_layer_gate);
    }
    if (s == GEIST_OK) {
        s = transformer_accel_prepare_weight(be, v, &L->per_layer_proj);
    }
    return s;
}

[[nodiscard]] static enum geist_status transformer_accel_prepare_weight_layouts(
    struct transformer_arch_state *state,
    const struct geist_backend_vtbl *v) {

    struct geist_backend *be = state->backend;
    enum geist_status s =
        transformer_accel_prepare_weight(be, v, &state->embed_table);
    if (s == GEIST_OK) {
        s = transformer_accel_prepare_weight(be, v, &state->ple_table);
    }
    for (size_t i = 0; s == GEIST_OK && i < state->n_layers; i++) {
        s = transformer_accel_prepare_layer_weights(be, v, &state->layers[i]);
    }
    return s;
}

static bool transformer_accel_caps_are_sufficient(
    const struct geist_backend_accel_caps *caps) {

    return caps != nullptr &&
           caps->device_resident_buffers &&
           caps->compute_queue &&
           caps->pipeline_cache;
}

static bool transformer_accel_backend_has_decode_greedy_path(
    const struct geist_backend_vtbl *v) {

    return v != nullptr &&
           v->buffer_copy != nullptr &&
           v->embedding_lookup_scaled != nullptr &&
           v->rmsnorm != nullptr &&
           v->matvec_f32_dense != nullptr &&
           v->matvec_q4k != nullptr &&
           v->matvec_q6k != nullptr &&
           v->add != nullptr &&
           v->scale_f32 != nullptr &&
           v->attention_block != nullptr &&
           v->attention_query_block != nullptr &&
           v->ffn_geglu_block != nullptr &&
           v->greedy_head != nullptr &&
           v->command_sequence_begin != nullptr &&
           v->command_sequence_end != nullptr &&
           v->command_sequence_read_token != nullptr;
}

enum geist_status
transformer_accel_try_create(struct transformer_arch_state *state,
                             struct transformer_accel    **out) {
    if (out == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    *out = nullptr;
    if (state == nullptr || state->backend == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    const struct geist_backend_vtbl *v = state->backend->desc->vtbl;
    if (v == nullptr || v->query_accel_caps == nullptr) {
        transformer_accel_debug("backend has no accel caps");
        return GEIST_OK;
    }
    if (!transformer_accel_backend_has_decode_greedy_path(v)) {
        transformer_accel_debug("backend missing decode greedy fastpath ops");
        return GEIST_OK;
    }
    if (!transformer_accel_model_is_target(state)) {
        transformer_accel_debug("model geometry is not Gemma4 E2B target");
        return GEIST_OK;
    }
    if (!transformer_accel_model_weights_are_supported(state)) {
        transformer_accel_debug("model weights are not supported by fastpath");
        return GEIST_OK;
    }

    struct geist_backend_accel_caps caps = {
        .struct_size = sizeof(caps),
    };
    enum geist_status s = v->query_accel_caps(state->backend, &caps);
    if (s != GEIST_OK) {
        return s;
    }
    if (!transformer_accel_caps_are_sufficient(&caps)) {
        transformer_accel_debug("backend caps are insufficient");
        return GEIST_OK;
    }

    s = transformer_accel_prepare_weight_layouts(state, v);
    if (s != GEIST_OK) {
        transformer_accel_debug("backend weight layout preparation failed");
        return s;
    }

    struct transformer_accel *accel =
        heap_alloc_aligned(sizeof(*accel), alignof(struct transformer_accel));
    if (accel == nullptr) {
        geist_backend_set_error(state->backend, GEIST_E_OOM,
                                "transformer_accel: %zu-byte alloc failed",
                                sizeof(*accel));
        return GEIST_E_OOM;
    }
    *accel = (struct transformer_accel){
        .backend = state->backend,
        .caps = caps,
        .decode_greedy_path = true,
        .prefill_text_path = true,
    };
    transformer_accel_debug("enabled greedy/prefill device fastpath");
    *out = accel;
    return GEIST_OK;
}

void transformer_accel_destroy(struct transformer_accel *accel) {
    if (accel == nullptr) {
        return;
    }
    void *p = accel;
    safe_free(&p);
}

enum geist_status
transformer_accel_session_create(struct transformer_accel          *accel,
                                 struct transformer_arch_session   *session,
                                 struct transformer_accel_session **out) {
    if (out == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    *out = nullptr;
    if (accel == nullptr) {
        return GEIST_OK;
    }
    if (session == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    if (session->temperature > 0.0f || session->top_k > 1 ||
        (session->top_p > 0.0f && session->top_p < 1.0f)) {
        return GEIST_OK;
    }
    if (session->kv_int8_enabled || session->kv_kivi_enabled ||
        !accel->decode_greedy_path) {
        return GEIST_OK;
    }

    struct transformer_accel_session *accel_session =
        heap_alloc_aligned(sizeof(*accel_session),
                           alignof(struct transformer_accel_session));
    if (accel_session == nullptr) {
        geist_backend_set_error(accel->backend, GEIST_E_OOM,
                                "transformer_accel_session: %zu-byte alloc failed",
                                sizeof(*accel_session));
        return GEIST_E_OOM;
    }
    *accel_session = (struct transformer_accel_session){
        .greedy_device_path = true,
        .prefill_text_device_path = accel->prefill_text_path,
        .verify_greedy_device_path = true,
    };
    *out = accel_session;
    return GEIST_OK;
}

void transformer_accel_session_destroy(struct transformer_accel         *accel,
                                       struct transformer_accel_session *session) {
    (void) accel;
    if (session == nullptr) {
        return;
    }
    void *p = session;
    safe_free(&p);
}

bool transformer_accel_session_decode_greedy_enabled(
    const struct transformer_accel_session *session) {

    return session != nullptr && session->greedy_device_path;
}

bool transformer_accel_session_prefill_text_enabled(
    const struct transformer_accel_session *session) {

    return session != nullptr && session->prefill_text_device_path;
}

bool transformer_accel_session_verify_greedy_enabled(
    const struct transformer_accel_session *session) {

    return session != nullptr && session->verify_greedy_device_path;
}
