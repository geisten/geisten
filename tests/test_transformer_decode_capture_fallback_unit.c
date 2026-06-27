/*
 * test_transformer_decode_capture_fallback_unit - decode capture fallback.
 *
 * A backend can be eligible for the Vulkan-style decode command sequence, yet
 * reject a specific captured fastpath before submission. The architecture must
 * discard that command sequence, restore logical session state, and retry the
 * ordinary decomposed path instead of surfacing GEIST_E_UNSUPPORTED.
 */
#include "test_helpers.h"

#define GEIST_INTERNAL_ARCH_LAYER
#define GEIST_INTERNAL_BACKEND_LAYER
#include "src/archs/transformer/accel.h"
#include "src/archs/transformer/arch_state.h"
#include "src/archs/transformer/forward.h"
#include "src/backends/cpu_scalar/internal.h"

#include <geist.h>
#include <geist_backend.h>
#include <geist_weight.h>

#include <stdio.h>
#include <string.h>

struct capture_backend_state {
    struct geist_backend *delegate;
    bool capture_active;
    int begin_count;
    int greedy_step_begin_count;
    int layer_loop_begin_count;
    int prefill_text_begin_count;
    int verify_greedy_begin_count;
    int discard_count;
    int submit_count;
    int read_count;
    int replay_count;
    geist_token_t replay_token_id;
    size_t replay_q_position;
    enum geist_status replay_status;
    geist_token_t replay_out_token;
    int captured_embedding_success_count;
    int captured_buffer_copy_count;
    int captured_greedy_unsupported_count;
    int captured_greedy_success_count;
    int captured_greedy_batch_success_count;
    int captured_map_count;
    int captured_map_null_count;
    bool null_map_during_capture;
    bool captured_token_ready;
    geist_token_t captured_token;
    geist_token_t captured_tokens[8];
    size_t captured_token_count;
};

static struct capture_backend_state *g_capture_map_state;

static int check(bool cond, const char *what) {
    if (!cond) {
        fprintf(stderr, "FAIL: %s\n", what);
        return 1;
    }
    return 0;
}

static struct geist_tensor make_weight_tensor(enum geist_dtype dtype,
                                              enum geist_layout layout,
                                              size_t rows,
                                              size_t cols) {
    return (struct geist_tensor){
        .dtype = dtype,
        .layout = layout,
        .ndim = 2,
        .shape = {(int64_t) rows, (int64_t) cols},
        .stride = {layout == GEIST_LAYOUT_DENSE ? (int64_t) cols : 0, 1},
    };
}

static struct geist_tensor make_norm_tensor(size_t n) {
    return (struct geist_tensor){
        .dtype = GEIST_DTYPE_F32,
        .layout = GEIST_LAYOUT_DENSE,
        .ndim = 1,
        .shape = {(int64_t) n},
        .stride = {1},
    };
}

static void fill_target_weight_metadata(struct transformer_arch_state *state) {
    state->embed_table = make_weight_tensor(
        GEIST_DTYPE_Q6_K, GEIST_LAYOUT_BLOCK_QUANTIZED,
        state->vocab_size, state->d_model);
    state->ple_table = make_weight_tensor(
        GEIST_DTYPE_Q6_K, GEIST_LAYOUT_BLOCK_QUANTIZED,
        state->vocab_size, state->ple_out);
    state->model_proj = make_weight_tensor(
        GEIST_DTYPE_F32, GEIST_LAYOUT_DENSE,
        state->ple_out, state->d_model);
    state->model_proj_norm = make_norm_tensor(state->hidden_per_layer);
    state->output_norm = make_norm_tensor(state->d_model);

    for (size_t i = 0; i < state->n_layers; i++) {
        struct transformer_layer_weights *L = &state->layers[i];
        *L = (struct transformer_layer_weights){
            .layer_idx = (int) i,
            .is_full = ((i % 5u) == 4u),
            .is_kv_shared = i >= 15u,
            .head_dim = ((i % 5u) == 4u) ? 512u : 256u,
            .intermediate = i >= 15u ? 12288u : 6144u,
        };
        L->q_out = state->n_q_heads * L->head_dim;
        L->kv_out = state->n_kv_heads * L->head_dim;
        L->attn_norm = make_norm_tensor(state->d_model);
        L->q_norm = make_norm_tensor(L->head_dim);
        L->k_norm = L->is_kv_shared ? (struct geist_tensor){0} :
            make_norm_tensor(L->head_dim);
        L->post_attn_norm = make_norm_tensor(state->d_model);
        L->ffn_norm = make_norm_tensor(state->d_model);
        L->post_ffw_norm = make_norm_tensor(state->d_model);
        L->post_per_layer_norm = make_norm_tensor(state->d_model);
        L->q_proj = make_weight_tensor(
            GEIST_DTYPE_Q4_K, GEIST_LAYOUT_BLOCK_QUANTIZED,
            L->q_out, state->d_model);
        L->o_proj = make_weight_tensor(
            GEIST_DTYPE_Q4_K, GEIST_LAYOUT_BLOCK_QUANTIZED,
            state->d_model, L->q_out);
        L->k_proj = L->is_kv_shared ? (struct geist_tensor){0} :
            make_weight_tensor(GEIST_DTYPE_Q4_K,
                               GEIST_LAYOUT_BLOCK_QUANTIZED,
                               L->kv_out, state->d_model);
        L->v_proj = L->is_kv_shared ? (struct geist_tensor){0} :
            make_weight_tensor(GEIST_DTYPE_Q4_K,
                               GEIST_LAYOUT_BLOCK_QUANTIZED,
                               L->kv_out, state->d_model);
        L->gate_proj = make_weight_tensor(
            GEIST_DTYPE_Q4_K, GEIST_LAYOUT_BLOCK_QUANTIZED,
            L->intermediate, state->d_model);
        L->up_proj = make_weight_tensor(
            GEIST_DTYPE_Q4_K, GEIST_LAYOUT_BLOCK_QUANTIZED,
            L->intermediate, state->d_model);
        L->down_proj = make_weight_tensor(
            GEIST_DTYPE_Q4_K, GEIST_LAYOUT_BLOCK_QUANTIZED,
            state->d_model, L->intermediate);
        L->per_layer_gate = make_weight_tensor(
            GEIST_DTYPE_Q4_K, GEIST_LAYOUT_BLOCK_QUANTIZED,
            state->hidden_per_layer, state->d_model);
        L->per_layer_proj = make_weight_tensor(
            GEIST_DTYPE_Q4_K, GEIST_LAYOUT_BLOCK_QUANTIZED,
            state->d_model, state->hidden_per_layer);
    }
}

static struct capture_backend_state *capture_state(struct geist_backend *be) {
    return be != nullptr ? (struct capture_backend_state *) be->state : nullptr;
}

static const struct geist_backend_vtbl *delegate_vtbl(
    const struct capture_backend_state *st) {
    return st->delegate->desc->vtbl;
}

static enum geist_status capture_query_accel_caps(
    struct geist_backend *be,
    struct geist_backend_accel_caps *out) {
    (void) be;
    if (out == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    *out = (struct geist_backend_accel_caps){
        .struct_size = sizeof(*out),
        .device_resident_buffers = true,
        .compute_queue = true,
        .pipeline_cache = true,
        .subgroup_basic = true,
        .shader_integer_dot_product = true,
        .descriptor_indexing = true,
        .timeline_semaphore = true,
    };
    return GEIST_OK;
}

static enum geist_status capture_buffer_create(
    struct geist_backend *be,
    size_t bytes,
    enum geist_buffer_role role,
    unsigned int memory_flags,
    struct geist_buffer **out) {
    struct capture_backend_state *st = capture_state(be);
    return delegate_vtbl(st)->buffer_create(st->delegate, bytes, role,
                                            memory_flags, out);
}

static void capture_buffer_destroy(struct geist_backend *be,
                                   struct geist_buffer *buf) {
    struct capture_backend_state *st = capture_state(be);
    delegate_vtbl(st)->buffer_destroy(st->delegate, buf);
}

static enum geist_status capture_buffer_upload(
    struct geist_buffer *buf,
    size_t n_bytes,
    const uint8_t src[static n_bytes]) {
    if (buf == nullptr || src == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    if (n_bytes > buf->bytes) {
        return GEIST_E_INVALID_ARG;
    }
    memcpy(buf->host, src, n_bytes);
    return GEIST_OK;
}

static enum geist_status capture_buffer_download(
    size_t n_bytes,
    uint8_t dst[static n_bytes],
    const struct geist_buffer *buf) {
    if (buf == nullptr || dst == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    if (n_bytes > buf->bytes) {
        return GEIST_E_INVALID_ARG;
    }
    memcpy(dst, buf->host, n_bytes);
    return GEIST_OK;
}

static enum geist_status capture_buffer_copy(
    struct geist_buffer *dst,
    size_t dst_offset,
    const struct geist_buffer *src,
    size_t src_offset,
    size_t n_bytes) {
    if (dst == nullptr || src == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    if (g_capture_map_state != nullptr &&
        g_capture_map_state->capture_active) {
        g_capture_map_state->captured_buffer_copy_count++;
    }
    if (dst_offset > dst->bytes || src_offset > src->bytes ||
        n_bytes > dst->bytes - dst_offset ||
        n_bytes > src->bytes - src_offset) {
        return GEIST_E_INVALID_ARG;
    }
    memmove((uint8_t *) dst->host + dst_offset,
            (const uint8_t *) src->host + src_offset,
            n_bytes);
    return GEIST_OK;
}

static void *capture_buffer_map(struct geist_buffer *buf) {
    if (g_capture_map_state != nullptr &&
        g_capture_map_state->capture_active) {
        g_capture_map_state->captured_map_count++;
        if (g_capture_map_state->null_map_during_capture) {
            g_capture_map_state->captured_map_null_count++;
            return nullptr;
        }
    }
    return buf != nullptr ? buf->host : nullptr;
}

static void capture_buffer_unmap(struct geist_buffer *buf) {
    (void) buf;
}

static enum geist_status capture_rmsnorm(
    struct geist_backend *be,
    const struct geist_tensor *x,
    const struct geist_tensor *w,
    float eps,
    struct geist_tensor *y) {
    struct capture_backend_state *st = capture_state(be);
    return delegate_vtbl(st)->rmsnorm(st->delegate, x, w, eps, y);
}

static enum geist_status capture_embedding_lookup(
    struct geist_backend *be,
    const struct geist_tensor *embed_table,
    geist_token_t token_id,
    struct geist_tensor *out) {
    struct capture_backend_state *st = capture_state(be);
    return delegate_vtbl(st)->embedding_lookup(st->delegate, embed_table,
                                               token_id, out);
}

static enum geist_status capture_fake_embedding_lookup_scaled(
    struct geist_backend *be,
    const struct geist_tensor *embed_table,
    geist_token_t token_id,
    float scale,
    struct geist_tensor *out) {
    (void) be;
    (void) embed_table;
    (void) token_id;
    (void) scale;
    (void) out;
    return GEIST_E_UNSUPPORTED;
}

static enum geist_status capture_success_embedding_lookup_scaled(
    struct geist_backend *be,
    const struct geist_tensor *embed_table,
    geist_token_t token_id,
    float scale,
    struct geist_tensor *out) {
    (void) be;
    (void) embed_table;
    (void) token_id;
    (void) scale;
    (void) out;
    struct capture_backend_state *st = capture_state(be);
    if (st != nullptr && st->capture_active) {
        st->captured_embedding_success_count++;
    }
    return GEIST_OK;
}

static enum geist_status capture_success_embedding_lookup(
    struct geist_backend *be,
    const struct geist_tensor *embed_table,
    geist_token_t token_id,
    struct geist_tensor *out) {
    (void) be;
    (void) embed_table;
    (void) token_id;
    (void) out;
    struct capture_backend_state *st = capture_state(be);
    if (st != nullptr && st->capture_active) {
        st->captured_embedding_success_count++;
    }
    return GEIST_OK;
}

static enum geist_status capture_fake_matvec(
    struct geist_backend *be,
    const struct geist_tensor *x,
    const struct geist_tensor *w,
    struct geist_tensor *y) {
    (void) be;
    (void) x;
    (void) w;
    (void) y;
    return GEIST_E_UNSUPPORTED;
}

static enum geist_status capture_fake_add(
    struct geist_backend *be,
    const struct geist_tensor *a,
    const struct geist_tensor *b,
    struct geist_tensor *y) {
    (void) be;
    (void) a;
    (void) b;
    (void) y;
    return GEIST_E_UNSUPPORTED;
}

static enum geist_status capture_fake_scale_f32(
    struct geist_backend *be,
    const struct geist_tensor *x,
    float scale,
    struct geist_tensor *y) {
    (void) be;
    (void) x;
    (void) scale;
    (void) y;
    return GEIST_E_UNSUPPORTED;
}

static enum geist_status capture_fake_attention_block(
    struct geist_backend *be,
    const struct geist_backend_attention_block *block) {
    (void) be;
    (void) block;
    return GEIST_E_UNSUPPORTED;
}

static enum geist_status capture_fake_attention_query_block(
    struct geist_backend *be,
    const struct geist_backend_attention_query_block *block) {
    (void) be;
    (void) block;
    return GEIST_E_UNSUPPORTED;
}

static enum geist_status capture_fake_ffn_geglu_block(
    struct geist_backend *be,
    const struct geist_backend_ffn_geglu_block *block) {
    (void) be;
    (void) block;
    return GEIST_E_UNSUPPORTED;
}

static enum geist_status capture_greedy_head(
    struct geist_backend *be,
    const struct geist_backend_greedy_head *head,
    geist_token_t *out_token) {
    (void) head;
    if (out_token != nullptr) {
        *out_token = -1;
    }
    struct capture_backend_state *st = capture_state(be);
    if (st->capture_active) {
        st->captured_greedy_unsupported_count++;
    }
    return GEIST_E_UNSUPPORTED;
}

static enum geist_status capture_success_greedy_head(
    struct geist_backend *be,
    const struct geist_backend_greedy_head *head,
    geist_token_t *out_token) {
    (void) head;
    struct capture_backend_state *st = capture_state(be);
    if (st == nullptr || out_token == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    if (st->capture_active) {
        st->captured_greedy_success_count++;
        if (head->token_output_offset >=
            sizeof(st->captured_tokens) / sizeof(st->captured_tokens[0])) {
            return GEIST_E_INVALID_ARG;
        }
        st->captured_token = 1;
        st->captured_tokens[head->token_output_offset] = 1;
        if (head->token_output_offset + 1 > st->captured_token_count) {
            st->captured_token_count = head->token_output_offset + 1;
        }
        st->captured_token_ready = true;
        *out_token = -1;
        return GEIST_OK;
    }
    *out_token = 1;
    return GEIST_OK;
}

static enum geist_status capture_success_greedy_head_batch(
    struct geist_backend *be,
    const struct geist_backend_greedy_head_batch *head,
    geist_token_t out_tokens[static head->row_count]) {
    struct capture_backend_state *st = capture_state(be);
    if (st == nullptr || head == nullptr || out_tokens == nullptr ||
        head->row_count == 0 ||
        head->token_output_offset >
            sizeof(st->captured_tokens) / sizeof(st->captured_tokens[0]) ||
        head->row_count >
            sizeof(st->captured_tokens) / sizeof(st->captured_tokens[0]) -
                head->token_output_offset) {
        return GEIST_E_INVALID_ARG;
    }
    if (st->capture_active) {
        st->captured_greedy_batch_success_count++;
        for (size_t row = 0; row < head->row_count; row++) {
            st->captured_tokens[head->token_output_offset + row] = 1;
            out_tokens[row] = -1;
        }
        st->captured_token = 1;
        st->captured_token_count =
            head->token_output_offset + head->row_count;
        st->captured_token_ready = true;
        return GEIST_OK;
    }
    for (size_t row = 0; row < head->row_count; row++) {
        out_tokens[row] = 1;
    }
    return GEIST_OK;
}

static enum geist_status capture_command_sequence_begin(
    struct geist_backend *be,
    enum geist_command_sequence_kind kind,
    int *out_token) {
    struct capture_backend_state *st = capture_state(be);
    if (st == nullptr || st->capture_active || out_token == nullptr) {
        return GEIST_E_INVALID_STATE;
    }
    st->capture_active = true;
    st->begin_count++;
    if (kind == GEIST_COMMAND_SEQUENCE_DECODE_GREEDY_STEP) {
        st->greedy_step_begin_count++;
    } else if (kind == GEIST_COMMAND_SEQUENCE_DECODE_LAYER_LOOP) {
        st->layer_loop_begin_count++;
    } else if (kind == GEIST_COMMAND_SEQUENCE_PREFILL_TEXT) {
        st->prefill_text_begin_count++;
    } else if (kind == GEIST_COMMAND_SEQUENCE_VERIFY_GREEDY) {
        st->verify_greedy_begin_count++;
    }
    *out_token = 17;
    return GEIST_OK;
}

static enum geist_status capture_command_sequence_end(
    struct geist_backend *be,
    int token,
    bool submit) {
    struct capture_backend_state *st = capture_state(be);
    if (st == nullptr || !st->capture_active || token != 17) {
        return GEIST_E_INVALID_STATE;
    }
    st->capture_active = false;
    if (submit) {
        st->submit_count++;
    } else {
        st->discard_count++;
    }
    return GEIST_OK;
}

static enum geist_status capture_command_sequence_read_token(
    struct geist_backend *be,
    geist_token_t *out_token) {
    struct capture_backend_state *st = capture_state(be);
    st->read_count++;
    if (st != nullptr && !st->capture_active && st->captured_token_ready) {
        if (out_token != nullptr) {
            *out_token = st->captured_token;
        }
        st->captured_token_ready = false;
        return GEIST_OK;
    }
    if (out_token != nullptr) {
        *out_token = -1;
    }
    return GEIST_E_INVALID_STATE;
}

static enum geist_status capture_command_sequence_read_tokens(
    struct geist_backend *be,
    size_t n,
    geist_token_t out_tokens[static n]) {
    struct capture_backend_state *st = capture_state(be);
    st->read_count++;
    if (st != nullptr && !st->capture_active && st->captured_token_ready &&
        out_tokens != nullptr && n <= st->captured_token_count) {
        for (size_t i = 0; i < n; i++) {
            out_tokens[i] = st->captured_tokens[i];
        }
        st->captured_token_ready = false;
        st->captured_token_count = 0;
        return GEIST_OK;
    }
    return GEIST_E_INVALID_STATE;
}

static enum geist_status capture_command_sequence_replay_decode_greedy_step(
    struct geist_backend *be,
    geist_token_t token_id,
    size_t q_position,
    geist_token_t *out_token) {
    struct capture_backend_state *st = capture_state(be);
    if (st == nullptr || out_token == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    st->replay_count++;
    st->replay_token_id = token_id;
    st->replay_q_position = q_position;
    if (st->replay_status != GEIST_OK) {
        return st->replay_status;
    }
    *out_token = st->replay_out_token;
    return GEIST_OK;
}

static void dense_linear_m1(const float *x,
                            const struct geist_weight *w,
                            struct geist_backend *be,
                            float *y) {
    (void) be;
    const float *raw = (const float *) w->raw;
    for (int32_t row = 0; row < w->n_out; row++) {
        float acc = 0.0f;
        for (int32_t col = 0; col < w->n_in; col++) {
            acc += x[col] * raw[(size_t) row * (size_t) w->n_in +
                                (size_t) col];
        }
        y[row] = acc;
    }
}

static void dense_linear_mN(const float *x,
                            const struct geist_weight *w,
                            size_t m,
                            struct geist_backend *be,
                            float *y) {
    (void) be;
    const float *raw = (const float *) w->raw;
    for (size_t row_idx = 0; row_idx < m; row_idx++) {
        const float *x_row = x + row_idx * (size_t) w->n_in;
        float *y_row = y + row_idx * (size_t) w->n_out;
        for (int32_t row = 0; row < w->n_out; row++) {
            float acc = 0.0f;
            for (int32_t col = 0; col < w->n_in; col++) {
                acc += x_row[col] * raw[(size_t) row * (size_t) w->n_in +
                                         (size_t) col];
            }
            y_row[row] = acc;
        }
    }
}

static enum geist_status delegate_upload(struct geist_backend *delegate,
                                         struct geist_buffer *buf,
                                         size_t bytes,
                                         const void *src) {
    return delegate->desc->vtbl->buffer_upload(buf, bytes,
                                               (const uint8_t *) src);
}

int main(void) {
    int fails = 0;
    struct geist_backend *cpu = nullptr;
    enum geist_status s =
        geist_backend_create("cpu_scalar", nullptr, nullptr, &cpu);
    fails += check(s == GEIST_OK, "cpu_scalar backend create OK");
    if (s != GEIST_OK || cpu == nullptr) {
        return GEIST_TEST_FAIL;
    }

    const struct geist_backend_vtbl eligibility_vtbl = {
        .query_accel_caps = capture_query_accel_caps,
        .buffer_create = capture_buffer_create,
        .buffer_destroy = capture_buffer_destroy,
        .buffer_upload = capture_buffer_upload,
        .buffer_download = capture_buffer_download,
        .buffer_copy = capture_buffer_copy,
        .buffer_map = capture_buffer_map,
        .buffer_unmap = capture_buffer_unmap,
        .embedding_lookup_scaled = capture_fake_embedding_lookup_scaled,
        .rmsnorm = capture_rmsnorm,
        .matvec_f32_dense = capture_fake_matvec,
        .matvec_q4k = capture_fake_matvec,
        .matvec_q6k = capture_fake_matvec,
        .add = capture_fake_add,
        .scale_f32 = capture_fake_scale_f32,
        .attention_block = capture_fake_attention_block,
        .attention_query_block = capture_fake_attention_query_block,
        .ffn_geglu_block = capture_fake_ffn_geglu_block,
        .greedy_head = capture_greedy_head,
        .command_sequence_begin = capture_command_sequence_begin,
        .command_sequence_end = capture_command_sequence_end,
        .command_sequence_read_token = capture_command_sequence_read_token,
        .command_sequence_read_tokens = capture_command_sequence_read_tokens,
    };
    const struct geist_backend_vtbl runtime_vtbl = {
        .buffer_create = capture_buffer_create,
        .buffer_destroy = capture_buffer_destroy,
        .buffer_upload = capture_buffer_upload,
        .buffer_download = capture_buffer_download,
        .buffer_copy = capture_buffer_copy,
        .buffer_map = capture_buffer_map,
        .buffer_unmap = capture_buffer_unmap,
        .embedding_lookup = capture_embedding_lookup,
        .rmsnorm = capture_rmsnorm,
        .greedy_head = capture_greedy_head,
        .command_sequence_begin = capture_command_sequence_begin,
        .command_sequence_end = capture_command_sequence_end,
        .command_sequence_read_token = capture_command_sequence_read_token,
        .command_sequence_read_tokens = capture_command_sequence_read_tokens,
    };
    const struct geist_backend_vtbl runtime_no_embed_vtbl = {
        .buffer_create = capture_buffer_create,
        .buffer_destroy = capture_buffer_destroy,
        .buffer_upload = capture_buffer_upload,
        .buffer_download = capture_buffer_download,
        .buffer_copy = capture_buffer_copy,
        .buffer_map = capture_buffer_map,
        .buffer_unmap = capture_buffer_unmap,
        .rmsnorm = capture_rmsnorm,
        .greedy_head = capture_greedy_head,
        .command_sequence_begin = capture_command_sequence_begin,
        .command_sequence_end = capture_command_sequence_end,
        .command_sequence_read_token = capture_command_sequence_read_token,
        .command_sequence_read_tokens = capture_command_sequence_read_tokens,
    };
    const struct geist_backend_vtbl runtime_success_vtbl = {
        .buffer_create = capture_buffer_create,
        .buffer_destroy = capture_buffer_destroy,
        .buffer_upload = capture_buffer_upload,
        .buffer_download = capture_buffer_download,
        .buffer_copy = capture_buffer_copy,
        .buffer_map = capture_buffer_map,
        .buffer_unmap = capture_buffer_unmap,
        .embedding_lookup = capture_success_embedding_lookup,
        .embedding_lookup_scaled = capture_success_embedding_lookup_scaled,
        .rmsnorm = capture_rmsnorm,
        .greedy_head = capture_success_greedy_head,
        .greedy_head_batch = capture_success_greedy_head_batch,
        .command_sequence_begin = capture_command_sequence_begin,
        .command_sequence_end = capture_command_sequence_end,
        .command_sequence_read_token = capture_command_sequence_read_token,
        .command_sequence_read_tokens = capture_command_sequence_read_tokens,
    };
    const struct geist_backend_vtbl runtime_replay_vtbl = {
        .buffer_create = capture_buffer_create,
        .buffer_destroy = capture_buffer_destroy,
        .buffer_upload = capture_buffer_upload,
        .buffer_download = capture_buffer_download,
        .buffer_copy = capture_buffer_copy,
        .buffer_map = capture_buffer_map,
        .buffer_unmap = capture_buffer_unmap,
        .embedding_lookup = capture_success_embedding_lookup,
        .embedding_lookup_scaled = capture_success_embedding_lookup_scaled,
        .rmsnorm = capture_rmsnorm,
        .greedy_head = capture_success_greedy_head,
        .greedy_head_batch = capture_success_greedy_head_batch,
        .command_sequence_begin = capture_command_sequence_begin,
        .command_sequence_end = capture_command_sequence_end,
        .command_sequence_read_token = capture_command_sequence_read_token,
        .command_sequence_read_tokens = capture_command_sequence_read_tokens,
        .command_sequence_replay_decode_greedy_step =
            capture_command_sequence_replay_decode_greedy_step,
    };
    const struct geist_backend_descriptor eligibility_desc = {
        .name = "capture_fallback_eligibility",
        .vtbl = &eligibility_vtbl,
    };
    const struct geist_backend_descriptor runtime_desc = {
        .name = "capture_fallback_runtime",
        .vtbl = &runtime_vtbl,
    };
    const struct geist_backend_descriptor runtime_no_embed_desc = {
        .name = "capture_fallback_runtime_no_embed",
        .vtbl = &runtime_no_embed_vtbl,
    };
    const struct geist_backend_descriptor runtime_success_desc = {
        .name = "capture_success_runtime",
        .vtbl = &runtime_success_vtbl,
    };
    const struct geist_backend_descriptor runtime_replay_desc = {
        .name = "capture_replay_runtime",
        .vtbl = &runtime_replay_vtbl,
    };
    struct capture_backend_state cap_state = {
        .delegate = cpu,
    };
    g_capture_map_state = &cap_state;
    struct geist_backend wrapper = {
        .desc = &eligibility_desc,
        .state = &cap_state,
    };

    struct transformer_layer_weights target_layers[GEIST_GEMMA4_NUM_LAYERS] = {0};
    struct transformer_arch_state state = {
        .backend = &wrapper,
        .config = {
            .family = "gemma4",
            .has_ple = true,
            .has_gemma_attn_norms = true,
            .ffn_activation = GEIST_FFN_GEGLU,
        },
        .n_layers = GEIST_GEMMA4_NUM_LAYERS,
        .d_model = GEIST_GEMMA4_HIDDEN,
        .vocab_size = GEIST_GEMMA4_VOCAB,
        .n_q_heads = GEIST_GEMMA4_N_Q_HEADS,
        .n_kv_heads = GEIST_GEMMA4_N_KV_HEADS,
        .hidden_per_layer = GEIST_GEMMA4_HIDDEN_PER_LAYER,
        .ple_out = GEIST_GEMMA4_PLE_OUT,
        .layers = target_layers,
    };
    fill_target_weight_metadata(&state);
    struct transformer_accel *accel = nullptr;
    s = transformer_accel_try_create(&state, &accel);
    fails += check(s == GEIST_OK, "synthetic accelerator create OK");
    fails += check(accel != nullptr, "synthetic accelerator eligible");

    struct transformer_arch_session sess = {
        .temperature = 0.0f,
        .top_p = 1.0f,
        .top_k = 0,
    };
    if (s == GEIST_OK && accel != nullptr) {
        s = transformer_accel_session_create(accel, &sess,
                                             &sess.accel_session);
        fails += check(s == GEIST_OK,
                       "synthetic accel session create OK");
        fails += check(sess.accel_session != nullptr,
                       "synthetic accel session handle non-null");
        fails += check(transformer_accel_session_prefill_text_enabled(
                           sess.accel_session),
                       "synthetic accel session enables prefill capture");
    }

    struct geist_buffer *embed_buf = nullptr;
    struct geist_buffer *norm_buf = nullptr;
    struct geist_buffer *scratch_h_a = nullptr;
    struct geist_buffer *scratch_h_b = nullptr;
    struct geist_buffer *logits_buf = nullptr;
    const struct geist_backend_vtbl *cpu_v = cpu->desc->vtbl;
    if (s == GEIST_OK) {
        s = cpu_v->buffer_create(cpu, 4u * sizeof(float), GEIST_BUFFER_WEIGHT,
                                 GEIST_MEMORY_HOST_VISIBLE, &embed_buf);
    }
    if (s == GEIST_OK) {
        s = cpu_v->buffer_create(cpu, 2u * sizeof(float), GEIST_BUFFER_WEIGHT,
                                 GEIST_MEMORY_HOST_VISIBLE, &norm_buf);
    }
    if (s == GEIST_OK) {
        s = cpu_v->buffer_create(cpu, 4u * sizeof(float),
                                 GEIST_BUFFER_ACTIVATION,
                                 GEIST_MEMORY_HOST_VISIBLE, &scratch_h_a);
    }
    if (s == GEIST_OK) {
        s = cpu_v->buffer_create(cpu, 4u * sizeof(float),
                                 GEIST_BUFFER_ACTIVATION,
                                 GEIST_MEMORY_HOST_VISIBLE, &scratch_h_b);
    }
    if (s == GEIST_OK) {
        s = cpu_v->buffer_create(cpu, 2u * sizeof(float),
                                 GEIST_BUFFER_ACTIVATION,
                                 GEIST_MEMORY_HOST_VISIBLE, &logits_buf);
    }
    const float embed_values[4] = {
        1.0f, 0.0f,
        0.0f, 2.0f,
    };
    const float norm_values[2] = {1.0f, 1.0f};
    if (s == GEIST_OK) {
        s = delegate_upload(cpu, embed_buf, sizeof(embed_values),
                            embed_values);
    }
    if (s == GEIST_OK) {
        s = delegate_upload(cpu, norm_buf, sizeof(norm_values), norm_values);
    }
    fails += check(s == GEIST_OK, "synthetic buffers allocate/upload OK");

    state = (struct transformer_arch_state){
        .backend = &wrapper,
        .sess = &sess,
        .config = {
            .family = "unit",
            .has_ple = false,
            .rms_eps = 1.0e-6f,
            .logit_softcap = 0.0f,
        },
        .n_layers = 0,
        .d_model = 2,
        .vocab_size = 2,
        .m_max = 2,
        .embed_table = {
            .buffer = embed_buf,
            .offset = 0,
            .dtype = GEIST_DTYPE_F32,
            .layout = GEIST_LAYOUT_DENSE,
            .ndim = 2,
            .shape = {2, 2},
            .stride = {2, 1},
        },
        .embed_table_w = {
            .raw = embed_values,
            .n_in = 2,
            .n_out = 2,
            .dtype = (uint16_t) GEIST_DTYPE_F32,
            .linear_m1 = dense_linear_m1,
            .linear_mN = dense_linear_mN,
        },
        .output_norm = {
            .buffer = norm_buf,
            .offset = 0,
            .dtype = GEIST_DTYPE_F32,
            .layout = GEIST_LAYOUT_DENSE,
            .ndim = 1,
            .shape = {2},
            .stride = {1},
        },
    };
    sess.scratch_h_a = scratch_h_a;
    sess.scratch_h_b = scratch_h_b;
    sess.scratch_logits = logits_buf;
    sess.kv_len = 3;
    sess.logits_valid = false;
    sess.logits_on_device = false;
    sess.logits_host_valid = false;
    sess.next_token_pending = -1;
    wrapper.desc = &runtime_desc;

    cap_state.begin_count = 0;
    cap_state.greedy_step_begin_count = 0;
    cap_state.layer_loop_begin_count = 0;
    cap_state.prefill_text_begin_count = 0;
    cap_state.verify_greedy_begin_count = 0;
    cap_state.discard_count = 0;
    cap_state.submit_count = 0;
    cap_state.read_count = 0;
    cap_state.captured_embedding_success_count = 0;
    cap_state.captured_buffer_copy_count = 0;
    cap_state.captured_greedy_unsupported_count = 0;
    cap_state.captured_greedy_success_count = 0;
    cap_state.captured_greedy_batch_success_count = 0;
    cap_state.captured_map_count = 0;
    cap_state.captured_map_null_count = 0;
    cap_state.null_map_during_capture = false;
    cap_state.captured_token_ready = false;
    cap_state.captured_token = -1;
    cap_state.captured_token_count = 0;
    sess.kv_len = 0;
    sess.logits_valid = false;
    sess.logits_on_device = false;
    sess.logits_host_valid = false;
    sess.next_token_pending = -1;
    wrapper.desc = &runtime_no_embed_desc;
    const geist_token_t prefill_ids[2] = {0, 1};
    if (s == GEIST_OK) {
        s = transformer_prefill_text_batch(&state, 2, prefill_ids);
    }
    fails += check(s == GEIST_OK,
                   "prefill retries without capture after missing device embed");
    fails += check(sess.kv_len == 2,
                   "prefill fallback advances kv_len by chunk");
    fails += check(sess.next_token_pending == 1,
                   "prefill fallback stores pending token");
    fails += check(sess.logits_valid && sess.logits_host_valid,
                   "prefill fallback leaves host logits valid");
    fails += check(!sess.logits_on_device,
                   "prefill fallback does not claim device logits");
    fails += check(cap_state.begin_count == 1,
                   "prefill capture attempted exactly once");
    fails += check(cap_state.prefill_text_begin_count == 1,
                   "prefill capture uses prefill command sequence");
    fails += check(cap_state.verify_greedy_begin_count == 0,
                   "prefill capture avoids verify command sequence");
    fails += check(cap_state.greedy_step_begin_count == 0,
                   "prefill capture avoids decode-greedy sequence");
    fails += check(cap_state.layer_loop_begin_count == 0,
                   "prefill capture avoids nested layer sequence");
    fails += check(cap_state.discard_count == 1,
                   "prefill missing device embed capture is discarded");
    fails += check(cap_state.submit_count == 0,
                   "prefill missing device embed capture is not submitted");
    fails += check(cap_state.read_count == 0,
                   "discarded prefill capture does not read token");
    fails += check(cap_state.captured_map_count == 0,
                   "captured prefill missing device embed does not map buffers");

    cap_state.begin_count = 0;
    cap_state.greedy_step_begin_count = 0;
    cap_state.layer_loop_begin_count = 0;
    cap_state.prefill_text_begin_count = 0;
    cap_state.verify_greedy_begin_count = 0;
    cap_state.discard_count = 0;
    cap_state.submit_count = 0;
    cap_state.read_count = 0;
    cap_state.captured_embedding_success_count = 0;
    cap_state.captured_buffer_copy_count = 0;
    cap_state.captured_greedy_unsupported_count = 0;
    cap_state.captured_greedy_success_count = 0;
    cap_state.captured_greedy_batch_success_count = 0;
    cap_state.captured_map_count = 0;
    cap_state.captured_map_null_count = 0;
    cap_state.null_map_during_capture = false;
    cap_state.captured_token_ready = false;
    cap_state.captured_token = -1;
    cap_state.captured_token_count = 0;
    sess.kv_len = 2;
    sess.logits_valid = false;
    sess.logits_on_device = false;
    sess.logits_host_valid = false;
    sess.next_token_pending = -1;
    wrapper.desc = &runtime_success_desc;
    if (s == GEIST_OK) {
        s = transformer_prefill_text_batch(&state, 2, prefill_ids);
    }
    fails += check(s == GEIST_OK,
                   "successful prefill capture returns OK");
    fails += check(sess.kv_len == 4,
                   "successful prefill capture advances kv_len by chunk");
    fails += check(sess.next_token_pending == 1,
                   "successful prefill capture stores pending token");
    fails += check(sess.logits_valid,
                   "successful prefill capture marks logits valid");
    fails += check(sess.logits_on_device,
                   "successful prefill capture marks logits device-resident");
    fails += check(!sess.logits_host_valid,
                   "successful prefill capture does not claim host logits");
    fails += check(cap_state.begin_count == 1,
                   "successful prefill capture begins exactly once");
    fails += check(cap_state.prefill_text_begin_count == 1,
                   "successful prefill capture uses prefill command sequence");
    fails += check(cap_state.verify_greedy_begin_count == 0,
                   "successful prefill capture avoids verify command sequence");
    fails += check(cap_state.greedy_step_begin_count == 0,
                   "successful prefill capture avoids decode-greedy sequence");
    fails += check(cap_state.layer_loop_begin_count == 0,
                   "successful prefill capture avoids nested layer sequence");
    fails += check(cap_state.submit_count == 1,
                   "successful prefill capture submits exactly once");
    fails += check(cap_state.discard_count == 0,
                   "successful prefill capture is not discarded");
    fails += check(cap_state.read_count == 1,
                   "successful prefill capture reads token once");
    fails += check(cap_state.captured_embedding_success_count == 2,
                   "successful prefill capture records device embeddings");
    fails += check(cap_state.captured_buffer_copy_count == 1,
                   "successful prefill capture avoids redundant layer-loop seed copy");
    fails += check(cap_state.captured_map_count == 0,
                   "successful prefill capture does not map buffers");
    fails += check(cap_state.captured_greedy_success_count == 1,
                   "successful prefill capture records greedy_head once");

    cap_state.begin_count = 0;
    cap_state.greedy_step_begin_count = 0;
    cap_state.layer_loop_begin_count = 0;
    cap_state.prefill_text_begin_count = 0;
    cap_state.verify_greedy_begin_count = 0;
    cap_state.discard_count = 0;
    cap_state.submit_count = 0;
    cap_state.read_count = 0;
    cap_state.captured_embedding_success_count = 0;
    cap_state.captured_buffer_copy_count = 0;
    cap_state.captured_greedy_unsupported_count = 0;
    cap_state.captured_greedy_success_count = 0;
    cap_state.captured_greedy_batch_success_count = 0;
    cap_state.captured_map_count = 0;
    cap_state.captured_map_null_count = 0;
    cap_state.null_map_during_capture = false;
    cap_state.captured_token_ready = false;
    cap_state.captured_token = -1;
    sess.kv_len = 4;
    sess.logits_valid = true;
    sess.logits_on_device = true;
    sess.logits_host_valid = false;
    sess.next_token_pending = 7;
    wrapper.desc = &runtime_no_embed_desc;

    const geist_token_t verify_ids[1] = {1};
    geist_token_t verify_out[1] = {-1};
    if (s == GEIST_OK) {
        s = transformer_verify_forward(&state, 1, verify_ids, verify_out);
    }
    fails += check(s == GEIST_OK,
                   "verify retries without capture after missing device embed");
    fails += check(verify_out[0] == 1,
                   "verify missing embed fallback returns CPU token");
    fails += check(sess.kv_len == 5,
                   "verify missing embed fallback advances kv_len once");
    fails += check(!sess.logits_valid && !sess.logits_host_valid &&
                   !sess.logits_on_device,
                   "verify missing embed fallback invalidates logits");
    fails += check(sess.next_token_pending == 0,
                   "verify missing embed fallback clears pending token");
    fails += check(cap_state.begin_count == 1,
                   "verify missing embed capture attempted once");
    fails += check(cap_state.verify_greedy_begin_count == 1,
                   "verify missing embed uses verify command sequence");
    fails += check(cap_state.greedy_step_begin_count == 0,
                   "verify missing embed avoids decode-greedy sequence");
    fails += check(cap_state.prefill_text_begin_count == 0,
                   "verify missing embed avoids prefill sequence");
    fails += check(cap_state.layer_loop_begin_count == 0,
                   "verify missing embed avoids nested layer sequence");
    fails += check(cap_state.discard_count == 1,
                   "verify missing embed capture is discarded");
    fails += check(cap_state.submit_count == 0,
                   "verify missing embed capture is not submitted");
    fails += check(cap_state.read_count == 0,
                   "discarded verify capture does not read token");
    fails += check(cap_state.captured_map_count == 0,
                   "captured verify missing embed does not map buffers");

    cap_state.begin_count = 0;
    cap_state.greedy_step_begin_count = 0;
    cap_state.layer_loop_begin_count = 0;
    cap_state.prefill_text_begin_count = 0;
    cap_state.verify_greedy_begin_count = 0;
    cap_state.discard_count = 0;
    cap_state.submit_count = 0;
    cap_state.read_count = 0;
    cap_state.captured_embedding_success_count = 0;
    cap_state.captured_buffer_copy_count = 0;
    cap_state.captured_greedy_unsupported_count = 0;
    cap_state.captured_greedy_success_count = 0;
    cap_state.captured_greedy_batch_success_count = 0;
    cap_state.captured_map_count = 0;
    cap_state.captured_map_null_count = 0;
    cap_state.null_map_during_capture = false;
    cap_state.captured_token_ready = false;
    cap_state.captured_token = -1;
    sess.kv_len = 6;
    sess.logits_valid = true;
    sess.logits_on_device = false;
    sess.logits_host_valid = true;
    sess.next_token_pending = 9;
    wrapper.desc = &runtime_success_desc;

    verify_out[0] = -1;
    if (s == GEIST_OK) {
        s = transformer_verify_forward(&state, 1, verify_ids, verify_out);
    }
    fails += check(s == GEIST_OK,
                   "successful verify capture returns OK");
    fails += check(verify_out[0] == 1,
                   "successful verify capture reads back token id");
    fails += check(sess.kv_len == 7,
                   "successful verify capture advances kv_len once");
    fails += check(!sess.logits_valid && !sess.logits_host_valid &&
                   !sess.logits_on_device,
                   "successful verify capture leaves logits invalid");
    fails += check(sess.next_token_pending == 0,
                   "successful verify capture clears pending token");
    fails += check(cap_state.begin_count == 1,
                   "successful verify capture begins once");
    fails += check(cap_state.verify_greedy_begin_count == 1,
                   "successful verify capture uses verify command sequence");
    fails += check(cap_state.greedy_step_begin_count == 0,
                   "successful verify capture avoids decode-greedy sequence");
    fails += check(cap_state.prefill_text_begin_count == 0,
                   "successful verify capture avoids prefill sequence");
    fails += check(cap_state.layer_loop_begin_count == 0,
                   "successful verify capture avoids nested layer sequence");
    fails += check(cap_state.submit_count == 1,
                   "successful verify capture submits once");
    fails += check(cap_state.discard_count == 0,
                   "successful verify capture is not discarded");
    fails += check(cap_state.read_count == 1,
                   "successful verify capture reads token once");
    fails += check(cap_state.captured_embedding_success_count == 1,
                   "successful verify capture records device embedding");
    fails += check(cap_state.captured_buffer_copy_count == 1,
                   "successful verify capture avoids redundant layer-loop seed copy");
    fails += check(cap_state.captured_map_count == 0,
                   "successful verify capture does not map buffers");
    fails += check(cap_state.captured_greedy_success_count == 1,
                   "successful verify capture records greedy_head once");

    cap_state.begin_count = 0;
    cap_state.greedy_step_begin_count = 0;
    cap_state.layer_loop_begin_count = 0;
    cap_state.prefill_text_begin_count = 0;
    cap_state.verify_greedy_begin_count = 0;
    cap_state.discard_count = 0;
    cap_state.submit_count = 0;
    cap_state.read_count = 0;
    cap_state.captured_embedding_success_count = 0;
    cap_state.captured_buffer_copy_count = 0;
    cap_state.captured_greedy_unsupported_count = 0;
    cap_state.captured_greedy_success_count = 0;
    cap_state.captured_greedy_batch_success_count = 0;
    cap_state.captured_map_count = 0;
    cap_state.captured_map_null_count = 0;
    cap_state.null_map_during_capture = false;
    cap_state.captured_token_ready = false;
    cap_state.captured_token = -1;
    cap_state.captured_token_count = 0;
    sess.kv_len = 7;
    sess.logits_valid = true;
    sess.logits_on_device = true;
    sess.logits_host_valid = false;
    sess.next_token_pending = 11;
    wrapper.desc = &runtime_no_embed_desc;

    const geist_token_t verify_ids2[2] = {0, 1};
    geist_token_t verify_out2[2] = {-1, -1};
    if (s == GEIST_OK) {
        s = transformer_verify_forward(&state, 2, verify_ids2, verify_out2);
    }
    fails += check(s == GEIST_OK,
                   "verify k2 retries without capture after missing device embed");
    fails += check(verify_out2[0] == 0 && verify_out2[1] == 1,
                   "verify k2 missing embed fallback returns CPU tokens");
    fails += check(sess.kv_len == 9,
                   "verify k2 missing embed fallback advances kv_len twice");
    fails += check(!sess.logits_valid && !sess.logits_host_valid &&
                   !sess.logits_on_device,
                   "verify k2 missing embed fallback invalidates logits");
    fails += check(sess.next_token_pending == 0,
                   "verify k2 missing embed fallback clears pending token");
    fails += check(cap_state.begin_count == 1,
                   "verify k2 missing embed capture attempted once");
    fails += check(cap_state.verify_greedy_begin_count == 1,
                   "verify k2 missing embed uses verify command sequence");
    fails += check(cap_state.discard_count == 1,
                   "verify k2 missing embed capture is discarded");
    fails += check(cap_state.submit_count == 0,
                   "verify k2 missing embed capture is not submitted");
    fails += check(cap_state.read_count == 0,
                   "discarded verify k2 capture does not read tokens");
    fails += check(cap_state.captured_map_count == 0,
                   "captured verify k2 missing embed does not map buffers");

    cap_state.begin_count = 0;
    cap_state.greedy_step_begin_count = 0;
    cap_state.layer_loop_begin_count = 0;
    cap_state.prefill_text_begin_count = 0;
    cap_state.verify_greedy_begin_count = 0;
    cap_state.discard_count = 0;
    cap_state.submit_count = 0;
    cap_state.read_count = 0;
    cap_state.captured_embedding_success_count = 0;
    cap_state.captured_buffer_copy_count = 0;
    cap_state.captured_greedy_unsupported_count = 0;
    cap_state.captured_greedy_success_count = 0;
    cap_state.captured_greedy_batch_success_count = 0;
    cap_state.captured_map_count = 0;
    cap_state.captured_map_null_count = 0;
    cap_state.null_map_during_capture = false;
    cap_state.captured_token_ready = false;
    cap_state.captured_token = -1;
    cap_state.captured_token_count = 0;
    sess.kv_len = 9;
    sess.logits_valid = true;
    sess.logits_on_device = false;
    sess.logits_host_valid = true;
    sess.next_token_pending = 13;
    wrapper.desc = &runtime_success_desc;

    verify_out2[0] = -1;
    verify_out2[1] = -1;
    if (s == GEIST_OK) {
        s = transformer_verify_forward(&state, 2, verify_ids2, verify_out2);
    }
    fails += check(s == GEIST_OK,
                   "successful verify k2 capture returns OK");
    fails += check(verify_out2[0] == 1 && verify_out2[1] == 1,
                   "successful verify k2 capture reads back token ids");
    fails += check(sess.kv_len == 11,
                   "successful verify k2 capture advances kv_len twice");
    fails += check(!sess.logits_valid && !sess.logits_host_valid &&
                   !sess.logits_on_device,
                   "successful verify k2 capture leaves logits invalid");
    fails += check(sess.next_token_pending == 0,
                   "successful verify k2 capture clears pending token");
    fails += check(cap_state.begin_count == 1,
                   "successful verify k2 capture begins once");
    fails += check(cap_state.verify_greedy_begin_count == 1,
                   "successful verify k2 capture uses verify command sequence");
    fails += check(cap_state.greedy_step_begin_count == 0,
                   "successful verify k2 capture avoids decode-greedy sequence");
    fails += check(cap_state.prefill_text_begin_count == 0,
                   "successful verify k2 capture avoids prefill sequence");
    fails += check(cap_state.layer_loop_begin_count == 0,
                   "successful verify k2 capture avoids nested layer sequence");
    fails += check(cap_state.submit_count == 1,
                   "successful verify k2 capture submits once");
    fails += check(cap_state.discard_count == 0,
                   "successful verify k2 capture is not discarded");
    fails += check(cap_state.read_count == 1,
                   "successful verify k2 capture reads tokens once");
    fails += check(cap_state.captured_embedding_success_count == 2,
                   "successful verify k2 capture records device embeddings");
    fails += check(cap_state.captured_buffer_copy_count == 1,
                   "successful verify k2 capture avoids redundant layer-loop seed copy");
    fails += check(cap_state.captured_map_count == 0,
                   "successful verify k2 capture does not map buffers");
    fails += check(cap_state.captured_greedy_success_count == 0,
                   "successful verify k2 capture avoids single-row greedy_head");
    fails += check(cap_state.captured_greedy_batch_success_count == 1,
                   "successful verify k2 capture records greedy_head_batch once");

    cap_state.begin_count = 0;
    cap_state.greedy_step_begin_count = 0;
    cap_state.layer_loop_begin_count = 0;
    cap_state.prefill_text_begin_count = 0;
    cap_state.verify_greedy_begin_count = 0;
    cap_state.discard_count = 0;
    cap_state.submit_count = 0;
    cap_state.read_count = 0;
    cap_state.captured_embedding_success_count = 0;
    cap_state.captured_buffer_copy_count = 0;
    cap_state.captured_greedy_unsupported_count = 0;
    cap_state.captured_greedy_success_count = 0;
    cap_state.captured_greedy_batch_success_count = 0;
    cap_state.captured_map_count = 0;
    cap_state.captured_map_null_count = 0;
    cap_state.null_map_during_capture = false;
    cap_state.captured_token_ready = false;
    cap_state.captured_token = -1;
    sess.kv_len = 3;
    sess.logits_valid = false;
    sess.logits_on_device = false;
    sess.logits_host_valid = false;
    sess.next_token_pending = -1;
    wrapper.desc = &runtime_desc;

    geist_token_t out_token = -1;
    if (s == GEIST_OK) {
        s = transformer_decode_step(&state, 1, &out_token);
    }
    fails += check(s == GEIST_OK,
                   "decode retries without capture after unsupported fastpath");
    fails += check(out_token == 1, "fallback decode returns CPU token");
    fails += check(sess.kv_len == 4, "fallback decode advances kv_len once");
    fails += check(sess.next_token_pending == 1,
                   "fallback decode stores pending token");
    fails += check(sess.logits_valid, "fallback decode leaves logits valid");
    fails += check(sess.logits_host_valid,
                   "fallback decode marks host logits valid");
    fails += check(!sess.logits_on_device,
                   "fallback decode does not claim device logits");
    fails += check(cap_state.begin_count == 1,
                   "capture attempted exactly once");
    fails += check(cap_state.greedy_step_begin_count == 1,
                   "unsupported capture uses decode-greedy command sequence");
    fails += check(cap_state.layer_loop_begin_count == 0,
                   "unsupported capture does not start nested layer sequence");
    fails += check(cap_state.discard_count == 1,
                   "unsupported capture is discarded");
    fails += check(cap_state.submit_count == 0,
                   "unsupported capture is not submitted");
    fails += check(cap_state.read_count == 0,
                   "discarded capture does not read token");
    fails += check(cap_state.captured_greedy_unsupported_count == 1,
                   "captured greedy_head rejected once");

    cap_state.begin_count = 0;
    cap_state.greedy_step_begin_count = 0;
    cap_state.layer_loop_begin_count = 0;
    cap_state.discard_count = 0;
    cap_state.submit_count = 0;
    cap_state.read_count = 0;
    cap_state.captured_embedding_success_count = 0;
    cap_state.captured_buffer_copy_count = 0;
    cap_state.captured_greedy_unsupported_count = 0;
    cap_state.captured_map_count = 0;
    cap_state.captured_map_null_count = 0;
    cap_state.null_map_during_capture = false;
    sess.kv_len = 5;
    sess.logits_valid = false;
    sess.logits_on_device = false;
    sess.logits_host_valid = false;
    sess.next_token_pending = -1;
    wrapper.desc = &runtime_no_embed_desc;

    out_token = -1;
    if (s == GEIST_OK) {
        s = transformer_decode_step(&state, 1, &out_token);
    }
    fails += check(s == GEIST_OK,
                   "decode retries after captured host-visible embed fallback");
    fails += check(out_token == 1,
                   "host-visible embed fallback retry returns CPU token");
    fails += check(sess.kv_len == 6,
                   "host-visible embed fallback retry advances kv_len once");
    fails += check(cap_state.begin_count == 1,
                   "host-visible embed fallback capture attempted once");
    fails += check(cap_state.greedy_step_begin_count == 1,
                   "host-visible embed fallback uses decode-greedy sequence");
    fails += check(cap_state.layer_loop_begin_count == 0,
                   "host-visible embed fallback avoids nested layer sequence");
    fails += check(cap_state.discard_count == 1,
                   "host-visible embed fallback capture is discarded");
    fails += check(cap_state.submit_count == 0,
                   "host-visible embed fallback capture is not submitted");
    fails += check(cap_state.read_count == 0,
                   "discarded host-visible embed capture does not read token");
    fails += check(cap_state.captured_map_count == 0,
                   "captured host-visible embed fallback does not map buffers");
    fails += check(cap_state.captured_greedy_unsupported_count == 0,
                   "captured host-visible embed fallback aborts before greedy_head");

    cap_state.begin_count = 0;
    cap_state.greedy_step_begin_count = 0;
    cap_state.layer_loop_begin_count = 0;
    cap_state.discard_count = 0;
    cap_state.submit_count = 0;
    cap_state.read_count = 0;
    cap_state.captured_embedding_success_count = 0;
    cap_state.captured_buffer_copy_count = 0;
    cap_state.captured_greedy_unsupported_count = 0;
    cap_state.captured_map_count = 0;
    cap_state.captured_map_null_count = 0;
    cap_state.null_map_during_capture = true;
    sess.kv_len = 7;
    sess.logits_valid = false;
    sess.logits_on_device = false;
    sess.logits_host_valid = false;
    sess.next_token_pending = -1;
    wrapper.desc = &runtime_no_embed_desc;

    out_token = -1;
    if (s == GEIST_OK) {
        s = transformer_decode_step(&state, 1, &out_token);
    }
    fails += check(s == GEIST_OK,
                   "decode retries after captured map fallback is unsupported");
    fails += check(out_token == 1,
                   "map-fallback retry returns CPU token");
    fails += check(sess.kv_len == 8,
                   "map-fallback retry advances kv_len once");
    fails += check(sess.next_token_pending == 1,
                   "map-fallback retry stores pending token");
    fails += check(sess.logits_valid && sess.logits_host_valid,
                   "map-fallback retry leaves host logits valid");
    fails += check(!sess.logits_on_device,
                   "map-fallback retry does not claim device logits");
    fails += check(cap_state.begin_count == 1,
                   "map-fallback capture attempted exactly once");
    fails += check(cap_state.greedy_step_begin_count == 1,
                   "map-fallback capture uses decode-greedy sequence");
    fails += check(cap_state.layer_loop_begin_count == 0,
                   "map-fallback capture avoids nested layer sequence");
    fails += check(cap_state.discard_count == 1,
                   "map-fallback capture is discarded");
    fails += check(cap_state.submit_count == 0,
                   "map-fallback capture is not submitted");
    fails += check(cap_state.read_count == 0,
                   "discarded map-fallback capture does not read token");
    fails += check(cap_state.captured_map_null_count == 0,
                   "captured missing-embed fallback does not reach null map");
    fails += check(cap_state.captured_map_count == 0,
                   "captured missing-embed fallback does not map buffers");
    fails += check(cap_state.captured_greedy_unsupported_count == 0,
                   "map-fallback capture aborts before greedy_head");

    cap_state.begin_count = 0;
    cap_state.greedy_step_begin_count = 0;
    cap_state.layer_loop_begin_count = 0;
    cap_state.discard_count = 0;
    cap_state.submit_count = 0;
    cap_state.read_count = 0;
    cap_state.captured_embedding_success_count = 0;
    cap_state.captured_buffer_copy_count = 0;
    cap_state.captured_greedy_unsupported_count = 0;
    cap_state.captured_greedy_success_count = 0;
    cap_state.captured_greedy_batch_success_count = 0;
    cap_state.captured_map_count = 0;
    cap_state.captured_map_null_count = 0;
    cap_state.null_map_during_capture = false;
    cap_state.captured_token_ready = false;
    cap_state.captured_token = -1;
    sess.kv_len = 9;
    sess.logits_valid = false;
    sess.logits_on_device = false;
    sess.logits_host_valid = false;
    sess.next_token_pending = -1;
    sess.temperature = 1.0f;
    sess.top_p = 1.0f;
    sess.top_k = 0;
    struct transformer_accel_session *greedy_accel_session =
        sess.accel_session;
    sess.accel_session = nullptr;
    wrapper.desc = &runtime_desc;

    out_token = -1;
    if (s == GEIST_OK) {
        s = transformer_decode_step(&state, 1, &out_token);
    }
    fails += check(s == GEIST_OK,
                   "sampled decode without accel session returns OK");
    fails += check(sess.kv_len == 10,
                   "sampled decode without accel advances kv_len once");
    fails += check(sess.logits_valid && sess.logits_host_valid,
                   "sampled decode without accel leaves host logits valid");
    fails += check(!sess.logits_on_device,
                   "sampled decode without accel does not claim device logits");
    fails += check(cap_state.begin_count == 0,
                   "sampled decode without accel does not begin capture");
    fails += check(cap_state.greedy_step_begin_count == 0,
                   "sampled decode without accel does not use greedy sequence");
    fails += check(cap_state.layer_loop_begin_count == 0,
                   "sampled decode without accel avoids layer-loop sequence");
    fails += check(cap_state.submit_count == 0 && cap_state.discard_count == 0,
                   "sampled decode without accel neither submits nor discards capture");
    fails += check(cap_state.read_count == 0,
                   "sampled decode without accel does not read captured token");
    sess.temperature = 0.0f;
    sess.top_p = 1.0f;
    sess.top_k = 0;
    sess.accel_session = greedy_accel_session;

    cap_state.begin_count = 0;
    cap_state.greedy_step_begin_count = 0;
    cap_state.layer_loop_begin_count = 0;
    cap_state.discard_count = 0;
    cap_state.submit_count = 0;
    cap_state.read_count = 0;
    cap_state.captured_embedding_success_count = 0;
    cap_state.captured_buffer_copy_count = 0;
    cap_state.captured_greedy_unsupported_count = 0;
    cap_state.captured_greedy_success_count = 0;
    cap_state.captured_greedy_batch_success_count = 0;
    cap_state.captured_map_count = 0;
    cap_state.captured_map_null_count = 0;
    cap_state.null_map_during_capture = false;
    cap_state.captured_token_ready = false;
    cap_state.captured_token = -1;
    sess.kv_len = 11;
    sess.logits_valid = false;
    sess.logits_on_device = false;
    sess.logits_host_valid = false;
    sess.next_token_pending = -1;
    wrapper.desc = &runtime_success_desc;

    out_token = -1;
    if (s == GEIST_OK) {
        s = transformer_decode_step(&state, 1, &out_token);
    }
    fails += check(s == GEIST_OK,
                   "successful capture decode returns OK");
    fails += check(out_token == 1,
                   "successful capture reads back token id");
    fails += check(sess.kv_len == 12,
                   "successful capture advances kv_len once");
    fails += check(sess.next_token_pending == 1,
                   "successful capture stores pending token");
    fails += check(sess.logits_valid,
                   "successful capture marks logits valid");
    fails += check(sess.logits_on_device,
                   "successful capture marks logits device-resident");
    fails += check(!sess.logits_host_valid,
                   "successful capture does not claim host logits");
    fails += check(cap_state.begin_count == 1,
                   "successful capture begins exactly once");
    fails += check(cap_state.greedy_step_begin_count == 1,
                   "successful capture uses decode-greedy command sequence");
    fails += check(cap_state.layer_loop_begin_count == 0,
                   "successful capture does not start nested layer sequence");
    fails += check(cap_state.submit_count == 1,
                   "successful capture submits exactly once");
    fails += check(cap_state.discard_count == 0,
                   "successful capture is not discarded");
    fails += check(cap_state.read_count == 1,
                   "successful capture reads token once");
    fails += check(cap_state.captured_embedding_success_count == 1,
                   "successful capture records embedding in command sequence");
    fails += check(cap_state.captured_buffer_copy_count == 1,
                   "successful capture avoids redundant layer-loop seed copy");
    fails += check(cap_state.captured_map_count == 0,
                   "successful capture does not map buffers");
    fails += check(cap_state.captured_greedy_success_count == 1,
                   "successful capture records greedy_head once");
    fails += check(cap_state.captured_greedy_unsupported_count == 0,
                   "successful capture avoids unsupported greedy fallback");

    cap_state.begin_count = 0;
    cap_state.greedy_step_begin_count = 0;
    cap_state.layer_loop_begin_count = 0;
    cap_state.discard_count = 0;
    cap_state.submit_count = 0;
    cap_state.read_count = 0;
    cap_state.replay_count = 0;
    cap_state.replay_token_id = -1;
    cap_state.replay_q_position = 0;
    cap_state.replay_status = GEIST_OK;
    cap_state.replay_out_token = 7;
    cap_state.captured_embedding_success_count = 0;
    cap_state.captured_buffer_copy_count = 0;
    cap_state.captured_greedy_unsupported_count = 0;
    cap_state.captured_greedy_success_count = 0;
    cap_state.captured_greedy_batch_success_count = 0;
    cap_state.captured_map_count = 0;
    cap_state.captured_map_null_count = 0;
    cap_state.null_map_during_capture = false;
    cap_state.captured_token_ready = false;
    cap_state.captured_token = -1;
    sess.kv_len = 13;
    sess.logits_valid = false;
    sess.logits_on_device = false;
    sess.logits_host_valid = false;
    sess.next_token_pending = -1;
    wrapper.desc = &runtime_replay_desc;

    out_token = -1;
    if (s == GEIST_OK) {
        s = transformer_decode_step(&state, 2, &out_token);
    }
    fails += check(s == GEIST_OK,
                   "decode replay returns OK");
    fails += check(out_token == 7,
                   "decode replay returns replay token id");
    fails += check(sess.kv_len == 14,
                   "decode replay advances kv_len once");
    fails += check(sess.next_token_pending == 7,
                   "decode replay stores pending token");
    fails += check(sess.logits_valid,
                   "decode replay marks logits valid");
    fails += check(sess.logits_on_device,
                   "decode replay marks logits device-resident");
    fails += check(!sess.logits_host_valid,
                   "decode replay does not claim host logits");
    fails += check(cap_state.replay_count == 1,
                   "decode replay hook called once");
    fails += check(cap_state.replay_token_id == 2,
                   "decode replay receives input token id");
    fails += check(cap_state.replay_q_position == 13,
                   "decode replay receives current q_position");
    fails += check(cap_state.begin_count == 0,
                   "decode replay does not begin command capture");
    fails += check(cap_state.greedy_step_begin_count == 0,
                   "decode replay avoids decode-greedy capture");
    fails += check(cap_state.submit_count == 0 && cap_state.discard_count == 0,
                   "decode replay neither submits nor discards capture");
    fails += check(cap_state.read_count == 0,
                   "decode replay does not call captured-token readback");
    fails += check(cap_state.captured_embedding_success_count == 0,
                   "decode replay skips embedding op calls");
    fails += check(cap_state.captured_greedy_success_count == 0,
                   "decode replay skips greedy_head op calls");

    transformer_accel_session_destroy(accel, sess.accel_session);
    transformer_accel_destroy(accel);
    if (embed_buf != nullptr) { cpu_v->buffer_destroy(cpu, embed_buf); }
    if (norm_buf != nullptr) { cpu_v->buffer_destroy(cpu, norm_buf); }
    if (scratch_h_a != nullptr) { cpu_v->buffer_destroy(cpu, scratch_h_a); }
    if (scratch_h_b != nullptr) { cpu_v->buffer_destroy(cpu, scratch_h_b); }
    if (logits_buf != nullptr) { cpu_v->buffer_destroy(cpu, logits_buf); }
    geist_backend_destroy(cpu);
    g_capture_map_state = nullptr;

    if (fails == 0) {
        printf("PASS: transformer decode capture fallback\n");
        return GEIST_TEST_PASS;
    }
    fprintf(stderr, "FAILED: %d check(s)\n", fails);
    return GEIST_TEST_FAIL;
}
