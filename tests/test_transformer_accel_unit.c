/*
 * test_transformer_accel_unit - optional transformer accelerator lifecycle.
 *
 * Verifies the no-op accelerator shim used before the Vulkan fastpath becomes
 * eligible. No model is loaded; this test only checks ownership contracts and
 * argument handling.
 */
#include "test_helpers.h"

#define GEIST_INTERNAL_ARCH_LAYER
#include "src/archs/transformer/accel.h"
#include "src/archs/transformer/arch_state.h"

#include <geist.h>
#include <geist_backend.h>

#include <stdio.h>

static struct geist_backend_accel_caps fake_default_accel_caps(void) {
    return (struct geist_backend_accel_caps){
        .struct_size = sizeof(struct geist_backend_accel_caps),
        .device_resident_buffers = true,
        .compute_queue = true,
        .pipeline_cache = true,
        .subgroup_basic = true,
        .shader_integer_dot_product = true,
        .descriptor_indexing = true,
        .timeline_semaphore = true,
    };
}

struct fake_accel_caps_state {
    struct geist_backend_accel_caps caps;
};

static enum geist_status fake_query_accel_caps(
    struct geist_backend *be,
    struct geist_backend_accel_caps *out) {
    if (out == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    const struct fake_accel_caps_state *state =
        be != nullptr ? (const struct fake_accel_caps_state *) be->state :
                        nullptr;
    *out = state != nullptr ? state->caps : fake_default_accel_caps();
    return GEIST_OK;
}

static enum geist_status fake_buffer_copy(
    struct geist_buffer *dst,
    size_t dst_offset,
    const struct geist_buffer *src,
    size_t src_offset,
    size_t n_bytes) {
    (void) dst;
    (void) dst_offset;
    (void) src;
    (void) src_offset;
    (void) n_bytes;
    return GEIST_E_UNSUPPORTED;
}

static enum geist_status fake_matvec_f32_dense(
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

static enum geist_status fake_matvec_q4k(
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

static enum geist_status fake_matvec_q6k(
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

static enum geist_status fake_embedding_lookup_scaled(
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

static enum geist_status fake_rmsnorm(
    struct geist_backend *be,
    const struct geist_tensor *x,
    const struct geist_tensor *w,
    float eps,
    struct geist_tensor *y) {
    (void) be;
    (void) x;
    (void) w;
    (void) eps;
    (void) y;
    return GEIST_E_UNSUPPORTED;
}

static enum geist_status fake_add(
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

static enum geist_status fake_scale_f32(
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

static enum geist_status fake_attention_block(
    struct geist_backend *be,
    const struct geist_backend_attention_block *block) {
    (void) be;
    (void) block;
    return GEIST_E_UNSUPPORTED;
}

static enum geist_status fake_attention_query_block(
    struct geist_backend *be,
    const struct geist_backend_attention_query_block *block) {
    (void) be;
    (void) block;
    return GEIST_E_UNSUPPORTED;
}

static enum geist_status fake_ffn_geglu_block(
    struct geist_backend *be,
    const struct geist_backend_ffn_geglu_block *block) {
    (void) be;
    (void) block;
    return GEIST_E_UNSUPPORTED;
}

static enum geist_status fake_greedy_head(
    struct geist_backend *be,
    const struct geist_backend_greedy_head *head,
    geist_token_t *out_token) {
    (void) be;
    (void) head;
    (void) out_token;
    return GEIST_E_UNSUPPORTED;
}

static enum geist_status fake_command_sequence_begin(
    struct geist_backend *be,
    enum geist_command_sequence_kind kind,
    int *out_token) {
    (void) be;
    (void) kind;
    if (out_token != nullptr) {
        *out_token = 1;
    }
    return GEIST_OK;
}

static enum geist_status fake_command_sequence_end(
    struct geist_backend *be,
    int token,
    bool submit) {
    (void) be;
    (void) token;
    (void) submit;
    return GEIST_OK;
}

static enum geist_status fake_command_sequence_read_token(
    struct geist_backend *be,
    geist_token_t *out_token) {
    (void) be;
    if (out_token != nullptr) {
        *out_token = 0;
    }
    return GEIST_OK;
}

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

static void fill_target_weight_metadata(struct transformer_arch_state *state,
                                        enum geist_dtype proj_dtype,
                                        enum geist_layout proj_layout,
                                        enum geist_dtype embed_dtype,
                                        enum geist_layout embed_layout) {
    state->embed_table = make_weight_tensor(
        embed_dtype, embed_layout, state->vocab_size, state->d_model);
    state->ple_table = make_weight_tensor(
        embed_dtype, embed_layout, state->vocab_size, state->ple_out);
    state->model_proj = make_weight_tensor(
        GEIST_DTYPE_F32, GEIST_LAYOUT_DENSE, state->ple_out, state->d_model);
    state->model_proj_norm = (struct geist_tensor){
        .dtype = GEIST_DTYPE_F32,
        .layout = GEIST_LAYOUT_DENSE,
        .ndim = 1,
        .shape = {(int64_t) state->hidden_per_layer},
        .stride = {1},
    };
    state->output_norm = (struct geist_tensor){
        .dtype = GEIST_DTYPE_F32,
        .layout = GEIST_LAYOUT_DENSE,
        .ndim = 1,
        .shape = {(int64_t) state->d_model},
        .stride = {1},
    };

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
        L->attn_norm = (struct geist_tensor){
            .dtype = GEIST_DTYPE_F32,
            .layout = GEIST_LAYOUT_DENSE,
            .ndim = 1,
            .shape = {(int64_t) state->d_model},
            .stride = {1},
        };
        L->q_norm = (struct geist_tensor){
            .dtype = GEIST_DTYPE_F32,
            .layout = GEIST_LAYOUT_DENSE,
            .ndim = 1,
            .shape = {(int64_t) L->head_dim},
            .stride = {1},
        };
        L->k_norm = L->is_kv_shared ? (struct geist_tensor){0} :
            (struct geist_tensor){
                .dtype = GEIST_DTYPE_F32,
                .layout = GEIST_LAYOUT_DENSE,
                .ndim = 1,
                .shape = {(int64_t) L->head_dim},
                .stride = {1},
            };
        L->post_attn_norm = L->attn_norm;
        L->ffn_norm = L->attn_norm;
        L->post_ffw_norm = L->attn_norm;
        L->post_per_layer_norm = L->attn_norm;
        L->q_proj = make_weight_tensor(
            proj_dtype, proj_layout, L->q_out, state->d_model);
        L->o_proj = make_weight_tensor(
            proj_dtype, proj_layout, state->d_model, L->q_out);
        L->k_proj = L->is_kv_shared ? (struct geist_tensor){0} :
            make_weight_tensor(proj_dtype, proj_layout,
                               L->kv_out, state->d_model);
        L->v_proj = L->is_kv_shared ? (struct geist_tensor){0} :
            make_weight_tensor(proj_dtype, proj_layout,
                               L->kv_out, state->d_model);
        L->gate_proj = make_weight_tensor(
            proj_dtype, proj_layout, L->intermediate, state->d_model);
        L->up_proj = make_weight_tensor(
            proj_dtype, proj_layout, L->intermediate, state->d_model);
        L->down_proj = make_weight_tensor(
            proj_dtype, proj_layout, state->d_model, L->intermediate);
        L->per_layer_gate = make_weight_tensor(
            proj_dtype, proj_layout, state->hidden_per_layer, state->d_model);
        L->per_layer_proj = make_weight_tensor(
            proj_dtype, proj_layout, state->d_model, state->hidden_per_layer);
    }
}

int main(void) {
    int fails = 0;

    struct geist_backend *be = nullptr;
    enum geist_status s = geist_backend_create("cpu_scalar", nullptr, nullptr, &be);
    fails += check(s == GEIST_OK, "cpu_scalar backend create OK");
    fails += check(be != nullptr, "cpu_scalar backend handle non-null");
    if (s != GEIST_OK || be == nullptr) {
        fprintf(stderr, "create-time error: %s\n", geist_last_create_error());
        return GEIST_TEST_FAIL;
    }

    struct transformer_arch_state state = {.backend = be};

    struct transformer_accel *accel = (struct transformer_accel *) 0x1;
    s = transformer_accel_try_create(nullptr, &accel);
    fails += check(s == GEIST_E_INVALID_ARG, "try_create rejects null state");
    fails += check(accel == nullptr, "try_create clears out on null state");

    s = transformer_accel_try_create(&state, nullptr);
    fails += check(s == GEIST_E_INVALID_ARG, "try_create rejects null out");

    accel = (struct transformer_accel *) 0x1;
    s = transformer_accel_try_create(&state, &accel);
    fails += check(s == GEIST_OK, "try_create declines acceleration cleanly");
    fails += check(accel == nullptr, "try_create returns null accelerator");

    struct transformer_layer_weights target_layers[GEIST_GEMMA4_NUM_LAYERS] = {0};
    state = (struct transformer_arch_state){
        .backend = be,
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
    accel = (struct transformer_accel *) 0x1;
    s = transformer_accel_try_create(&state, &accel);
    fails += check(s == GEIST_OK, "cpu target model still declines accelerator");
    fails += check(accel == nullptr, "cpu target model returns null accelerator");

    const struct geist_backend_vtbl fake_vtbl_without_read_token = {
        .query_accel_caps = fake_query_accel_caps,
        .buffer_copy = fake_buffer_copy,
        .embedding_lookup_scaled = fake_embedding_lookup_scaled,
        .rmsnorm = fake_rmsnorm,
        .matvec_f32_dense = fake_matvec_f32_dense,
        .matvec_q4k = fake_matvec_q4k,
        .matvec_q6k = fake_matvec_q6k,
        .add = fake_add,
        .scale_f32 = fake_scale_f32,
        .attention_block = fake_attention_block,
        .attention_query_block = fake_attention_query_block,
        .ffn_geglu_block = fake_ffn_geglu_block,
        .greedy_head = fake_greedy_head,
        .command_sequence_begin = fake_command_sequence_begin,
        .command_sequence_end = fake_command_sequence_end,
    };
    const struct geist_backend_descriptor fake_desc_without_read_token = {
        .name = "fake_accel_without_read_token",
        .vtbl = &fake_vtbl_without_read_token,
    };
    struct geist_backend fake_backend_without_read_token = {
        .desc = &fake_desc_without_read_token,
    };
    state.backend = &fake_backend_without_read_token;
    accel = (struct transformer_accel *) 0x1;
    s = transformer_accel_try_create(&state, &accel);
    fails += check(s == GEIST_OK,
                   "fake accelerator without token readback probes cleanly");
    fails += check(accel == nullptr,
                   "decode greedy accelerator requires token readback hook");

    const struct geist_backend_vtbl fake_vtbl_with_read_token = {
        .query_accel_caps = fake_query_accel_caps,
        .buffer_copy = fake_buffer_copy,
        .embedding_lookup_scaled = fake_embedding_lookup_scaled,
        .rmsnorm = fake_rmsnorm,
        .matvec_f32_dense = fake_matvec_f32_dense,
        .matvec_q4k = fake_matvec_q4k,
        .matvec_q6k = fake_matvec_q6k,
        .add = fake_add,
        .scale_f32 = fake_scale_f32,
        .attention_block = fake_attention_block,
        .attention_query_block = fake_attention_query_block,
        .ffn_geglu_block = fake_ffn_geglu_block,
        .greedy_head = fake_greedy_head,
        .command_sequence_begin = fake_command_sequence_begin,
        .command_sequence_end = fake_command_sequence_end,
        .command_sequence_read_token = fake_command_sequence_read_token,
    };
    const struct geist_backend_descriptor fake_desc_with_read_token = {
        .name = "fake_accel_with_read_token",
        .vtbl = &fake_vtbl_with_read_token,
    };
    struct geist_backend fake_backend_with_read_token = {
        .desc = &fake_desc_with_read_token,
    };
    state.backend = &fake_backend_with_read_token;
    accel = (struct transformer_accel *) 0x1;
    s = transformer_accel_try_create(&state, &accel);
    fails += check(s == GEIST_OK,
                   "fake accelerator without target weights probes cleanly");
    fails += check(accel == nullptr,
                   "decode greedy accelerator requires target weight layouts");

    fill_target_weight_metadata(&state,
                                GEIST_DTYPE_Q4_K,
                                GEIST_LAYOUT_BLOCK_QUANTIZED,
                                GEIST_DTYPE_Q6_K,
                                GEIST_LAYOUT_BLOCK_QUANTIZED);
    accel = nullptr;
    s = transformer_accel_try_create(&state, &accel);
    fails += check(s == GEIST_OK,
                   "fake accelerator with target weights probes cleanly");
    fails += check(accel != nullptr,
                   "fake accelerator with target weights is eligible");
    transformer_accel_destroy(accel);

    fill_target_weight_metadata(&state,
                                GEIST_DTYPE_Q4_K,
                                GEIST_LAYOUT_BLOCK_QUANTIZED,
                                GEIST_DTYPE_Q5_K,
                                GEIST_LAYOUT_BLOCK_QUANTIZED);
    accel = nullptr;
    s = transformer_accel_try_create(&state, &accel);
    fails += check(s == GEIST_OK,
                   "fake accelerator with q5 embedding weights probes cleanly");
    fails += check(accel != nullptr,
                   "fake accelerator permits q5 embedding weights");
    transformer_accel_destroy(accel);

    fill_target_weight_metadata(&state,
                                GEIST_DTYPE_F32,
                                GEIST_LAYOUT_DENSE,
                                GEIST_DTYPE_Q6_K,
                                GEIST_LAYOUT_BLOCK_QUANTIZED);
    accel = (struct transformer_accel *) 0x1;
    s = transformer_accel_try_create(&state, &accel);
    fails += check(s == GEIST_OK,
                   "fake accelerator with dense projection weights probes cleanly");
    fails += check(accel == nullptr,
                   "decode greedy accelerator requires q4/q6 projection weights");

    fill_target_weight_metadata(&state,
                                GEIST_DTYPE_Q4_K,
                                GEIST_LAYOUT_BLOCK_QUANTIZED,
                                GEIST_DTYPE_F32,
                                GEIST_LAYOUT_DENSE);
    accel = (struct transformer_accel *) 0x1;
    s = transformer_accel_try_create(&state, &accel);
    fails += check(s == GEIST_OK,
                   "fake accelerator with dense embedding weights probes cleanly");
    fails += check(accel == nullptr,
                   "decode greedy accelerator requires q4/q5/q6 embedding weights");

    fill_target_weight_metadata(&state,
                                GEIST_DTYPE_Q4_K,
                                GEIST_LAYOUT_BLOCK_QUANTIZED,
                                GEIST_DTYPE_Q6_K,
                                GEIST_LAYOUT_BLOCK_QUANTIZED);

    struct fake_accel_caps_state missing_dot = {
        .caps = fake_default_accel_caps(),
    };
    missing_dot.caps.shader_integer_dot_product = false;
    fake_backend_with_read_token.state = &missing_dot;
    accel = (struct transformer_accel *) 0x1;
    s = transformer_accel_try_create(&state, &accel);
    fails += check(s == GEIST_OK,
                   "fake accelerator without integer dot probes cleanly");
    fails += check(accel != nullptr,
                   "decode greedy accelerator permits missing integer dot product");
    transformer_accel_destroy(accel);

    struct fake_accel_caps_state missing_subgroup = {
        .caps = fake_default_accel_caps(),
    };
    missing_subgroup.caps.subgroup_basic = false;
    fake_backend_with_read_token.state = &missing_subgroup;
    accel = (struct transformer_accel *) 0x1;
    s = transformer_accel_try_create(&state, &accel);
    fails += check(s == GEIST_OK,
                   "fake accelerator without subgroup probes cleanly");
    fails += check(accel != nullptr,
                   "decode greedy accelerator permits missing subgroup");
    transformer_accel_destroy(accel);

    struct fake_accel_caps_state missing_descriptor_indexing = {
        .caps = fake_default_accel_caps(),
    };
    missing_descriptor_indexing.caps.descriptor_indexing = false;
    fake_backend_with_read_token.state = &missing_descriptor_indexing;
    accel = (struct transformer_accel *) 0x1;
    s = transformer_accel_try_create(&state, &accel);
    fails += check(s == GEIST_OK,
                   "fake accelerator without descriptor indexing probes cleanly");
    fails += check(accel != nullptr,
                   "decode greedy accelerator permits missing descriptor indexing");
    transformer_accel_destroy(accel);

    struct fake_accel_caps_state missing_timeline = {
        .caps = fake_default_accel_caps(),
    };
    missing_timeline.caps.timeline_semaphore = false;
    fake_backend_with_read_token.state = &missing_timeline;
    accel = (struct transformer_accel *) 0x1;
    s = transformer_accel_try_create(&state, &accel);
    fails += check(s == GEIST_OK,
                   "fake accelerator without timeline semaphore probes cleanly");
    fails += check(accel != nullptr,
                   "decode greedy accelerator permits missing timeline semaphore");
    transformer_accel_destroy(accel);

    fake_backend_with_read_token.state = nullptr;
    state.backend = be;

    struct transformer_accel_session *accel_sess =
        (struct transformer_accel_session *) 0x1;
    s = transformer_accel_session_create(nullptr, nullptr, &accel_sess);
    fails += check(s == GEIST_OK, "session_create accepts null accelerator");
    fails += check(accel_sess == nullptr, "session_create returns null session");
    fails += check(!transformer_accel_session_decode_greedy_enabled(accel_sess),
                   "null accel session does not enable greedy decode");

    s = transformer_accel_session_create(nullptr, nullptr, nullptr);
    fails += check(s == GEIST_E_INVALID_ARG, "session_create rejects null out");

    transformer_accel_session_destroy(nullptr, nullptr);
    transformer_accel_destroy(nullptr);
    geist_backend_destroy(be);

    struct geist_backend *vk = nullptr;
    s = geist_backend_create("vulkan", nullptr, nullptr, &vk);
    if (s == GEIST_OK) {
        state.backend = vk;
        accel = nullptr;
        s = transformer_accel_try_create(&state, &accel);
        fails += check(s == GEIST_OK, "vulkan target model accelerator probe OK");
        if (accel != nullptr) {
            struct transformer_arch_session greedy = {
                .temperature = 0.0f,
                .top_p = 1.0f,
                .top_k = 0,
            };
            accel_sess = (struct transformer_accel_session *) 0x1;
            s = transformer_accel_session_create(accel, &greedy, &accel_sess);
            fails += check(s == GEIST_OK, "vulkan greedy accel session create OK");
            fails += check(accel_sess != nullptr,
                           "vulkan greedy accel session handle non-null");
            fails += check(transformer_accel_session_decode_greedy_enabled(accel_sess),
                           "vulkan greedy accel session enables decode path");
            transformer_accel_session_destroy(accel, accel_sess);

            struct transformer_arch_session sampled = {
                .temperature = 1.0f,
                .top_p = 1.0f,
                .top_k = 0,
            };
            accel_sess = (struct transformer_accel_session *) 0x1;
            s = transformer_accel_session_create(accel, &sampled, &accel_sess);
            fails += check(s == GEIST_OK,
                           "vulkan sampled accel session falls back cleanly");
            fails += check(accel_sess == nullptr,
                           "vulkan sampled accel session remains null");
            fails += check(!transformer_accel_session_decode_greedy_enabled(accel_sess),
                           "sampled accel session does not enable decode path");

            struct transformer_arch_session int8_kv = {
                .temperature = 0.0f,
                .top_p = 1.0f,
                .top_k = 0,
                .kv_int8_enabled = true,
            };
            accel_sess = (struct transformer_accel_session *) 0x1;
            s = transformer_accel_session_create(accel, &int8_kv, &accel_sess);
            fails += check(s == GEIST_OK,
                           "vulkan int8-kv accel session falls back cleanly");
            fails += check(accel_sess == nullptr,
                           "vulkan int8-kv accel session remains null");

            struct transformer_arch_session kivi_kv = {
                .temperature = 0.0f,
                .top_p = 1.0f,
                .top_k = 0,
                .kv_kivi_enabled = true,
            };
            accel_sess = (struct transformer_accel_session *) 0x1;
            s = transformer_accel_session_create(accel, &kivi_kv, &accel_sess);
            fails += check(s == GEIST_OK,
                           "vulkan kivi accel session falls back cleanly");
            fails += check(accel_sess == nullptr,
                           "vulkan kivi accel session remains null");
            transformer_accel_destroy(accel);
        }
        geist_backend_destroy(vk);
    } else {
        fails += check(s == GEIST_E_NOT_FOUND || s == GEIST_E_BACKEND,
                       "vulkan backend unavailable cleanly for accel test");
    }

    if (fails == 0) {
        printf("PASS: transformer accelerator no-op lifecycle\n");
        return GEIST_TEST_PASS;
    }
    fprintf(stderr, "FAILED: %d check(s)\n", fails);
    return GEIST_TEST_FAIL;
}
