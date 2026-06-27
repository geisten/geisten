/*
 * test_transformer_capture_layer_blocks_unit - capture block fusion boundary.
 *
 * During a device-resident decode command sequence, unsupported layer-level
 * block fastpaths must abort the capture before the architecture enters the
 * decomposed Host-pointer fallback path.
 */
#include "test_helpers.h"

#define GEIST_INTERNAL_ARCH_LAYER
#include "src/archs/transformer/arch_state.h"
#include "src/archs/transformer/forward.h"
#include "src/archs/transformer/forward/internal.h"

#include <geist.h>
#include <geist_backend.h>

#include <stdio.h>

struct fake_stage_state {
    int attention_block_count;
    int ffn_block_count;
    int ple_block_count;
    int rmsnorm_count;
    int scale_count;
    int buffer_map_count;
    int buffer_copy_count;
    int sequence_begin_count;
    int sequence_submit_count;
    int sequence_discard_count;
    bool sequence_active;
};

static struct fake_stage_state *g_fake_map_state;

static int check(bool cond, const char *what) {
    if (!cond) {
        fprintf(stderr, "FAIL: %s\n", what);
        return 1;
    }
    return 0;
}

static struct fake_stage_state *fake_state(struct geist_backend *be) {
    return be != nullptr ? (struct fake_stage_state *) be->state : nullptr;
}

static enum geist_status fake_attention_block(
    struct geist_backend *be,
    const struct geist_backend_attention_block *block) {
    (void) block;
    struct fake_stage_state *st = fake_state(be);
    if (st != nullptr) {
        st->attention_block_count++;
    }
    return GEIST_E_UNSUPPORTED;
}

static enum geist_status fake_attention_block_ok(
    struct geist_backend *be,
    const struct geist_backend_attention_block *block) {
    (void) block;
    struct fake_stage_state *st = fake_state(be);
    if (st != nullptr) {
        st->attention_block_count++;
    }
    return GEIST_OK;
}

static enum geist_status fake_ffn_geglu_block(
    struct geist_backend *be,
    const struct geist_backend_ffn_geglu_block *block) {
    (void) block;
    struct fake_stage_state *st = fake_state(be);
    if (st != nullptr) {
        st->ffn_block_count++;
    }
    return GEIST_E_UNSUPPORTED;
}

static enum geist_status fake_ffn_geglu_block_ok(
    struct geist_backend *be,
    const struct geist_backend_ffn_geglu_block *block) {
    (void) block;
    struct fake_stage_state *st = fake_state(be);
    if (st != nullptr) {
        st->ffn_block_count++;
    }
    return GEIST_OK;
}

static enum geist_status fake_ple_block(
    struct geist_backend *be,
    const struct geist_backend_ple_block *block) {
    (void) block;
    struct fake_stage_state *st = fake_state(be);
    if (st != nullptr) {
        st->ple_block_count++;
    }
    return GEIST_E_UNSUPPORTED;
}

static enum geist_status fake_ple_block_ok(
    struct geist_backend *be,
    const struct geist_backend_ple_block *block) {
    (void) block;
    struct fake_stage_state *st = fake_state(be);
    if (st != nullptr) {
        st->ple_block_count++;
    }
    return GEIST_OK;
}

static enum geist_status fake_rmsnorm(
    struct geist_backend *be,
    const struct geist_tensor *x,
    const struct geist_tensor *w,
    float eps,
    struct geist_tensor *y) {
    (void) x;
    (void) w;
    (void) eps;
    (void) y;
    struct fake_stage_state *st = fake_state(be);
    if (st != nullptr) {
        st->rmsnorm_count++;
    }
    return GEIST_E_BACKEND;
}

static enum geist_status fake_scale_f32(
    struct geist_backend *be,
    const struct geist_tensor *x,
    float scale,
    struct geist_tensor *y) {
    (void) x;
    (void) scale;
    (void) y;
    struct fake_stage_state *st = fake_state(be);
    if (st != nullptr) {
        st->scale_count++;
    }
    return GEIST_E_UNSUPPORTED;
}

static enum geist_status fake_scale_f32_ok(
    struct geist_backend *be,
    const struct geist_tensor *x,
    float scale,
    struct geist_tensor *y) {
    (void) x;
    (void) scale;
    (void) y;
    struct fake_stage_state *st = fake_state(be);
    if (st != nullptr) {
        st->scale_count++;
    }
    return GEIST_OK;
}

static void *fake_buffer_map(struct geist_buffer *buf) {
    (void) buf;
    if (g_fake_map_state != nullptr) {
        g_fake_map_state->buffer_map_count++;
    }
    return nullptr;
}

static enum geist_status fake_buffer_copy(struct geist_buffer *dst,
                                          size_t dst_offset,
                                          const struct geist_buffer *src,
                                          size_t src_offset,
                                          size_t n_bytes) {
    (void) dst;
    (void) dst_offset;
    (void) src;
    (void) src_offset;
    (void) n_bytes;
    if (g_fake_map_state != nullptr) {
        g_fake_map_state->buffer_copy_count++;
    }
    return GEIST_OK;
}

static enum geist_status fake_command_sequence_begin(
    struct geist_backend *be,
    enum geist_command_sequence_kind kind,
    int *out_token) {
    (void) kind;
    struct fake_stage_state *st = fake_state(be);
    if (st == nullptr || st->sequence_active || out_token == nullptr) {
        return GEIST_E_INVALID_STATE;
    }
    st->sequence_active = true;
    st->sequence_begin_count++;
    *out_token = 23;
    return GEIST_OK;
}

static enum geist_status fake_command_sequence_end(
    struct geist_backend *be,
    int token,
    bool submit) {
    struct fake_stage_state *st = fake_state(be);
    if (st == nullptr || !st->sequence_active || token != 23) {
        return GEIST_E_INVALID_STATE;
    }
    st->sequence_active = false;
    if (submit) {
        st->sequence_submit_count++;
    } else {
        st->sequence_discard_count++;
    }
    return GEIST_OK;
}

static struct geist_buffer *fake_buf(uintptr_t tag) {
    return (struct geist_buffer *) tag;
}

static void fill_common_context(
    struct transformer_arch_state *st,
    struct transformer_arch_session *sess,
    struct transformer_layer_weights *layer,
    struct transformer_layer_forward_ctx *ctx,
    struct geist_backend *be,
    const struct geist_backend_vtbl *vtbl) {

    *sess = (struct transformer_arch_session){
        .backend_command_sequence_active = true,
        .scratch_h_post_attn = fake_buf(0x101u),
        .scratch_h_post_ff = fake_buf(0x102u),
        .scratch_normed = fake_buf(0x103u),
        .scratch_q = fake_buf(0x104u),
        .scratch_k = fake_buf(0x105u),
        .scratch_v = fake_buf(0x106u),
        .scratch_attn = fake_buf(0x107u),
        .scratch_o = fake_buf(0x108u),
        .scratch_post_attn = fake_buf(0x109u),
        .scratch_pre_ff = fake_buf(0x10au),
        .scratch_gate = fake_buf(0x10bu),
        .scratch_up = fake_buf(0x10cu),
        .scratch_ffn_out = fake_buf(0x10du),
        .scratch_post_ff = fake_buf(0x10eu),
        .scratch_ones_headdim_max = fake_buf(0x10fu),
        .scratch_proj_ple = fake_buf(0x110u),
        .scratch_gate_ple = fake_buf(0x111u),
    };
    *layer = (struct transformer_layer_weights){
        .head_dim = 2,
        .q_out = 4,
        .kv_out = 4,
        .intermediate = 8,
        .is_full = true,
        .layer_scalar = 1.0f,
        .attn_norm = {.buffer = fake_buf(0x201u)},
        .q_norm = {.buffer = fake_buf(0x202u)},
        .k_norm = {.buffer = fake_buf(0x203u)},
        .post_attn_norm = {.buffer = fake_buf(0x204u)},
        .ffn_norm = {.buffer = fake_buf(0x205u)},
        .post_ffw_norm = {.buffer = fake_buf(0x206u)},
        .post_per_layer_norm = {.buffer = fake_buf(0x207u)},
        .q_proj = {.buffer = fake_buf(0x208u)},
        .k_proj = {.buffer = fake_buf(0x209u)},
        .v_proj = {.buffer = fake_buf(0x20au)},
        .o_proj = {.buffer = fake_buf(0x20bu)},
        .gate_proj = {.buffer = fake_buf(0x20cu)},
        .up_proj = {.buffer = fake_buf(0x20du)},
        .down_proj = {.buffer = fake_buf(0x20eu)},
    };
    *st = (struct transformer_arch_state){
        .backend = be,
        .sess = sess,
        .layers = layer,
        .n_layers = 1,
        .d_model = 4,
        .n_q_heads = 2,
        .n_kv_heads = 2,
        .hidden_per_layer = 3,
        .max_seq_len = 4,
        .rope_cos_full = fake_buf(0x301u),
        .rope_sin_full = fake_buf(0x302u),
    };
    *ctx = (struct transformer_layer_forward_ctx){
        .st = st,
        .be = be,
        .v = vtbl,
        .L = layer,
        .layer_idx = 0,
        .q_position = 0,
        .seq = 1,
        .h_in_buf = fake_buf(0x401u),
        .per_layer_input_buf = nullptr,
        .h_out_buf = fake_buf(0x402u),
        .compute_kv = true,
        .apply_gemma_attn_norms = true,
        .ffn_activation = GEIST_FFN_GEGLU,
        .eps = 1.0e-6f,
        .hd = 2,
        .q_out = 4,
        .kv_out = 4,
        .inter = 8,
        .SEQ = 1,
        .k_cache_buf = fake_buf(0x501u),
        .v_cache_buf = fake_buf(0x502u),
    };
}

int main(void) {
    int fails = 0;

    const struct geist_backend_vtbl vtbl = {
        .attention_block = fake_attention_block,
        .ffn_geglu_block = fake_ffn_geglu_block,
        .ple_block = fake_ple_block,
        .rmsnorm = fake_rmsnorm,
        .scale_f32 = fake_scale_f32,
        .buffer_map = fake_buffer_map,
    };
    const struct geist_backend_vtbl sequence_vtbl = {
        .attention_block = fake_attention_block,
        .ffn_geglu_block = fake_ffn_geglu_block,
        .ple_block = fake_ple_block,
        .rmsnorm = fake_rmsnorm,
        .scale_f32 = fake_scale_f32,
        .buffer_copy = fake_buffer_copy,
        .buffer_map = fake_buffer_map,
        .command_sequence_begin = fake_command_sequence_begin,
        .command_sequence_end = fake_command_sequence_end,
    };
    const struct geist_backend_descriptor desc = {
        .name = "fake_capture_layer_blocks",
        .vtbl = &vtbl,
    };
    const struct geist_backend_descriptor sequence_desc = {
        .name = "fake_capture_layer_loop",
        .vtbl = &sequence_vtbl,
    };
    const struct geist_backend_vtbl sequence_ok_vtbl = {
        .attention_block = fake_attention_block_ok,
        .ffn_geglu_block = fake_ffn_geglu_block_ok,
        .rmsnorm = fake_rmsnorm,
        .scale_f32 = fake_scale_f32_ok,
        .buffer_copy = fake_buffer_copy,
        .buffer_map = fake_buffer_map,
        .command_sequence_begin = fake_command_sequence_begin,
        .command_sequence_end = fake_command_sequence_end,
    };
    const struct geist_backend_vtbl sequence_full_ok_vtbl = {
        .attention_block = fake_attention_block_ok,
        .ffn_geglu_block = fake_ffn_geglu_block_ok,
        .ple_block = fake_ple_block_ok,
        .rmsnorm = fake_rmsnorm,
        .scale_f32 = fake_scale_f32_ok,
        .buffer_copy = fake_buffer_copy,
        .buffer_map = fake_buffer_map,
        .command_sequence_begin = fake_command_sequence_begin,
        .command_sequence_end = fake_command_sequence_end,
    };
    const struct geist_backend_descriptor sequence_ok_desc = {
        .name = "fake_capture_layer_attention_ffn",
        .vtbl = &sequence_ok_vtbl,
    };
    const struct geist_backend_descriptor sequence_full_ok_desc = {
        .name = "fake_capture_full_layer",
        .vtbl = &sequence_full_ok_vtbl,
    };
    struct fake_stage_state fake = {0};
    g_fake_map_state = &fake;
    struct geist_backend be = {
        .desc = &desc,
        .state = &fake,
    };
    struct transformer_arch_state st;
    struct transformer_arch_session sess;
    struct transformer_layer_weights layer;
    struct transformer_layer_forward_ctx ctx;

    fill_common_context(&st, &sess, &layer, &ctx, &be, &vtbl);
    enum geist_status s = transformer_layer_run_attention_block(&ctx);
    fails += check(s == GEIST_E_UNSUPPORTED,
                   "captured attention block unsupported aborts capture");
    fails += check(fake.attention_block_count == 1,
                   "captured attention attempts block fastpath once");
    fails += check(fake.rmsnorm_count == 0,
                   "captured attention does not enter decomposed fallback");

    fake = (struct fake_stage_state){0};
    fill_common_context(&st, &sess, &layer, &ctx, &be, &vtbl);
    s = transformer_layer_run_ffn_block(&ctx);
    fails += check(s == GEIST_E_UNSUPPORTED,
                   "captured ffn block unsupported aborts capture");
    fails += check(fake.ffn_block_count == 1,
                   "captured ffn attempts block fastpath once");
    fails += check(fake.rmsnorm_count == 0,
                   "captured ffn does not enter decomposed fallback");

    fake = (struct fake_stage_state){0};
    fill_common_context(&st, &sess, &layer, &ctx, &be, &vtbl);
    ctx.ffn_activation = GEIST_FFN_SWIGLU;
    s = transformer_layer_run_ffn_block(&ctx);
    fails += check(s == GEIST_E_UNSUPPORTED,
                   "captured ineligible ffn aborts capture");
    fails += check(fake.ffn_block_count == 0,
                   "captured ineligible ffn does not attempt block fastpath");
    fails += check(fake.rmsnorm_count == 0,
                   "captured ineligible ffn does not enter decomposed fallback");

    fake = (struct fake_stage_state){0};
    fill_common_context(&st, &sess, &layer, &ctx, &be, &vtbl);
    s = transformer_layer_run_ple_or_copy(&ctx);
    fails += check(s == GEIST_E_UNSUPPORTED,
                   "captured PLE copy fallback aborts capture");
    fails += check(fake.buffer_map_count == 0,
                   "captured PLE copy does not map host buffers");

    fake = (struct fake_stage_state){0};
    fill_common_context(&st, &sess, &layer, &ctx, &be, &vtbl);
    ctx.apply_ple = true;
    ctx.per_layer_input_buf = fake_buf(0x604u);
    s = transformer_layer_run_ple_or_copy(&ctx);
    fails += check(s == GEIST_E_UNSUPPORTED,
                   "captured PLE injection fallback aborts capture");
    fails += check(fake.ple_block_count == 1,
                   "captured PLE injection attempts block fastpath once");
    fails += check(fake.buffer_map_count == 0,
                   "captured PLE injection does not map host buffers");

    fake = (struct fake_stage_state){0};
    fill_common_context(&st, &sess, &layer, &ctx, &be, &vtbl);
    s = transformer_layer_scale_output(&ctx);
    fails += check(s == GEIST_E_UNSUPPORTED,
                   "captured scale fallback aborts capture");
    fails += check(fake.scale_count == 1,
                   "captured scale attempts backend op once");
    fails += check(fake.buffer_map_count == 0,
                   "captured scale does not map host buffers");

    fake = (struct fake_stage_state){0};
    fill_common_context(&st, &sess, &layer, &ctx, &be, &vtbl);
    st.vocab_size = 2;
    st.ple_out = st.n_layers * st.hidden_per_layer;
    st.config.ple_table_scale = 16.0f;
    st.ple_table = (struct geist_tensor){
        .buffer = fake_buf(0x601u),
        .dtype = GEIST_DTYPE_F32,
        .layout = GEIST_LAYOUT_DENSE,
        .ndim = 2,
        .shape = {2, st.ple_out},
        .stride = {st.ple_out, 1},
    };
    sess.scratch_ple_lookup = fake_buf(0x602u);
    s = transformer_compute_per_layer_input(
        &st, 0, ctx.h_in_buf, fake_buf(0x603u));
    fails += check(s == GEIST_E_UNSUPPORTED,
                   "captured PLE precompute fallback aborts capture");
    fails += check(fake.buffer_map_count == 0,
                   "captured PLE precompute does not map host buffers");

    fake = (struct fake_stage_state){0};
    fill_common_context(&st, &sess, &layer, &ctx, &be, &vtbl);
    st.output_norm.buffer = fake_buf(0x701u);
    st.embed_table.buffer = fake_buf(0x702u);
    st.embed_table_w.n_in = st.d_model;
    st.embed_table_w.n_out = 2;
    st.embed_table_w.dtype = GEIST_DTYPE_F32;
    sess.scratch_h_a = fake_buf(0x703u);
    sess.scratch_h_b = fake_buf(0x704u);
    sess.scratch_logits = fake_buf(0x705u);
    st.vocab_size = 2;
    sess.temperature = 0.0f;
    geist_token_t tok = -2;
    s = finalize_logits_one_row(&st, 0, &tok);
    fails += check(s == GEIST_E_UNSUPPORTED,
                   "captured head without greedy fastpath aborts capture");
    fails += check(fake.buffer_map_count == 0,
                   "captured head without greedy fastpath does not map buffers");

    fake = (struct fake_stage_state){0};
    fill_common_context(&st, &sess, &layer, &ctx, &be, &vtbl);
    st.output_norm.buffer = fake_buf(0x711u);
    st.embed_table.buffer = fake_buf(0x712u);
    st.embed_table_w.n_in = st.d_model;
    st.embed_table_w.n_out = 2;
    st.embed_table_w.dtype = GEIST_DTYPE_F32;
    sess.scratch_h_a = fake_buf(0x713u);
    sess.scratch_h_b = fake_buf(0x714u);
    sess.scratch_logits = fake_buf(0x715u);
    st.vocab_size = 2;
    sess.temperature = 0.8f;
    tok = -2;
    s = finalize_logits_one_row(&st, 0, &tok);
    fails += check(s == GEIST_E_UNSUPPORTED,
                   "captured sampling head aborts capture");
    fails += check(fake.buffer_map_count == 0,
                   "captured sampling head does not map buffers");

    fake = (struct fake_stage_state){0};
    be.desc = &sequence_ok_desc;
    fill_common_context(&st, &sess, &layer, &ctx, &be, &sequence_ok_vtbl);
    st.backend = &be;
    sess.backend_command_sequence_active = false;
    sess.m_max = 1;
    struct geist_buffer *layer_k_cache_slots[1] = {fake_buf(0x721u)};
    struct geist_buffer *layer_v_cache_slots[1] = {fake_buf(0x722u)};
    struct geist_buffer *layer_k_cache_q8_slots[1] = {fake_buf(0x723u)};
    struct geist_buffer *layer_v_cache_q8_slots[1] = {fake_buf(0x724u)};
    struct geist_buffer *layer_k_cache_scale_slots[1] = {fake_buf(0x725u)};
    struct geist_buffer *layer_v_cache_scale_slots[1] = {fake_buf(0x726u)};
    struct geist_buffer *layer_k_kivi_q_slots[1] = {fake_buf(0x727u)};
    struct geist_buffer *layer_v_kivi_q_slots[1] = {fake_buf(0x728u)};
    struct geist_buffer *layer_k_kivi_scales_slots[1] = {fake_buf(0x729u)};
    struct geist_buffer *layer_k_kivi_zeros_slots[1] = {fake_buf(0x72au)};
    struct geist_buffer *layer_v_kivi_scales_slots[1] = {fake_buf(0x72bu)};
    struct geist_buffer *layer_v_kivi_zeros_slots[1] = {fake_buf(0x72cu)};
    struct geist_buffer *layer_k_residual_slots[1] = {fake_buf(0x72du)};
    struct geist_buffer *layer_v_residual_slots[1] = {fake_buf(0x72eu)};
    sess.k_cache = layer_k_cache_slots;
    sess.v_cache = layer_v_cache_slots;
    sess.k_cache_q8 = layer_k_cache_q8_slots;
    sess.v_cache_q8 = layer_v_cache_q8_slots;
    sess.k_cache_scale = layer_k_cache_scale_slots;
    sess.v_cache_scale = layer_v_cache_scale_slots;
    sess.k_kivi_q = layer_k_kivi_q_slots;
    sess.v_kivi_q = layer_v_kivi_q_slots;
    sess.k_kivi_scales = layer_k_kivi_scales_slots;
    sess.k_kivi_zeros = layer_k_kivi_zeros_slots;
    sess.v_kivi_scales = layer_v_kivi_scales_slots;
    sess.v_kivi_zeros = layer_v_kivi_zeros_slots;
    sess.k_residual = layer_k_residual_slots;
    sess.v_residual = layer_v_residual_slots;
    st.config.has_gemma_attn_norms = true;
    st.config.ffn_activation = GEIST_FFN_GEGLU;
    s = transformer_forward_one_layer(&st, 0, 0, 1, true,
                                      fake_buf(0x731u), nullptr,
                                      fake_buf(0x732u));
    fails += check(s == GEIST_OK,
                   "per-layer attention+ffn capture succeeds");
    fails += check(fake.sequence_begin_count == 1,
                   "per-layer attention+ffn capture begins once");
    fails += check(fake.sequence_submit_count == 1,
                   "per-layer attention+ffn capture submits once");
    fails += check(fake.sequence_discard_count == 0,
                   "per-layer attention+ffn capture does not discard");
    fails += check(!sess.backend_command_sequence_active,
                   "per-layer attention+ffn capture clears active flag");
    fails += check(fake.attention_block_count == 1,
                   "per-layer capture tries attention block once");
    fails += check(fake.ffn_block_count == 1,
                   "per-layer capture tries ffn block once");
    fails += check(fake.buffer_copy_count == 1,
                   "per-layer PLE copy runs after capture");
    fails += check(fake.scale_count == 1,
                   "per-layer scale runs after capture");
    fails += check(fake.rmsnorm_count == 0,
                   "per-layer capture does not enter host fallback");

    fake = (struct fake_stage_state){0};
    sess.backend_command_sequence_active = false;
    sess.m_max = 2;
    s = transformer_forward_one_layer(&st, 0, 0, 2, true,
                                      fake_buf(0x733u), nullptr,
                                      fake_buf(0x734u));
    fails += check(s == GEIST_OK,
                   "mN per-layer attention+ffn capture succeeds");
    fails += check(fake.sequence_begin_count == 1,
                   "mN per-layer attention+ffn capture begins once");
    fails += check(fake.sequence_submit_count == 1,
                   "mN per-layer attention+ffn capture submits once");
    fails += check(fake.sequence_discard_count == 0,
                   "mN per-layer attention+ffn capture does not discard");
    fails += check(!sess.backend_command_sequence_active,
                   "mN per-layer attention+ffn capture clears active flag");
    fails += check(fake.attention_block_count == 1,
                   "mN per-layer capture tries attention block once");
    fails += check(fake.ffn_block_count == 1,
                   "mN per-layer capture tries ffn block once");
    fails += check(fake.buffer_copy_count == 1,
                   "mN per-layer PLE copy runs after capture");
    fails += check(fake.scale_count == 1,
                   "mN per-layer scale runs after capture");
    fails += check(fake.rmsnorm_count == 0,
                   "mN per-layer capture does not enter host fallback");

    fake = (struct fake_stage_state){0};
    be.desc = &sequence_full_ok_desc;
    fill_common_context(&st, &sess, &layer, &ctx, &be,
                        &sequence_full_ok_vtbl);
    st.backend = &be;
    st.config.has_gemma_attn_norms = true;
    st.config.has_ple = true;
    st.config.ffn_activation = GEIST_FFN_GEGLU;
    sess.backend_command_sequence_active = false;
    sess.m_max = 2;
    sess.k_cache = layer_k_cache_slots;
    sess.v_cache = layer_v_cache_slots;
    sess.k_cache_q8 = layer_k_cache_q8_slots;
    sess.v_cache_q8 = layer_v_cache_q8_slots;
    sess.k_cache_scale = layer_k_cache_scale_slots;
    sess.v_cache_scale = layer_v_cache_scale_slots;
    sess.k_kivi_q = layer_k_kivi_q_slots;
    sess.v_kivi_q = layer_v_kivi_q_slots;
    sess.k_kivi_scales = layer_k_kivi_scales_slots;
    sess.k_kivi_zeros = layer_k_kivi_zeros_slots;
    sess.v_kivi_scales = layer_v_kivi_scales_slots;
    sess.v_kivi_zeros = layer_v_kivi_zeros_slots;
    sess.k_residual = layer_k_residual_slots;
    sess.v_residual = layer_v_residual_slots;
    s = transformer_forward_one_layer(&st, 0, 0, 2, true,
                                      fake_buf(0x741u), fake_buf(0x742u),
                                      fake_buf(0x743u));
    fails += check(s == GEIST_OK,
                   "mN full-layer capture succeeds");
    fails += check(fake.sequence_begin_count == 1,
                   "mN full-layer capture begins once");
    fails += check(fake.sequence_submit_count == 1,
                   "mN full-layer capture submits once");
    fails += check(fake.sequence_discard_count == 0,
                   "mN full-layer capture does not discard");
    fails += check(fake.attention_block_count == 1,
                   "mN full-layer capture tries attention block once");
    fails += check(fake.ffn_block_count == 1,
                   "mN full-layer capture tries ffn block once");
    fails += check(fake.ple_block_count == 1,
                   "mN full-layer capture tries PLE block once");
    fails += check(fake.scale_count == 1,
                   "mN full-layer capture records scale once");
    fails += check(fake.buffer_copy_count == 0,
                   "mN full-layer capture avoids post-capture copy");
    fails += check(fake.rmsnorm_count == 0,
                   "mN full-layer capture does not enter host fallback");
    be.desc = &desc;

    fake = (struct fake_stage_state){0};
    be.desc = &sequence_desc;
    fill_common_context(&st, &sess, &layer, &ctx, &be, &vtbl);
    st.backend = &be;
    bool greedy_accel_session = true;
    sess.backend_command_sequence_active = false;
    sess.accel_session =
        (struct transformer_accel_session *) &greedy_accel_session;
    sess.m_max = 1;
    sess.scratch_h_a = fake_buf(0x801u);
    sess.scratch_h_b = fake_buf(0x802u);
    sess.scratch_h_post_attn = fake_buf(0x803u);
    sess.scratch_h_post_ff = fake_buf(0x804u);
    sess.scratch_ones_headdim_max = fake_buf(0x805u);
    struct geist_buffer *k_cache_slots[1] = {fake_buf(0x808u)};
    struct geist_buffer *v_cache_slots[1] = {fake_buf(0x809u)};
    struct geist_buffer *k_cache_q8_slots[1] = {fake_buf(0x80au)};
    struct geist_buffer *v_cache_q8_slots[1] = {fake_buf(0x80bu)};
    struct geist_buffer *k_cache_scale_slots[1] = {fake_buf(0x80cu)};
    struct geist_buffer *v_cache_scale_slots[1] = {fake_buf(0x80du)};
    struct geist_buffer *k_kivi_q_slots[1] = {fake_buf(0x80eu)};
    struct geist_buffer *v_kivi_q_slots[1] = {fake_buf(0x80fu)};
    struct geist_buffer *k_kivi_scales_slots[1] = {fake_buf(0x810u)};
    struct geist_buffer *k_kivi_zeros_slots[1] = {fake_buf(0x811u)};
    struct geist_buffer *v_kivi_scales_slots[1] = {fake_buf(0x812u)};
    struct geist_buffer *v_kivi_zeros_slots[1] = {fake_buf(0x813u)};
    struct geist_buffer *k_residual_slots[1] = {fake_buf(0x814u)};
    struct geist_buffer *v_residual_slots[1] = {fake_buf(0x815u)};
    sess.k_cache = k_cache_slots;
    sess.v_cache = v_cache_slots;
    sess.k_cache_q8 = k_cache_q8_slots;
    sess.v_cache_q8 = v_cache_q8_slots;
    sess.k_cache_scale = k_cache_scale_slots;
    sess.v_cache_scale = v_cache_scale_slots;
    sess.k_kivi_q = k_kivi_q_slots;
    sess.v_kivi_q = v_kivi_q_slots;
    sess.k_kivi_scales = k_kivi_scales_slots;
    sess.k_kivi_zeros = k_kivi_zeros_slots;
    sess.v_kivi_scales = v_kivi_scales_slots;
    sess.v_kivi_zeros = v_kivi_zeros_slots;
    sess.k_residual = k_residual_slots;
    sess.v_residual = v_residual_slots;
    st.config.has_gemma_attn_norms = true;
    s = transformer_run_all_layers(&st, 0, 1, fake_buf(0x806u),
                                   nullptr, fake_buf(0x807u));
    fails += check(s == GEIST_E_UNSUPPORTED,
                   "direct layer-loop capture aborts unsupported fastpath");
    fails += check(fake.sequence_begin_count == 1,
                   "direct layer-loop capture begins command sequence");
    fails += check(fake.sequence_discard_count == 1,
                   "direct layer-loop capture discards unsupported sequence");
    fails += check(fake.sequence_submit_count == 0,
                   "direct layer-loop capture does not submit unsupported sequence");
    fails += check(!sess.backend_command_sequence_active,
                   "direct layer-loop capture clears active flag after discard");
    fails += check(fake.attention_block_count == 1,
                   "direct layer-loop capture tries attention block once");
    fails += check(fake.rmsnorm_count == 0,
                   "direct layer-loop capture does not enter decomposed fallback");
    be.desc = &desc;

    fake = (struct fake_stage_state){0};
    be.desc = &sequence_full_ok_desc;
    fill_common_context(&st, &sess, &layer, &ctx, &be,
                        &sequence_full_ok_vtbl);
    struct transformer_layer_weights prefill_layers[2] = {layer, layer};
    st.backend = &be;
    st.layers = prefill_layers;
    st.n_layers = 2;
    st.config.has_gemma_attn_norms = true;
    st.config.ffn_activation = GEIST_FFN_GEGLU;
    sess.backend_command_sequence_active = true;
    sess.exec_plan.kv_f16_enabled = true;
    sess.m_max = 2;
    sess.scratch_h_a = fake_buf(0x821u);
    sess.scratch_h_b = fake_buf(0x822u);
    sess.scratch_h_post_attn = fake_buf(0x823u);
    sess.scratch_h_post_ff = fake_buf(0x824u);
    sess.scratch_ones_headdim_max = fake_buf(0x825u);
    struct geist_buffer *prefill_k_cache_slots[2] = {
        fake_buf(0x828u), fake_buf(0x829u)};
    struct geist_buffer *prefill_v_cache_slots[2] = {
        fake_buf(0x82au), fake_buf(0x82bu)};
    struct geist_buffer *prefill_k_cache_q8_slots[2] = {
        fake_buf(0x82cu), fake_buf(0x82du)};
    struct geist_buffer *prefill_v_cache_q8_slots[2] = {
        fake_buf(0x82eu), fake_buf(0x82fu)};
    struct geist_buffer *prefill_k_cache_scale_slots[2] = {
        fake_buf(0x830u), fake_buf(0x831u)};
    struct geist_buffer *prefill_v_cache_scale_slots[2] = {
        fake_buf(0x832u), fake_buf(0x833u)};
    struct geist_buffer *prefill_k_kivi_q_slots[2] = {
        fake_buf(0x834u), fake_buf(0x835u)};
    struct geist_buffer *prefill_v_kivi_q_slots[2] = {
        fake_buf(0x836u), fake_buf(0x837u)};
    struct geist_buffer *prefill_k_kivi_scales_slots[2] = {
        fake_buf(0x838u), fake_buf(0x839u)};
    struct geist_buffer *prefill_k_kivi_zeros_slots[2] = {
        fake_buf(0x83au), fake_buf(0x83bu)};
    struct geist_buffer *prefill_v_kivi_scales_slots[2] = {
        fake_buf(0x83cu), fake_buf(0x83du)};
    struct geist_buffer *prefill_v_kivi_zeros_slots[2] = {
        fake_buf(0x83eu), fake_buf(0x83fu)};
    struct geist_buffer *prefill_k_residual_slots[2] = {
        fake_buf(0x840u), fake_buf(0x841u)};
    struct geist_buffer *prefill_v_residual_slots[2] = {
        fake_buf(0x842u), fake_buf(0x843u)};
    sess.k_cache = prefill_k_cache_slots;
    sess.v_cache = prefill_v_cache_slots;
    sess.k_cache_q8 = prefill_k_cache_q8_slots;
    sess.v_cache_q8 = prefill_v_cache_q8_slots;
    sess.k_cache_scale = prefill_k_cache_scale_slots;
    sess.v_cache_scale = prefill_v_cache_scale_slots;
    sess.k_kivi_q = prefill_k_kivi_q_slots;
    sess.v_kivi_q = prefill_v_kivi_q_slots;
    sess.k_kivi_scales = prefill_k_kivi_scales_slots;
    sess.k_kivi_zeros = prefill_k_kivi_zeros_slots;
    sess.v_kivi_scales = prefill_v_kivi_scales_slots;
    sess.v_kivi_zeros = prefill_v_kivi_zeros_slots;
    sess.k_residual = prefill_k_residual_slots;
    sess.v_residual = prefill_v_residual_slots;
    s = transformer_run_all_layers(&st, 5, 2, fake_buf(0x826u),
                                   nullptr, fake_buf(0x827u));
    fails += check(s == GEIST_OK,
                   "outer prefill F16-KV sequence runs two captured layers");
    fails += check(fake.sequence_begin_count == 0,
                   "outer prefill sequence avoids nested layer capture begin");
    fails += check(fake.sequence_submit_count == 0,
                   "outer prefill sequence avoids nested layer submit");
    fails += check(fake.sequence_discard_count == 0,
                   "outer prefill sequence avoids nested layer discard");
    fails += check(sess.backend_command_sequence_active,
                   "outer prefill sequence remains active for caller");
    fails += check(fake.attention_block_count == 2,
                   "outer prefill sequence records one attention block per layer");
    fails += check(fake.ffn_block_count == 2,
                   "outer prefill sequence records one ffn block per layer");
    fails += check(fake.scale_count == 2,
                   "outer prefill sequence records layer scales");
    fails += check(fake.buffer_copy_count == 4,
                   "outer prefill sequence records copies inside capture");
    fails += check(fake.rmsnorm_count == 0,
                   "outer prefill sequence does not enter host fallback");
    be.desc = &desc;

    if (fails == 0) {
        g_fake_map_state = nullptr;
        printf("PASS: transformer capture layer block boundary\n");
        return GEIST_TEST_PASS;
    }
    g_fake_map_state = nullptr;
    fprintf(stderr, "FAILED: %d check(s)\n", fails);
    return GEIST_TEST_FAIL;
}
