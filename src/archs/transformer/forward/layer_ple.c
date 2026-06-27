/*
 * src/archs/transformer/forward/layer_ple.c - transformer layer PLE and
 * final output scaling.
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "internal.h"
#include <geist_types.h>

#include <stdint.h>
#include <string.h>

enum geist_status transformer_layer_run_ple_or_copy(
    struct transformer_layer_forward_ctx *ctx) {

    struct transformer_arch_state *st = ctx->st;
    struct transformer_layer_weights *L = ctx->L;
    struct geist_backend *be = ctx->be;
    const struct geist_backend_vtbl *v = ctx->v;
    enum geist_status s;
    const bool no_host_fallback = transformer_layer_command_sequence_active(ctx);

    struct geist_tensor t_h_post_ff_2d = view_2d(st->sess->scratch_h_post_ff,
                                                 ctx->SEQ, st->d_model);
    struct geist_tensor t_h_out_2d = view_2d(ctx->h_out_buf, ctx->SEQ, st->d_model);
    if (ctx->apply_ple && ctx->per_layer_input_buf != nullptr) {
        struct geist_tensor t_gate_ple_2d = view_2d(st->sess->scratch_gate_ple,
                                                    ctx->SEQ, st->hidden_per_layer);
        const size_t ple_off =
            ctx->seq == 1u &&
                    ctx->per_layer_input_buf == st->sess->scratch_per_layer_input
                ? (size_t) ctx->layer_idx *
                      (size_t) st->hidden_per_layer * sizeof(float)
                : 0u;
        struct geist_tensor t_ple_in_2d =
            view_2d_at(ctx->per_layer_input_buf, ple_off,
                       ctx->SEQ, st->hidden_per_layer);
        struct geist_tensor t_proj_ple_2d = view_2d(st->sess->scratch_proj_ple,
                                                    ctx->SEQ, st->d_model);
        struct geist_tensor t_w_post_per = view_1d(L->post_per_layer_norm.buffer,
                                                   st->d_model);

        if (v->ple_block != nullptr) {
            const struct geist_backend_ple_block block = {
                .struct_size = sizeof(block),
                .seq = ctx->seq,
                .d_model = st->d_model,
                .hidden_per_layer = st->hidden_per_layer,
                .eps = ctx->eps,
                .hidden = &t_h_post_ff_2d,
                .per_layer_input = &t_ple_in_2d,
                .per_layer_gate_weight = &L->per_layer_gate,
                .per_layer_proj_weight = &L->per_layer_proj,
                .post_per_layer_norm_weight = &t_w_post_per,
                .gate_scratch = &t_gate_ple_2d,
                .proj_scratch = &t_proj_ple_2d,
                .out = &t_h_out_2d,
            };
            s = v->ple_block(be, &block);
            if (s == GEIST_OK) {
                return GEIST_OK;
            }
            if (s != GEIST_E_UNSUPPORTED || no_host_fallback) {
                return s;
            }
        }

        if (no_host_fallback) {
            s = linear_w_no_host_fallback(be, v, ctx->seq,
                                          &t_h_post_ff_2d,
                                          &L->per_layer_gate,
                                          &t_gate_ple_2d);
        } else {
            s = linear_w_or_legacy(be, v, st->sess->scratch_h_post_ff,
                                   st->sess->scratch_gate_ple,
                                   &L->per_layer_gate_w, ctx->seq,
                                   &t_h_post_ff_2d, &L->per_layer_gate,
                                   &t_gate_ple_2d);
        }
        if (s != GEIST_OK) { return s; }
        s = v->gelu_tanh(be, &t_gate_ple_2d, &t_gate_ple_2d);
        if (s != GEIST_OK) { return s; }
        s = v->mul(be, &t_gate_ple_2d, &t_ple_in_2d, &t_gate_ple_2d);
        if (s != GEIST_OK) { return s; }

        if (no_host_fallback) {
            s = linear_w_no_host_fallback(be, v, ctx->seq,
                                          &t_gate_ple_2d,
                                          &L->per_layer_proj,
                                          &t_proj_ple_2d);
        } else {
            s = linear_w_or_legacy(be, v, st->sess->scratch_gate_ple,
                                   st->sess->scratch_proj_ple,
                                   &L->per_layer_proj_w, ctx->seq,
                                   &t_gate_ple_2d, &L->per_layer_proj,
                                   &t_proj_ple_2d);
        }
        if (s != GEIST_OK) { return s; }
        s = v->rmsnorm(be, &t_proj_ple_2d, &t_w_post_per, ctx->eps,
                       &t_proj_ple_2d);
        if (s != GEIST_OK) { return s; }
        s = v->add(be, &t_h_post_ff_2d, &t_proj_ple_2d, &t_h_out_2d);
        if (s != GEIST_OK) { return s; }
    } else {
        const size_t bytes = ctx->seq * st->d_model * sizeof(float);
        if (v->buffer_copy != nullptr) {
            s = v->buffer_copy(ctx->h_out_buf, 0,
                               st->sess->scratch_h_post_ff, 0, bytes);
            if (s != GEIST_OK) { return s; }
        } else {
            if (no_host_fallback) {
                return GEIST_E_UNSUPPORTED;
            }
            uint8_t *src = (uint8_t *) v->buffer_map(st->sess->scratch_h_post_ff);
            uint8_t *dst = (uint8_t *) v->buffer_map(ctx->h_out_buf);
            if (src == nullptr || dst == nullptr) {
                return GEIST_E_BACKEND;
            }
            memcpy(dst, src, bytes);
            v->buffer_unmap(st->sess->scratch_h_post_ff);
            v->buffer_unmap(ctx->h_out_buf);
        }
    }
    return GEIST_OK;
}

enum geist_status transformer_layer_scale_output(
    struct transformer_layer_forward_ctx *ctx) {

    const size_t n_total = ctx->seq * ctx->st->d_model;
    if (ctx->v->scale_f32 != nullptr) {
        struct geist_tensor x = view_2d(ctx->h_out_buf, ctx->seq,
                                        ctx->st->d_model);
        enum geist_status s = ctx->v->scale_f32(ctx->be, &x,
                                                ctx->L->layer_scalar, &x);
        if (s == GEIST_OK) {
            return GEIST_OK;
        }
        if (s != GEIST_E_UNSUPPORTED) {
            return s;
        }
    }

    if (transformer_layer_command_sequence_active(ctx)) {
        return GEIST_E_UNSUPPORTED;
    }

    float *hout = (float *) ctx->v->buffer_map(ctx->h_out_buf);
    if (hout == nullptr) {
        return GEIST_E_BACKEND;
    }
    for (size_t i = 0; i < n_total; i++) {
        hout[i] *= ctx->L->layer_scalar;
    }
    ctx->v->buffer_unmap(ctx->h_out_buf);
    return GEIST_OK;
}
