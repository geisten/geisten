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

    struct geist_tensor t_h_post_ff_2d = view_2d(st->sess->scratch_h_post_ff,
                                                 ctx->SEQ, st->d_model);
    struct geist_tensor t_h_out_2d = view_2d(ctx->h_out_buf, ctx->SEQ, st->d_model);
    if (ctx->apply_ple && ctx->per_layer_input_buf != nullptr) {
        struct geist_tensor t_gate_ple_2d = view_2d(st->sess->scratch_gate_ple,
                                                    ctx->SEQ, st->hidden_per_layer);
        struct geist_tensor t_ple_in_2d = view_2d(ctx->per_layer_input_buf,
                                                  ctx->SEQ, st->hidden_per_layer);

        s = linear_w_or_legacy(be, v, st->sess->scratch_h_post_ff,
                               st->sess->scratch_gate_ple,
                               &L->per_layer_gate_w, ctx->seq,
                               &t_h_post_ff_2d, &L->per_layer_gate,
                               &t_gate_ple_2d);
        if (s != GEIST_OK) { return s; }
        s = v->gelu_tanh(be, &t_gate_ple_2d, &t_gate_ple_2d);
        if (s != GEIST_OK) { return s; }
        s = v->mul(be, &t_gate_ple_2d, &t_ple_in_2d, &t_gate_ple_2d);
        if (s != GEIST_OK) { return s; }

        struct geist_tensor t_proj_ple_2d = view_2d(st->sess->scratch_proj_ple,
                                                    ctx->SEQ, st->d_model);
        struct geist_tensor t_w_post_per = view_1d(L->post_per_layer_norm.buffer,
                                                   st->d_model);
        s = linear_w_or_legacy(be, v, st->sess->scratch_gate_ple,
                               st->sess->scratch_proj_ple,
                               &L->per_layer_proj_w, ctx->seq,
                               &t_gate_ple_2d, &L->per_layer_proj,
                               &t_proj_ple_2d);
        if (s != GEIST_OK) { return s; }
        s = v->rmsnorm(be, &t_proj_ple_2d, &t_w_post_per, ctx->eps,
                       &t_proj_ple_2d);
        if (s != GEIST_OK) { return s; }
        s = v->add(be, &t_h_post_ff_2d, &t_proj_ple_2d, &t_h_out_2d);
        if (s != GEIST_OK) { return s; }
    } else {
        const size_t bytes = ctx->seq * st->d_model * sizeof(float);
        uint8_t *src = (uint8_t *) v->buffer_map(st->sess->scratch_h_post_ff);
        uint8_t *dst = (uint8_t *) v->buffer_map(ctx->h_out_buf);
        memcpy(dst, src, bytes);
        v->buffer_unmap(st->sess->scratch_h_post_ff);
        v->buffer_unmap(ctx->h_out_buf);
    }
    return GEIST_OK;
}

void transformer_layer_scale_output(struct transformer_layer_forward_ctx *ctx) {
    float *hout = (float *) ctx->v->buffer_map(ctx->h_out_buf);
    const size_t n_total = ctx->seq * ctx->st->d_model;
    for (size_t i = 0; i < n_total; i++) {
        hout[i] *= ctx->L->layer_scalar;
    }
    ctx->v->buffer_unmap(ctx->h_out_buf);
}
