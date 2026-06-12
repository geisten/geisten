/*
 * profile_decode — instruments a single decode step to identify hot ops.
 *
 * Loads a Q4_K_M model, prefills with N tokens, runs ONE decode step
 * with timing of each kernel category. Prints breakdown.
 *
 * Categorisation per decode step (1-token forward through 35 layers):
 *   linear_attn      — Q/K/V/O proj matmuls (~140 calls)
 *   linear_mlp       — gate/up/down proj matmuls (~105 calls)
 *   linear_ple       — input_gate + projection matmuls (~70 calls)
 *   linear_lm_head   — final 1×1536 → 262144 matmul (1 call, but huge)
 *   linear_model_proj — per_layer_model_projection (1 call, 1×1536→8960)
 *   attention        — attention_mqa_causal_kv calls (35 per step)
 *   rmsnorm          — all rmsnorm_fp32 calls
 *   rope             — rope_compute_at + rope_apply
 *   gelu / mul / add — elementwise ops
 *   embedding        — token + PLE table lookups
 */
#include "gguf_reader.h"
#include "gguf_quant.h"
#include "gemma4_kernels.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define HIDDEN 1536
#define NUM_LAYERS 35
#define HIDDEN_PER_LAYER 256
#define PLE_OUT (NUM_LAYERS * HIDDEN_PER_LAYER)
#define N_Q_HEADS 8
#define N_KV_HEADS 1
#define VOCAB 262144
#define LOGIT_SOFTCAP 30.0f
#define RMS_EPS 1e-6f
#define PLE_INPUT_SCALE 0.7071067811865476f
#define PLE_MODEL_PROJ_SCALE 0.02551551815399144f
#define PLE_TABLE_SCALE 16.0f

static double t_linear_attn_us = 0;
static double t_linear_mlp_us = 0;
static double t_linear_ple_us = 0;
static double t_linear_lm_us = 0;
static double t_linear_modelp_us = 0;
static double t_attn_us = 0;
static double t_rmsnorm_us = 0;
static double t_rope_us = 0;
static double t_elementwise_us = 0;
static double t_embed_us = 0;
static int n_linear_calls = 0;
static int n_attn_calls = 0;

static inline double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec * 1e6 + (double) ts.tv_nsec / 1e3;
}

#define TIMED(counter, expr)       \
    do {                           \
        double _t0 = now_us();     \
        expr;                      \
        counter += now_us() - _t0; \
    } while (0)

static const bool LAYER_IS_FULL[NUM_LAYERS] = {
        false, false, false, false, true,  false, false, false, false, true,  false, false,
        false, false, true,  false, false, false, false, true,  false, false, false, false,
        true,  false, false, false, false, true,  false, false, false, false, true,
};
#define KV_SLIDING_SRC 13
#define KV_FULL_SRC 14

typedef struct {
    int layer_idx;
    bool is_full;
    bool is_kv_shared;
    int head_dim, q_out, kv_out, intermediate, sliding_window, n_rotated_dims;
    float rope_theta;
    const float *input_ln_w, *q_proj_w, *k_proj_w, *v_proj_w, *o_proj_w;
    const float *q_norm_w, *k_norm_w;
    const float *post_attn_ln_w, *pre_ff_ln_w;
    const float *gate_w, *up_w, *down_w;
    const float* post_ff_ln_w;
    const float *per_layer_gate_w, *per_layer_proj_w, *post_per_layer_norm_w;
    float layer_scalar;
} LayerW;

typedef struct {
    int head_dim;
    size_t capacity, length;
    float *k, *v;
} KVCache;

static void kv_init(KVCache* c, int hd, size_t cap) {
    c->head_dim = hd;
    c->capacity = cap;
    c->length = 0;
    c->k = (float*) calloc(cap * (size_t) N_KV_HEADS * (size_t) hd, sizeof(float));
    c->v = (float*) calloc(cap * (size_t) N_KV_HEADS * (size_t) hd, sizeof(float));
}
static void kv_free(KVCache* c) {
    free(c->k);
    free(c->v);
    c->k = c->v = nullptr;
}

static float* load_gguf(struct gguf_ctx* c, const char* n, size_t expected) {
    const struct gguf_tensor_t* t = gguf_get_tensor(c, n);
    if (!t) {
        fprintf(stderr, "miss: %s\n", n);
        return nullptr;
    }
    size_t e = gguf_tensor_elem_count(t);
    if (e != expected) {
        fprintf(stderr, "size %zu vs %zu for %s\n", e, expected, n);
        return nullptr;
    }
    return gguf_dequant_to_fp32(t);
}
static void layer_path(char* b, size_t n, int i, const char* s) {
    snprintf(b, n, "blk.%d.%s", i, s);
}
static bool load_layer_w(struct gguf_ctx* ctx, LayerW* L) {
    L->is_full = LAYER_IS_FULL[L->layer_idx];
    L->is_kv_shared = L->layer_idx >= 15;
    L->head_dim = L->is_full ? 512 : 256;
    L->q_out = N_Q_HEADS * L->head_dim;
    L->kv_out = N_KV_HEADS * L->head_dim;
    L->intermediate = L->is_kv_shared ? 12288 : 6144;
    L->sliding_window = L->is_full ? 0 : 512;
    L->rope_theta = L->is_full ? 1000000.0f : 10000.0f;
    L->n_rotated_dims = L->is_full ? 128 : L->head_dim;
    char p[256];
    layer_path(p, sizeof(p), L->layer_idx, "attn_norm.weight");
    L->input_ln_w = load_gguf(ctx, p, HIDDEN);
    layer_path(p, sizeof(p), L->layer_idx, "attn_q.weight");
    L->q_proj_w = load_gguf(ctx, p, (size_t) L->q_out * HIDDEN);
    layer_path(p, sizeof(p), L->layer_idx, "attn_q_norm.weight");
    L->q_norm_w = load_gguf(ctx, p, L->head_dim);
    layer_path(p, sizeof(p), L->layer_idx, "attn_output.weight");
    L->o_proj_w = load_gguf(ctx, p, (size_t) HIDDEN * L->q_out);
    if (!L->is_kv_shared) {
        layer_path(p, sizeof(p), L->layer_idx, "attn_k.weight");
        L->k_proj_w = load_gguf(ctx, p, (size_t) L->kv_out * HIDDEN);
        layer_path(p, sizeof(p), L->layer_idx, "attn_v.weight");
        L->v_proj_w = load_gguf(ctx, p, (size_t) L->kv_out * HIDDEN);
        layer_path(p, sizeof(p), L->layer_idx, "attn_k_norm.weight");
        L->k_norm_w = load_gguf(ctx, p, L->head_dim);
    }
    layer_path(p, sizeof(p), L->layer_idx, "post_attention_norm.weight");
    L->post_attn_ln_w = load_gguf(ctx, p, HIDDEN);
    layer_path(p, sizeof(p), L->layer_idx, "ffn_norm.weight");
    L->pre_ff_ln_w = load_gguf(ctx, p, HIDDEN);
    layer_path(p, sizeof(p), L->layer_idx, "ffn_gate.weight");
    L->gate_w = load_gguf(ctx, p, (size_t) L->intermediate * HIDDEN);
    layer_path(p, sizeof(p), L->layer_idx, "ffn_up.weight");
    L->up_w = load_gguf(ctx, p, (size_t) L->intermediate * HIDDEN);
    layer_path(p, sizeof(p), L->layer_idx, "ffn_down.weight");
    L->down_w = load_gguf(ctx, p, (size_t) HIDDEN * L->intermediate);
    layer_path(p, sizeof(p), L->layer_idx, "post_ffw_norm.weight");
    L->post_ff_ln_w = load_gguf(ctx, p, HIDDEN);
    layer_path(p, sizeof(p), L->layer_idx, "inp_gate.weight");
    L->per_layer_gate_w = load_gguf(ctx, p, (size_t) HIDDEN_PER_LAYER * HIDDEN);
    layer_path(p, sizeof(p), L->layer_idx, "proj.weight");
    L->per_layer_proj_w = load_gguf(ctx, p, (size_t) HIDDEN * HIDDEN_PER_LAYER);
    layer_path(p, sizeof(p), L->layer_idx, "post_norm.weight");
    L->post_per_layer_norm_w = load_gguf(ctx, p, HIDDEN);
    layer_path(p, sizeof(p), L->layer_idx, "layer_output_scale.weight");
    float* ls = load_gguf(ctx, p, 1);
    L->layer_scalar = ls[0];
    free(ls);
    return true;
}

static void forward_layer_kv_profiled(const LayerW* L,
                                      size_t n_q,
                                      size_t q_offset,
                                      const float* h_in,
                                      const float* per_layer_input,
                                      KVCache* kv_self,
                                      const KVCache* kv_shared,
                                      float* h_out) {
    const int hd = L->head_dim, q_out = L->q_out, kv_out = L->kv_out, inter = L->intermediate;
    size_t seq = n_q;
    float* h_pre = (float*) malloc(seq * HIDDEN * sizeof(float));
    memcpy(h_pre, h_in, seq * HIDDEN * sizeof(float));

    float* normed = (float*) malloc(seq * HIDDEN * sizeof(float));
    TIMED(t_rmsnorm_us, rmsnorm_fp32(h_pre, L->input_ln_w, seq, HIDDEN, RMS_EPS, normed));

    float* q = (float*) malloc(seq * (size_t) q_out * sizeof(float));
    TIMED(t_linear_attn_us, linear_fp32(normed, L->q_proj_w, nullptr, seq, HIDDEN, q_out, q));
    n_linear_calls++;
    TIMED(t_rmsnorm_us, rmsnorm_fp32(q, L->q_norm_w, seq * N_Q_HEADS, hd, RMS_EPS, q));

    float* cos_b = (float*) malloc(seq * (size_t) hd * sizeof(float));
    float* sin_b = (float*) malloc(seq * (size_t) hd * sizeof(float));
    TIMED(t_rope_us,
          rope_compute_at(q_offset, seq, hd, L->n_rotated_dims, L->rope_theta, cos_b, sin_b));
    TIMED(t_rope_us, rope_apply(q, cos_b, sin_b, seq, N_Q_HEADS, hd));

    const float *k_full, *v_full;
    size_t n_kv_full;
    if (kv_self) {
        float* k_new = kv_self->k + kv_self->length * (size_t) kv_out;
        float* v_new = kv_self->v + kv_self->length * (size_t) kv_out;
        TIMED(t_linear_attn_us,
              linear_fp32(normed, L->k_proj_w, nullptr, seq, HIDDEN, kv_out, k_new));
        TIMED(t_linear_attn_us,
              linear_fp32(normed, L->v_proj_w, nullptr, seq, HIDDEN, kv_out, v_new));
        n_linear_calls += 2;
        TIMED(t_rmsnorm_us, rmsnorm_fp32(k_new, L->k_norm_w, seq * N_KV_HEADS, hd, RMS_EPS, k_new));
        TIMED(t_rmsnorm_us, rmsnorm_fp32(v_new, nullptr, seq * N_KV_HEADS, hd, RMS_EPS, v_new));
        TIMED(t_rope_us, rope_apply(k_new, cos_b, sin_b, seq, N_KV_HEADS, hd));
        kv_self->length += seq;
        k_full = kv_self->k;
        v_full = kv_self->v;
        n_kv_full = kv_self->length;
    } else {
        k_full = kv_shared->k;
        v_full = kv_shared->v;
        n_kv_full = kv_shared->length;
    }

    float* attn_out = (float*) malloc(seq * (size_t) q_out * sizeof(float));
    TIMED(t_attn_us,
          attention_mqa_causal_kv(q,
                                  k_full,
                                  v_full,
                                  seq,
                                  n_kv_full,
                                  q_offset,
                                  N_Q_HEADS,
                                  N_KV_HEADS,
                                  hd,
                                  (size_t) L->sliding_window,
                                  attn_out));
    n_attn_calls++;

    float* o = (float*) malloc(seq * HIDDEN * sizeof(float));
    TIMED(t_linear_attn_us, linear_fp32(attn_out, L->o_proj_w, nullptr, seq, q_out, HIDDEN, o));
    n_linear_calls++;

    float* post_attn = (float*) malloc(seq * HIDDEN * sizeof(float));
    TIMED(t_rmsnorm_us, rmsnorm_fp32(o, L->post_attn_ln_w, seq, HIDDEN, RMS_EPS, post_attn));
    float* h_post_attn = (float*) malloc(seq * HIDDEN * sizeof(float));
    TIMED(t_elementwise_us, add_fp32(h_pre, post_attn, seq * HIDDEN, h_post_attn));

    float* pre_ff = (float*) malloc(seq * HIDDEN * sizeof(float));
    TIMED(t_rmsnorm_us, rmsnorm_fp32(h_post_attn, L->pre_ff_ln_w, seq, HIDDEN, RMS_EPS, pre_ff));
    float* gate = (float*) malloc(seq * (size_t) inter * sizeof(float));
    float* up = (float*) malloc(seq * (size_t) inter * sizeof(float));
    TIMED(t_linear_mlp_us, linear_fp32(pre_ff, L->gate_w, nullptr, seq, HIDDEN, inter, gate));
    TIMED(t_linear_mlp_us, linear_fp32(pre_ff, L->up_w, nullptr, seq, HIDDEN, inter, up));
    n_linear_calls += 2;
    TIMED(t_elementwise_us, gelu_tanh_fp32(gate, seq * (size_t) inter, gate));
    TIMED(t_elementwise_us, mul_fp32(gate, up, seq * (size_t) inter, gate));
    float* down = (float*) malloc(seq * HIDDEN * sizeof(float));
    TIMED(t_linear_mlp_us, linear_fp32(gate, L->down_w, nullptr, seq, inter, HIDDEN, down));
    n_linear_calls++;

    float* post_ff = (float*) malloc(seq * HIDDEN * sizeof(float));
    TIMED(t_rmsnorm_us, rmsnorm_fp32(down, L->post_ff_ln_w, seq, HIDDEN, RMS_EPS, post_ff));
    float* h_post_ff = (float*) malloc(seq * HIDDEN * sizeof(float));
    TIMED(t_elementwise_us, add_fp32(h_post_attn, post_ff, seq * HIDDEN, h_post_ff));

    float* gate_ple = (float*) malloc(seq * HIDDEN_PER_LAYER * sizeof(float));
    TIMED(t_linear_ple_us,
          linear_fp32(h_post_ff,
                      L->per_layer_gate_w,
                      nullptr,
                      seq,
                      HIDDEN,
                      HIDDEN_PER_LAYER,
                      gate_ple));
    n_linear_calls++;
    TIMED(t_elementwise_us, gelu_tanh_fp32(gate_ple, seq * HIDDEN_PER_LAYER, gate_ple));
    for (size_t t = 0; t < seq; t++) {
        float* g = gate_ple + t * HIDDEN_PER_LAYER;
        const float* p = per_layer_input + t * HIDDEN_PER_LAYER;
        for (size_t i = 0; i < HIDDEN_PER_LAYER; i++)
            g[i] *= p[i];
    }
    float* proj_ple = (float*) malloc(seq * HIDDEN * sizeof(float));
    TIMED(t_linear_ple_us,
          linear_fp32(
                  gate_ple, L->per_layer_proj_w, nullptr, seq, HIDDEN_PER_LAYER, HIDDEN, proj_ple));
    n_linear_calls++;
    TIMED(t_rmsnorm_us,
          rmsnorm_fp32(proj_ple, L->post_per_layer_norm_w, seq, HIDDEN, RMS_EPS, proj_ple));
    TIMED(t_elementwise_us, add_fp32(h_post_ff, proj_ple, seq * HIDDEN, h_out));

    for (size_t i = 0; i < seq * HIDDEN; i++)
        h_out[i] *= L->layer_scalar;

    free(proj_ple);
    free(gate_ple);
    free(h_post_ff);
    free(post_ff);
    free(down);
    free(up);
    free(gate);
    free(pre_ff);
    free(h_post_attn);
    free(post_attn);
    free(o);
    free(attn_out);
    free(cos_b);
    free(sin_b);
    free(q);
    free(normed);
    free(h_pre);
}

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <model.gguf> <n_prefill_tokens>\n", argv[0]);
        return 2;
    }
    int n_prefill = atoi(argv[2]);

    const char* err = nullptr;
    struct gguf_ctx* ctx = gguf_open(argv[1], &err);
    if (!ctx) {
        fprintf(stderr, "%s\n", err);
        return 1;
    }

    const struct gguf_tensor_t* embed_t = gguf_get_tensor(ctx, "token_embd.weight");
    const struct gguf_tensor_t* ple_table_t = gguf_get_tensor(ctx, "per_layer_token_embd.weight");
    fprintf(stderr, "Loading model...\n");
    float* embed_table = gguf_dequant_to_fp32(embed_t);
    float* ple_table_data = gguf_dequant_to_fp32(ple_table_t);
    float* model_proj_w = load_gguf(ctx, "per_layer_model_proj.weight", (size_t) PLE_OUT * HIDDEN);
    float* model_proj_norm_w = load_gguf(ctx, "per_layer_proj_norm.weight", HIDDEN_PER_LAYER);
    float* final_norm_w = load_gguf(ctx, "output_norm.weight", HIDDEN);
    float* lm_head_w = (float*) malloc((size_t) VOCAB * HIDDEN * sizeof(float));
    memcpy(lm_head_w, embed_table, (size_t) VOCAB * HIDDEN * sizeof(float));

    LayerW layers[NUM_LAYERS];
    for (int i = 0; i < NUM_LAYERS; i++) {
        layers[i].layer_idx = i;
        load_layer_w(ctx, &layers[i]);
    }
    fprintf(stderr, "Model loaded.\n");

    /* Random prompt + KV cache init */
    int32_t* ids = (int32_t*) malloc(n_prefill * sizeof(int32_t));
    for (int i = 0; i < n_prefill; i++)
        ids[i] = 2000 + (i * 7) % 100000;

    KVCache caches[NUM_LAYERS] = {0};
    for (int i = 0; i < 15; i++)
        kv_init(&caches[i], layers[i].head_dim, n_prefill + 16);

    /* === Prefill (not measured separately) === */
    const float embed_scale = sqrtf((float) HIDDEN);
    float* h_prompt = (float*) malloc(n_prefill * HIDDEN * sizeof(float));
    for (int t = 0; t < n_prefill; t++) {
        const float* row = embed_table + (size_t) ids[t] * HIDDEN;
        for (int i = 0; i < HIDDEN; i++)
            h_prompt[t * HIDDEN + i] = row[i] * embed_scale;
    }
    float* ple_lookup = (float*) malloc(n_prefill * PLE_OUT * sizeof(float));
    for (int t = 0; t < n_prefill; t++) {
        const float* row = ple_table_data + (size_t) ids[t] * PLE_OUT;
        for (int i = 0; i < PLE_OUT; i++)
            ple_lookup[t * PLE_OUT + i] = row[i] * PLE_TABLE_SCALE;
    }
    float* ple_proj = (float*) malloc(n_prefill * PLE_OUT * sizeof(float));
    linear_fp32(h_prompt, model_proj_w, nullptr, n_prefill, HIDDEN, PLE_OUT, ple_proj);
    for (size_t i = 0; i < (size_t) n_prefill * PLE_OUT; i++)
        ple_proj[i] *= PLE_MODEL_PROJ_SCALE;
    rmsnorm_fp32(ple_proj,
                 model_proj_norm_w,
                 (size_t) n_prefill * NUM_LAYERS,
                 HIDDEN_PER_LAYER,
                 RMS_EPS,
                 ple_proj);
    float* per_layer_inputs_prompt = (float*) malloc(n_prefill * PLE_OUT * sizeof(float));
    for (size_t i = 0; i < (size_t) n_prefill * PLE_OUT; i++)
        per_layer_inputs_prompt[i] = (ple_proj[i] + ple_lookup[i]) * PLE_INPUT_SCALE;
    free(ple_lookup);
    free(ple_proj);

    /* Prefill forward through 35 layers (no profiling) */
    float* h = (float*) malloc(n_prefill * HIDDEN * sizeof(float));
    memcpy(h, h_prompt, n_prefill * HIDDEN * sizeof(float));
    float* h_buf2 = (float*) malloc(n_prefill * HIDDEN * sizeof(float));
    float* ple_slice = (float*) malloc(n_prefill * HIDDEN_PER_LAYER * sizeof(float));
    for (int li = 0; li < NUM_LAYERS; li++) {
        for (int t = 0; t < n_prefill; t++) {
            memcpy(ple_slice + t * HIDDEN_PER_LAYER,
                   per_layer_inputs_prompt + t * PLE_OUT + (size_t) li * HIDDEN_PER_LAYER,
                   HIDDEN_PER_LAYER * sizeof(float));
        }
        KVCache* sc = nullptr;
        const KVCache* shc = nullptr;
        if (layers[li].is_kv_shared) {
            shc = &caches[layers[li].is_full ? KV_FULL_SRC : KV_SLIDING_SRC];
        } else {
            sc = &caches[li];
        }
        forward_layer_kv_profiled(&layers[li], n_prefill, 0, h, ple_slice, sc, shc, h_buf2);
        float* tmp = h;
        h = h_buf2;
        h_buf2 = tmp;
    }
    free(h_buf2);
    free(ple_slice);
    free(h);
    free(h_prompt);
    free(per_layer_inputs_prompt);
    fprintf(stderr, "Prefill done. KV cache lengths populated. Resetting timers.\n");

    /* === Reset timers, do ONE decode step === */
    t_linear_attn_us = t_linear_mlp_us = t_linear_ple_us = t_linear_lm_us = t_linear_modelp_us = 0;
    t_attn_us = t_rmsnorm_us = t_rope_us = t_elementwise_us = t_embed_us = 0;
    n_linear_calls = n_attn_calls = 0;

    /* Compute new-token inputs */
    int32_t next = 5000;
    double t_emb_start = now_us();
    float h_one[HIDDEN];
    const float* row_e = embed_table + (size_t) next * HIDDEN;
    for (int i = 0; i < HIDDEN; i++)
        h_one[i] = row_e[i] * embed_scale;
    float ple_lookup_one[PLE_OUT];
    const float* row_p = ple_table_data + (size_t) next * PLE_OUT;
    for (int i = 0; i < PLE_OUT; i++)
        ple_lookup_one[i] = row_p[i] * PLE_TABLE_SCALE;
    t_embed_us = now_us() - t_emb_start;

    float ple_proj_one[PLE_OUT];
    TIMED(t_linear_modelp_us,
          linear_fp32(h_one, model_proj_w, nullptr, 1, HIDDEN, PLE_OUT, ple_proj_one));
    for (size_t i = 0; i < PLE_OUT; i++)
        ple_proj_one[i] *= PLE_MODEL_PROJ_SCALE;
    TIMED(t_rmsnorm_us,
          rmsnorm_fp32(ple_proj_one,
                       model_proj_norm_w,
                       NUM_LAYERS,
                       HIDDEN_PER_LAYER,
                       RMS_EPS,
                       ple_proj_one));
    float per_layer_inputs[PLE_OUT];
    for (size_t i = 0; i < PLE_OUT; i++)
        per_layer_inputs[i] = (ple_proj_one[i] + ple_lookup_one[i]) * PLE_INPUT_SCALE;

    /* Decode forward */
    float h_in[HIDDEN];
    memcpy(h_in, h_one, sizeof(h_in));
    float h_out[HIDDEN];
    double t_decode_start = now_us();
    for (int li = 0; li < NUM_LAYERS; li++) {
        const float* slice = per_layer_inputs + (size_t) li * HIDDEN_PER_LAYER;
        KVCache* sc = nullptr;
        const KVCache* shc = nullptr;
        if (layers[li].is_kv_shared) {
            shc = &caches[layers[li].is_full ? KV_FULL_SRC : KV_SLIDING_SRC];
        } else {
            sc = &caches[li];
        }
        forward_layer_kv_profiled(&layers[li], 1, (size_t) n_prefill, h_in, slice, sc, shc, h_out);
        memcpy(h_in, h_out, sizeof(h_in));
    }

    /* Final norm + LM head + softcap */
    float h_final[HIDDEN];
    TIMED(t_rmsnorm_us, rmsnorm_fp32(h_in, final_norm_w, 1, HIDDEN, RMS_EPS, h_final));
    float* logits = (float*) malloc((size_t) VOCAB * sizeof(float));
    TIMED(t_linear_lm_us, linear_fp32(h_final, lm_head_w, nullptr, 1, HIDDEN, VOCAB, logits));
    n_linear_calls++;
    /* softcap: t_elementwise */
    double t0_sc = now_us();
    for (size_t i = 0; i < VOCAB; i++)
        logits[i] = tanhf(logits[i] / LOGIT_SOFTCAP) * LOGIT_SOFTCAP;
    t_elementwise_us += now_us() - t0_sc;

    double t_total = now_us() - t_decode_start;

    /* Print profile */
    printf("\n=== Decode Step Profile (1 token, q_offset=%d, model=%s) ===\n", n_prefill, argv[1]);
    printf("Total decode step:   %8.2f ms\n", t_total / 1000.0);
    printf("\nBy category (us, %% of total):\n");
    printf("  linear (attn QKVO):  %8.0f us  (%5.1f%%)  %d calls\n",
           t_linear_attn_us,
           100 * t_linear_attn_us / t_total,
           0);
    printf("  linear (MLP):        %8.0f us  (%5.1f%%)\n",
           t_linear_mlp_us,
           100 * t_linear_mlp_us / t_total);
    printf("  linear (PLE merge):  %8.0f us  (%5.1f%%)\n",
           t_linear_ple_us,
           100 * t_linear_ple_us / t_total);
    printf("  linear (model_proj): %8.0f us  (%5.1f%%)\n",
           t_linear_modelp_us,
           100 * t_linear_modelp_us / t_total);
    printf("  linear (lm_head):    %8.0f us  (%5.1f%%)  1 call (1×1536→%d)\n",
           t_linear_lm_us,
           100 * t_linear_lm_us / t_total,
           VOCAB);
    printf("  attention:           %8.0f us  (%5.1f%%)  %d calls\n",
           t_attn_us,
           100 * t_attn_us / t_total,
           n_attn_calls);
    printf("  rmsnorm:             %8.0f us  (%5.1f%%)\n",
           t_rmsnorm_us,
           100 * t_rmsnorm_us / t_total);
    printf("  rope:                %8.0f us  (%5.1f%%)\n", t_rope_us, 100 * t_rope_us / t_total);
    printf("  elementwise+softcap: %8.0f us  (%5.1f%%)\n",
           t_elementwise_us,
           100 * t_elementwise_us / t_total);
    printf("  embedding lookup:    %8.0f us  (%5.1f%%)\n", t_embed_us, 100 * t_embed_us / t_total);

    double sum_linear = t_linear_attn_us + t_linear_mlp_us + t_linear_ple_us + t_linear_modelp_us +
                        t_linear_lm_us;
    printf("\n  TOTAL linear matmul: %8.0f us  (%5.1f%%)  %d calls\n",
           sum_linear,
           100 * sum_linear / t_total,
           n_linear_calls);

    free(logits);
    for (int i = 0; i < 15; i++)
        kv_free(&caches[i]);
    free(model_proj_w);
    free(model_proj_norm_w);
    free(final_norm_w);
    free(lm_head_w);
    free(embed_table);
    free(ple_table_data);
    free(ids);
    gguf_close(ctx);
    return 0;
}
