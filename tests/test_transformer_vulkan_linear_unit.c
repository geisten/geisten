/*
 * test_transformer_vulkan_linear_unit - transformer linear helper Vulkan path.
 *
 * Verifies that the architecture-level linear_w_* helpers can use the
 * device-resident Vulkan F32 matvec primitive for seq==1 without requiring
 * host-mappable activation or output buffers.
 */
#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>
#include <geist_weight.h>

#define GEIST_INTERNAL_ARCH_LAYER
#include "src/archs/transformer/arch_state.h"
#include "src/archs/transformer/forward.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

[[nodiscard]] enum geist_status linear_w_or_legacy(
    struct geist_backend *be,
    const struct geist_backend_vtbl *v,
    struct geist_buffer *x_buf,
    struct geist_buffer *y_buf,
    const struct geist_weight *w,
    size_t seq,
    const struct geist_tensor *t_x,
    const struct geist_tensor *t_w,
    struct geist_tensor *t_y);

[[nodiscard]] enum geist_status linear_w_pair_or_legacy(
    struct geist_backend *be,
    const struct geist_backend_vtbl *v,
    struct geist_buffer *x_buf,
    struct geist_buffer *y0_buf,
    struct geist_buffer *y1_buf,
    const struct geist_weight *w0,
    const struct geist_weight *w1,
    size_t seq,
    const struct geist_tensor *t_x,
    const struct geist_tensor *t_w0,
    const struct geist_tensor *t_w1,
    struct geist_tensor *t_y0,
    struct geist_tensor *t_y1);

[[nodiscard]] enum geist_status linear_w_triple_or_legacy(
    struct geist_backend *be,
    const struct geist_backend_vtbl *v,
    struct geist_buffer *x_buf,
    struct geist_buffer *y0_buf,
    struct geist_buffer *y1_buf,
    struct geist_buffer *y2_buf,
    const struct geist_weight *w0,
    const struct geist_weight *w1,
    const struct geist_weight *w2,
    size_t seq,
    const struct geist_tensor *t_x,
    const struct geist_tensor *t_w0,
    const struct geist_tensor *t_w1,
    const struct geist_tensor *t_w2,
    struct geist_tensor *t_y0,
    struct geist_tensor *t_y1,
    struct geist_tensor *t_y2);

[[nodiscard]] enum geist_status
transformer_head_copy_rows(const struct geist_backend_vtbl *v,
                           struct geist_buffer *dst,
                           const struct geist_buffer *src,
                           size_t row_idx,
                           size_t row_count,
                           size_t row_bytes);

static int check(bool cond, const char *what) {
    if (!cond) {
        fprintf(stderr, "FAIL: %s\n", what);
        return 1;
    }
    return 0;
}

static int compare_bytes(const char *name,
                         size_t n,
                         const uint8_t got[static n],
                         const uint8_t want[static n]) {
    if (memcmp(got, want, n) == 0) {
        return 0;
    }
    for (size_t i = 0; i < n; i++) {
        if (got[i] != want[i]) {
            fprintf(stderr, "FAIL %s: byte %zu got %u expected %u\n",
                    name, i, (unsigned) got[i], (unsigned) want[i]);
            return 1;
        }
    }
    return 1;
}

static struct geist_tensor tensor_2d(struct geist_buffer *buf,
                                     size_t rows,
                                     size_t cols) {
    return (struct geist_tensor){
        .buffer = buf,
        .offset = 0,
        .dtype = GEIST_DTYPE_F32,
        .layout = GEIST_LAYOUT_DENSE,
        .ndim = 2,
        .shape = {(int64_t) rows, (int64_t) cols},
        .stride = {(int64_t) cols, 1},
    };
}

static struct geist_tensor tensor_2d_dtype(struct geist_buffer *buf,
                                           enum geist_dtype dtype,
                                           size_t rows,
                                           size_t cols) {
    return (struct geist_tensor){
        .buffer = buf,
        .offset = 0,
        .dtype = dtype,
        .layout = GEIST_LAYOUT_DENSE,
        .ndim = 2,
        .shape = {(int64_t) rows, (int64_t) cols},
        .stride = {(int64_t) cols, 1},
    };
}

static struct geist_tensor tensor_3d(struct geist_buffer *buf,
                                     size_t d0,
                                     size_t d1,
                                     size_t d2) {
    return (struct geist_tensor){
        .buffer = buf,
        .offset = 0,
        .dtype = GEIST_DTYPE_F32,
        .layout = GEIST_LAYOUT_DENSE,
        .ndim = 3,
        .shape = {(int64_t) d0, (int64_t) d1, (int64_t) d2},
        .stride = {(int64_t) (d1 * d2), (int64_t) d2, 1},
    };
}

static struct geist_tensor tensor_1d(struct geist_buffer *buf, size_t n) {
    return (struct geist_tensor){
        .buffer = buf,
        .offset = 0,
        .dtype = GEIST_DTYPE_F32,
        .layout = GEIST_LAYOUT_DENSE,
        .ndim = 1,
        .shape = {(int64_t) n},
        .stride = {1},
    };
}

static struct geist_tensor tensor_q6k_2d(struct geist_buffer *buf,
                                         size_t rows,
                                         size_t cols) {
    return (struct geist_tensor){
        .buffer = buf,
        .offset = 0,
        .dtype = GEIST_DTYPE_Q6_K,
        .layout = GEIST_LAYOUT_BLOCK_QUANTIZED,
        .ndim = 2,
        .shape = {(int64_t) rows, (int64_t) cols},
        .stride = {0, 0},
    };
}

static struct geist_tensor tensor_q4k_2d(struct geist_buffer *buf,
                                         size_t rows,
                                         size_t cols) {
    return (struct geist_tensor){
        .buffer = buf,
        .offset = 0,
        .dtype = GEIST_DTYPE_Q4_K,
        .layout = GEIST_LAYOUT_BLOCK_QUANTIZED,
        .ndim = 2,
        .shape = {(int64_t) rows, (int64_t) cols},
        .stride = {0, 0},
    };
}

static uint32_t f32_bits(float x) {
    uint32_t bits = 0;
    memcpy(&bits, &x, sizeof(bits));
    return bits;
}

static float bits_f32(uint32_t bits) {
    float x = 0.0f;
    memcpy(&x, &bits, sizeof(x));
    return x;
}

static uint16_t f32_to_bf16_trunc(float x) {
    return (uint16_t) (f32_bits(x) >> 16u);
}

static float bf16_to_f32(uint16_t x) {
    return bits_f32((uint32_t) x << 16u);
}

static uint16_t f32_to_f16(float x) {
    const uint32_t bits = f32_bits(x);
    const uint32_t sign = (bits >> 16u) & 0x8000u;
    int32_t exp = (int32_t) ((bits >> 23u) & 0xffu) - 127 + 15;
    uint32_t mant = bits & 0x7fffffu;

    if (exp <= 0) {
        if (exp < -10) {
            return (uint16_t) sign;
        }
        mant |= 0x800000u;
        const uint32_t shift = (uint32_t) (14 - exp);
        uint32_t half_mant = mant >> shift;
        if (((mant >> (shift - 1u)) & 1u) != 0u) {
            half_mant++;
        }
        return (uint16_t) (sign | half_mant);
    }
    if (exp >= 31) {
        return (uint16_t) (sign | 0x7c00u);
    }

    mant += 0x1000u;
    if ((mant & 0x800000u) != 0u) {
        mant = 0u;
        exp++;
        if (exp >= 31) {
            return (uint16_t) (sign | 0x7c00u);
        }
    }
    return (uint16_t) (sign | ((uint32_t) exp << 10u) | (mant >> 13u));
}

static float f16_to_f32(uint16_t h) {
    const uint32_t sign = ((uint32_t) h & 0x8000u) << 16u;
    const uint32_t exp = ((uint32_t) h >> 10u) & 0x1fu;
    const uint32_t mant = (uint32_t) h & 0x03ffu;

    if (exp == 0u) {
        if (mant == 0u) {
            return bits_f32(sign);
        }
        return (sign != 0u ? -1.0f : 1.0f) * ldexpf((float) mant, -24);
    }
    if (exp == 31u) {
        return bits_f32(sign | 0x7f800000u | (mant << 13u));
    }
    return bits_f32(sign | ((exp + 112u) << 23u) | (mant << 13u));
}

static void pack_q4k_simple(size_t n_out,
                            uint8_t dst[static n_out * 144u]) {
    for (size_t row = 0; row < n_out; row++) {
        uint8_t *b = dst + row * 144u;
        memset(b, 0, 144u);
        b[0] = 0x00u;
        b[1] = 0x3cu;
        b[2] = 0x00u;
        b[3] = 0x00u;
        for (size_t i = 0; i < 4u; i++) {
            b[4u + i] = 1u;
        }
        for (size_t i = 0; i < 4u; i++) {
            b[12u + i] = 1u;
        }
        for (size_t sub = 0; sub < 8u; sub += 2u) {
            for (size_t idx = 0; idx < 32u; idx++) {
                const uint8_t lo =
                    (uint8_t) ((row * 3u + sub * 5u + idx) & 15u);
                const uint8_t hi =
                    (uint8_t) ((row * 7u + sub * 3u + idx + 1u) & 15u);
                b[16u + (sub / 2u) * 32u + idx] =
                    (uint8_t) (lo | (hi << 4u));
            }
        }
    }
}

static void ref_q4k_simple(size_t n_out,
                           const float x[static 256],
                           float y[static n_out]) {
    for (size_t row = 0; row < n_out; row++) {
        double acc = 0.0;
        for (size_t sub = 0; sub < 8u; sub++) {
            for (size_t idx = 0; idx < 32u; idx++) {
                const size_t k = sub * 32u + idx;
                uint8_t q;
                if ((sub & 1u) == 0u) {
                    q = (uint8_t) ((row * 3u + sub * 5u + idx) & 15u);
                } else {
                    q = (uint8_t) ((row * 7u + (sub - 1u) * 3u + idx + 1u) &
                                   15u);
                }
                acc += (double) x[k] * (double) q;
            }
        }
        y[row] = (float) acc;
    }
}

static void encode_f16_array(size_t n,
                             const float src[static n],
                             uint16_t dst[static n],
                             float decoded[static n]) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = f32_to_f16(src[i]);
        decoded[i] = f16_to_f32(dst[i]);
    }
}

static void encode_bf16_array(size_t n,
                              const float src[static n],
                              uint16_t dst[static n],
                              float decoded[static n]) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = f32_to_bf16_trunc(src[i]);
        decoded[i] = bf16_to_f32(dst[i]);
    }
}

static void ref_matvec(size_t n_in, size_t n_out,
                       const float x[static n_in],
                       const float w[static n_in * n_out],
                       float y[static n_out]) {
    for (size_t row = 0; row < n_out; row++) {
        double acc = 0.0;
        for (size_t k = 0; k < n_in; k++) {
            acc += (double) x[k] * (double) w[row * n_in + k];
        }
        y[row] = (float) acc;
    }
}

static void ref_rmsnorm(size_t n,
                        const float x[static n],
                        const float w[static n],
                        float eps,
                        float y[static n]) {
    double sum_sq = 0.0;
    for (size_t i = 0; i < n; i++) {
        sum_sq += (double) x[i] * (double) x[i];
    }
    const float scale = 1.0f / sqrtf((float) (sum_sq / (double) n) + eps);
    for (size_t i = 0; i < n; i++) {
        y[i] = x[i] * scale * w[i];
    }
}

static float ref_gelu_tanh(float x) {
    const float inner = 0.7978845608028654f * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

static void ref_ffn_geglu_block(size_t d,
                                size_t inter,
                                const float residual[static d],
                                const float ffn_norm[static d],
                                const float gate_w[static inter * d],
                                const float up_w[static inter * d],
                                const float down_w[static d * inter],
                                const float post_norm[static d],
                                float eps,
                                float out[static d]) {
    float pre[d];
    float gate[inter];
    float up[inter];
    float ffn_out[d];
    float post[d];

    ref_rmsnorm(d, residual, ffn_norm, eps, pre);
    ref_matvec(d, inter, pre, gate_w, gate);
    ref_matvec(d, inter, pre, up_w, up);
    for (size_t i = 0; i < inter; i++) {
        gate[i] = ref_gelu_tanh(gate[i]) * up[i];
    }
    ref_matvec(inter, d, gate, down_w, ffn_out);
    ref_rmsnorm(d, ffn_out, post_norm, eps, post);
    for (size_t i = 0; i < d; i++) {
        out[i] = residual[i] + post[i];
    }
}

static void ref_rope_identity(size_t n, float x[static n]) {
    (void) n;
    (void) x;
}

static void ref_attention_block(float out[static 4],
                                float appended_k[static 2],
                                float appended_v[static 2]) {
    constexpr size_t d = 4;
    constexpr size_t q_out = 4;
    constexpr size_t kv_out = 2;
    const float eps = 1e-5f;
    const float residual[4] = {0.30f, -0.20f, 0.10f, 0.40f};
    const float attn_norm[4] = {1.00f, 0.90f, 1.10f, 0.80f};
    const float q_w[16] = {
        0.20f, -0.10f, 0.05f, 0.30f,
       -0.15f, 0.25f, 0.10f, -0.05f,
        0.12f, 0.07f, -0.18f, 0.22f,
        0.03f, -0.20f, 0.28f, 0.11f,
    };
    const float k_w[8] = {
        0.16f, -0.08f, 0.12f, 0.04f,
       -0.06f, 0.18f, 0.09f, -0.14f,
    };
    const float v_w[8] = {
       -0.11f, 0.21f, 0.05f, 0.13f,
        0.17f, -0.04f, 0.19f, 0.02f,
    };
    const float q_norm[2] = {0.95f, 1.05f};
    const float k_norm[2] = {1.10f, 0.85f};
    const float v_norm[2] = {1.00f, 1.00f};
    const float o_w[16] = {
        0.25f, -0.05f, 0.18f, 0.07f,
       -0.12f, 0.20f, 0.09f, -0.03f,
        0.04f, 0.11f, -0.16f, 0.24f,
        0.19f, -0.02f, 0.13f, 0.08f,
    };
    const float post_norm[4] = {0.90f, 1.00f, 1.10f, 0.95f};
    const float k_cache0[2] = {0.07f, -0.04f};
    const float v_cache0[2] = {0.03f, 0.08f};

    float normed[d];
    float q[q_out];
    float k[kv_out];
    float v[kv_out];
    float k_cache[4] = {k_cache0[0], k_cache0[1], 0.0f, 0.0f};
    float v_cache[4] = {v_cache0[0], v_cache0[1], 0.0f, 0.0f};
    float attn[q_out];
    float o[d];
    float post[d];

    ref_rmsnorm(d, residual, attn_norm, eps, normed);
    ref_matvec(d, q_out, normed, q_w, q);
    ref_matvec(d, kv_out, normed, k_w, k);
    ref_matvec(d, kv_out, normed, v_w, v);
    ref_rmsnorm(2, q + 0, q_norm, eps, q + 0);
    ref_rmsnorm(2, q + 2, q_norm, eps, q + 2);
    ref_rmsnorm(2, k, k_norm, eps, k);
    ref_rmsnorm(2, v, v_norm, eps, v);
    ref_rope_identity(q_out, q);
    ref_rope_identity(kv_out, k);
    k_cache[2] = k[0];
    k_cache[3] = k[1];
    v_cache[2] = v[0];
    v_cache[3] = v[1];
    appended_k[0] = k[0];
    appended_k[1] = k[1];
    appended_v[0] = v[0];
    appended_v[1] = v[1];

    for (size_t h = 0; h < 2; h++) {
        const float *qh = q + h * 2u;
        float scores[2];
        for (size_t s = 0; s < 2; s++) {
            const float *kh = k_cache + s * 2u;
            scores[s] = qh[0] * kh[0] + qh[1] * kh[1];
        }
        const float max_score = scores[0] > scores[1] ? scores[0] : scores[1];
        const float e0 = expf(scores[0] - max_score);
        const float e1 = expf(scores[1] - max_score);
        const float inv = 1.0f / (e0 + e1);
        const float w0 = e0 * inv;
        const float w1 = e1 * inv;
        attn[h * 2u + 0] = w0 * v_cache[0] + w1 * v_cache[2];
        attn[h * 2u + 1] = w0 * v_cache[1] + w1 * v_cache[3];
    }
    ref_matvec(q_out, d, attn, o_w, o);
    ref_rmsnorm(d, o, post_norm, eps, post);
    for (size_t i = 0; i < d; i++) {
        out[i] = residual[i] + post[i];
    }
}

static void ref_attention_block_weights(
    const float q_w[static 16],
    const float k_w[static 8],
    const float v_w[static 8],
    const float o_w[static 16],
    float out[static 4],
    float appended_k[static 2],
    float appended_v[static 2]) {

    constexpr size_t d = 4;
    constexpr size_t q_out = 4;
    constexpr size_t kv_out = 2;
    const float eps = 1e-5f;
    const float residual[4] = {0.30f, -0.20f, 0.10f, 0.40f};
    const float attn_norm[4] = {1.00f, 0.90f, 1.10f, 0.80f};
    const float q_norm[2] = {0.95f, 1.05f};
    const float k_norm[2] = {1.10f, 0.85f};
    const float v_norm[2] = {1.00f, 1.00f};
    const float post_norm[4] = {0.90f, 1.00f, 1.10f, 0.95f};
    const float k_cache0[2] = {0.07f, -0.04f};
    const float v_cache0[2] = {0.03f, 0.08f};

    float normed[d];
    float q[q_out];
    float k[kv_out];
    float v[kv_out];
    float k_cache[4] = {k_cache0[0], k_cache0[1], 0.0f, 0.0f};
    float v_cache[4] = {v_cache0[0], v_cache0[1], 0.0f, 0.0f};
    float attn[q_out];
    float o[d];
    float post[d];

    ref_rmsnorm(d, residual, attn_norm, eps, normed);
    ref_matvec(d, q_out, normed, q_w, q);
    ref_matvec(d, kv_out, normed, k_w, k);
    ref_matvec(d, kv_out, normed, v_w, v);
    ref_rmsnorm(2, q + 0, q_norm, eps, q + 0);
    ref_rmsnorm(2, q + 2, q_norm, eps, q + 2);
    ref_rmsnorm(2, k, k_norm, eps, k);
    ref_rmsnorm(2, v, v_norm, eps, v);
    ref_rope_identity(q_out, q);
    ref_rope_identity(kv_out, k);
    k_cache[2] = k[0];
    k_cache[3] = k[1];
    v_cache[2] = v[0];
    v_cache[3] = v[1];
    appended_k[0] = k[0];
    appended_k[1] = k[1];
    appended_v[0] = v[0];
    appended_v[1] = v[1];

    for (size_t h = 0; h < 2; h++) {
        const float *qh = q + h * 2u;
        float scores[2];
        for (size_t s = 0; s < 2; s++) {
            const float *kh = k_cache + s * 2u;
            scores[s] = qh[0] * kh[0] + qh[1] * kh[1];
        }
        const float max_score = scores[0] > scores[1] ? scores[0] : scores[1];
        const float e0 = expf(scores[0] - max_score);
        const float e1 = expf(scores[1] - max_score);
        const float inv = 1.0f / (e0 + e1);
        const float w0 = e0 * inv;
        const float w1 = e1 * inv;
        attn[h * 2u + 0] = w0 * v_cache[0] + w1 * v_cache[2];
        attn[h * 2u + 1] = w0 * v_cache[1] + w1 * v_cache[3];
    }
    ref_matvec(q_out, d, attn, o_w, o);
    ref_rmsnorm(d, o, post_norm, eps, post);
    for (size_t i = 0; i < d; i++) {
        out[i] = residual[i] + post[i];
    }
}

static int8_t q6_value(size_t row, size_t stream, size_t idx) {
    return (int8_t) ((int) ((row * 5u + stream * 3u + idx) & 15u) - 8);
}

static void pack_q6k(size_t n_out, uint8_t dst[static n_out * 210u]) {
    for (size_t row = 0; row < n_out; row++) {
        uint8_t *b = dst + row * 210u;
        memset(b, 0, 210u);
        for (size_t i = 0; i < 16u; i++) {
            b[192u + i] = 1u;
        }
        b[208] = 0x00u;
        b[209] = 0x3cu;
        for (size_t half_idx = 0; half_idx < 2u; half_idx++) {
            uint8_t *ql = b + half_idx * 64u;
            uint8_t *qh = b + 128u + half_idx * 32u;
            for (size_t idx = 0; idx < 32u; idx++) {
                const uint8_t q0 = (uint8_t) (q6_value(row, 0u, idx) + 32);
                const uint8_t q1 = (uint8_t) (q6_value(row, 1u, idx) + 32);
                const uint8_t q2 = (uint8_t) (q6_value(row, 2u, idx) + 32);
                const uint8_t q3 = (uint8_t) (q6_value(row, 3u, idx) + 32);
                ql[idx] = (uint8_t) ((q0 & 15u) | ((q2 & 15u) << 4u));
                ql[idx + 32u] =
                    (uint8_t) ((q1 & 15u) | ((q3 & 15u) << 4u));
                qh[idx] = (uint8_t) (((q0 >> 4u) & 3u) |
                                     (((q1 >> 4u) & 3u) << 2u) |
                                     (((q2 >> 4u) & 3u) << 4u) |
                                     (((q3 >> 4u) & 3u) << 6u));
            }
        }
    }
}

static void ref_q6k(size_t n_out,
                    const float x[static 256],
                    float y[static n_out]) {
    for (size_t row = 0; row < n_out; row++) {
        double acc = 0.0;
        for (size_t half_idx = 0; half_idx < 2u; half_idx++) {
            for (size_t stream = 0; stream < 4u; stream++) {
                for (size_t idx = 0; idx < 32u; idx++) {
                    const size_t k = half_idx * 128u + stream * 32u + idx;
                    acc += (double) x[k] * (double) q6_value(row, stream, idx);
                }
            }
        }
        y[row] = (float) acc;
    }
}

static void ref_ffn_geglu_q6_block(const float residual[static 256],
                                   const float ffn_norm[static 256],
                                   float eps,
                                   float out[static 256]) {
    float pre[256];
    float gate[256];
    float up[256];
    float ffn_out[256];

    ref_rmsnorm(256, residual, ffn_norm, eps, pre);
    ref_q6k(256, pre, gate);
    ref_q6k(256, pre, up);
    for (size_t i = 0; i < 256; i++) {
        gate[i] = ref_gelu_tanh(gate[i]) * up[i];
    }
    ref_q6k(256, gate, ffn_out);
    for (size_t i = 0; i < 256; i++) {
        out[i] = residual[i] + ffn_out[i];
    }
}

static int compare_f32(const char *name,
                       size_t n,
                       const float got[static n],
                       const float want[static n]) {
    const ptrdiff_t bad = geist_fp32_close_array(want, got, n, 1e-4f, 2e-5f);
    if (bad < 0) {
        return 0;
    }
    fprintf(stderr, "FAIL %s: idx %td got %.7f expected %.7f\n",
            name, bad, (double) got[bad], (double) want[bad]);
    return 1;
}

static int compare_f32_tol(const char *name,
                           size_t n,
                           const float got[static n],
                           const float want[static n],
                           float rel_tol,
                           float abs_tol) {
    const ptrdiff_t bad = geist_fp32_close_array(want, got, n, rel_tol, abs_tol);
    if (bad < 0) {
        return 0;
    }
    fprintf(stderr, "FAIL %s: idx %td got %.7f expected %.7f\n",
            name, bad, (double) got[bad], (double) want[bad]);
    return 1;
}

int main(void) {
    int fails = 0;

#if defined(GEIST_BACKEND_VULKAN) && GEIST_BACKEND_VULKAN
    fails += check(setenv("GEIST_VULKAN_DYNAMIC_DECODE_PARAMS", "1", 1) == 0,
                   "dynamic decode params env set OK");
#endif

    struct geist_backend *be = nullptr;
    enum geist_status s = geist_backend_create("vulkan", nullptr, nullptr, &be);
#if defined(GEIST_BACKEND_VULKAN) && GEIST_BACKEND_VULKAN
    if (s != GEIST_OK) {
        fails += check(s == GEIST_E_BACKEND,
                       "vulkan runtime unavailability is clean");
        fails += check(be == nullptr, "failed vulkan create leaves null handle");
        if (fails == 0) {
            printf("PASS: transformer Vulkan linear helper skipped (runtime unavailable)\n");
            return GEIST_TEST_PASS;
        }
        return GEIST_TEST_FAIL;
    }
#else
    fails += check(s == GEIST_E_NOT_FOUND, "vulkan backend is not compiled in");
    fails += check(be == nullptr, "failed vulkan create leaves null handle");
    if (fails == 0) {
        printf("PASS: transformer Vulkan linear helper skipped (not compiled)\n");
        return GEIST_TEST_PASS;
    }
    return GEIST_TEST_FAIL;
#endif

    const struct geist_backend_vtbl *v = be->desc->vtbl;
    fails += check(v->matvec_f32_dense != nullptr,
                   "vulkan matvec entrypoint is present");
    fails += check(v->ffn_geglu_block != nullptr,
                   "vulkan FFN GEGLU block entrypoint is present");
    fails += check(v->attention_block != nullptr,
                   "vulkan attention block entrypoint is present");
    fails += check(v->command_sequence_begin != nullptr,
                   "vulkan command sequence begin is present");
    fails += check(v->command_sequence_end != nullptr,
                   "vulkan command sequence end is present");
    fails += check(v->command_sequence_read_token != nullptr,
                   "vulkan command sequence token read is present");
    fails += check(v->embedding_lookup != nullptr,
                   "vulkan embedding lookup is present");
    fails += check(v->embedding_lookup_scaled != nullptr,
                   "vulkan scaled embedding lookup is present");
    fails += check(v->matmul_q4k != nullptr,
                   "vulkan Q4_K matmul entrypoint is present");
    fails += check(v->prepare_weight_layout != nullptr,
                   "vulkan weight-layout prepare entrypoint is present");

    if (v->matmul_q4k != nullptr) {
        constexpr size_t q4_n_in = 256;
        constexpr size_t q4_n_out = 5;
        float q4_x[q4_n_in];
        float q4_y_raw[q4_n_out] = {0};
        float q4_y_nt4[q4_n_out] = {0};
        float q4_ref[q4_n_out] = {0};
        uint8_t q4_w[q4_n_out * 144u];
        for (size_t i = 0; i < q4_n_in; i++) {
            q4_x[i] = ((float) ((i * 17u) % 29u) - 14.0f) * 0.03125f;
        }
        pack_q4k_simple(q4_n_out, q4_w);
        ref_q4k_simple(q4_n_out, q4_x, q4_ref);

        struct geist_buffer *q4_x_b = nullptr;
        struct geist_buffer *q4_w_b = nullptr;
        struct geist_buffer *q4_y_b = nullptr;
        s = v->buffer_create(be, sizeof(q4_x), GEIST_BUFFER_ACTIVATION,
                             GEIST_MEMORY_DEVICE, &q4_x_b);
        if (s == GEIST_OK) {
            s = v->buffer_create(be, sizeof(q4_w), GEIST_BUFFER_WEIGHT,
                                 GEIST_MEMORY_DEVICE, &q4_w_b);
        }
        if (s == GEIST_OK) {
            s = v->buffer_create(be, sizeof(q4_y_raw), GEIST_BUFFER_ACTIVATION,
                                 GEIST_MEMORY_DEVICE, &q4_y_b);
        }
        fails += check(s == GEIST_OK, "Q4_K buffers created");
        if (s == GEIST_OK) {
            s = v->buffer_upload(q4_x_b, sizeof(q4_x), (const uint8_t *) q4_x);
            if (s == GEIST_OK) {
                s = v->buffer_upload(q4_w_b, sizeof(q4_w), q4_w);
            }
            fails += check(s == GEIST_OK, "Q4_K buffers uploaded");
        }
        if (s == GEIST_OK) {
            struct geist_tensor q4_tx = tensor_2d(q4_x_b, 1, q4_n_in);
            struct geist_tensor q4_tw =
                tensor_q4k_2d(q4_w_b, q4_n_out, q4_n_in);
            struct geist_tensor q4_ty = tensor_2d(q4_y_b, 1, q4_n_out);

            unsetenv("GEIST_VULKAN_Q4K_NT4");
            s = v->matmul_q4k(be, &q4_tx, &q4_tw, &q4_ty);
            fails += check(s == GEIST_OK, "Q4_K raw matmul succeeds");
            if (s == GEIST_OK) {
                s = v->buffer_download(sizeof(q4_y_raw),
                                       (uint8_t *) q4_y_raw, q4_y_b);
                fails += check(s == GEIST_OK, "Q4_K raw output download OK");
            }
            setenv("GEIST_VULKAN_Q4K_NT4", "force", 1);
            if (s == GEIST_OK) {
                s = v->prepare_weight_layout(be, &q4_tw);
                fails += check(s == GEIST_OK,
                               "Q4_K ntile4 prepare succeeds");
            }
            if (s == GEIST_OK) {
                s = v->matmul_q4k(be, &q4_tx, &q4_tw, &q4_ty);
                fails += check(s == GEIST_OK, "Q4_K ntile4 matmul succeeds");
            }
            if (s == GEIST_OK) {
                s = v->buffer_download(sizeof(q4_y_nt4),
                                       (uint8_t *) q4_y_nt4, q4_y_b);
                fails += check(s == GEIST_OK,
                               "Q4_K ntile4 output download OK");
            }
            unsetenv("GEIST_VULKAN_Q4K_NT4");
            if (s == GEIST_OK) {
                fails += compare_f32_tol("Q4_K raw matmul", q4_n_out,
                                         q4_y_raw, q4_ref, 1e-4f, 1e-4f);
                fails += compare_f32_tol("Q4_K ntile4 matmul", q4_n_out,
                                         q4_y_nt4, q4_ref, 1e-4f, 1e-4f);
            }
        }
        if (q4_x_b != nullptr) { v->buffer_destroy(be, q4_x_b); }
        if (q4_w_b != nullptr) { v->buffer_destroy(be, q4_w_b); }
        if (q4_y_b != nullptr) { v->buffer_destroy(be, q4_y_b); }
    }

    {
        const uint16_t f16_rows[8] = {
            0x3c00u, 0xc000u, 0x3400u, 0x4400u,
            0x3e00u, 0xb800u, 0x4000u, 0xbc00u,
        };
        const uint16_t bf16_rows[8] = {
            0x3f80u, 0xc000u, 0x3e80u, 0x4080u,
            0x3fc0u, 0xbf00u, 0x4000u, 0xbf80u,
        };
        const float lookup_unscaled_ref[4] = {1.5f, -0.5f, 2.0f, -1.0f};
        const float lookup_ref[4] = {3.0f, -1.0f, 4.0f, -2.0f};
        float f16_lookup_unscaled_got[4] = {0};
        float bf16_lookup_unscaled_got[4] = {0};
        float f16_lookup_captured_got[4] = {0};
        float f16_lookup_got[4] = {0};
        float bf16_lookup_got[4] = {0};
        struct geist_buffer *f16_lookup_w = nullptr;
        struct geist_buffer *bf16_lookup_w = nullptr;
        struct geist_buffer *f16_lookup_unscaled_out = nullptr;
        struct geist_buffer *bf16_lookup_unscaled_out = nullptr;
        struct geist_buffer *f16_lookup_captured_out = nullptr;
        struct geist_buffer *f16_lookup_out = nullptr;
        struct geist_buffer *bf16_lookup_out = nullptr;
        enum geist_status ls = v->buffer_create(
            be, sizeof(f16_rows), GEIST_BUFFER_WEIGHT, GEIST_MEMORY_DEVICE,
            &f16_lookup_w);
        fails += check(ls == GEIST_OK, "F16 lookup weight create OK");
        if (ls == GEIST_OK) {
            ls = v->buffer_create(be, sizeof(bf16_rows), GEIST_BUFFER_WEIGHT,
                                  GEIST_MEMORY_DEVICE, &bf16_lookup_w);
            fails += check(ls == GEIST_OK, "BF16 lookup weight create OK");
        }
        if (ls == GEIST_OK) {
            ls = v->buffer_create(be, sizeof(f16_lookup_got),
                                  GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_DEVICE,
                                  &f16_lookup_out);
            fails += check(ls == GEIST_OK, "F16 lookup output create OK");
        }
        if (ls == GEIST_OK) {
            ls = v->buffer_create(be, sizeof(bf16_lookup_got),
                                  GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_DEVICE,
                                  &bf16_lookup_out);
            fails += check(ls == GEIST_OK, "BF16 lookup output create OK");
        }
        if (ls == GEIST_OK) {
            ls = v->buffer_create(be, sizeof(f16_lookup_unscaled_got),
                                  GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_DEVICE,
                                  &f16_lookup_unscaled_out);
            fails += check(ls == GEIST_OK,
                           "F16 unscaled lookup output create OK");
        }
        if (ls == GEIST_OK) {
            ls = v->buffer_create(be, sizeof(bf16_lookup_unscaled_got),
                                  GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_DEVICE,
                                  &bf16_lookup_unscaled_out);
            fails += check(ls == GEIST_OK,
                           "BF16 unscaled lookup output create OK");
        }
        if (ls == GEIST_OK) {
            ls = v->buffer_create(be, sizeof(f16_lookup_captured_got),
                                  GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_DEVICE,
                                  &f16_lookup_captured_out);
            fails += check(ls == GEIST_OK,
                           "F16 captured lookup output create OK");
        }
        if (ls == GEIST_OK) {
            ls = v->buffer_upload(f16_lookup_w, sizeof(f16_rows),
                                  (const uint8_t *) f16_rows);
            fails += check(ls == GEIST_OK, "F16 lookup weight upload OK");
        }
        if (ls == GEIST_OK) {
            ls = v->buffer_upload(bf16_lookup_w, sizeof(bf16_rows),
                                  (const uint8_t *) bf16_rows);
            fails += check(ls == GEIST_OK, "BF16 lookup weight upload OK");
        }
        if (ls == GEIST_OK) {
            struct geist_tensor f16_table = {
                .buffer = f16_lookup_w,
                .offset = 0,
                .dtype = GEIST_DTYPE_F16,
                .layout = GEIST_LAYOUT_DENSE,
                .ndim = 2,
                .shape = {2, 4},
                .stride = {4, 1},
            };
            struct geist_tensor f16_unscaled_out =
                tensor_1d(f16_lookup_unscaled_out, 4);
            ls = v->embedding_lookup(be, &f16_table, 1,
                                     &f16_unscaled_out);
            fails += check(ls == GEIST_OK,
                           "F16 embedding lookup succeeds");
        }
        if (ls == GEIST_OK) {
            struct geist_tensor bf16_table = {
                .buffer = bf16_lookup_w,
                .offset = 0,
                .dtype = GEIST_DTYPE_BF16,
                .layout = GEIST_LAYOUT_DENSE,
                .ndim = 2,
                .shape = {2, 4},
                .stride = {4, 1},
            };
            struct geist_tensor bf16_unscaled_out =
                tensor_1d(bf16_lookup_unscaled_out, 4);
            ls = v->embedding_lookup(be, &bf16_table, 1,
                                     &bf16_unscaled_out);
            fails += check(ls == GEIST_OK,
                           "BF16 embedding lookup succeeds");
        }
        if (ls == GEIST_OK) {
            struct geist_tensor f16_table = {
                .buffer = f16_lookup_w,
                .offset = 0,
                .dtype = GEIST_DTYPE_F16,
                .layout = GEIST_LAYOUT_DENSE,
                .ndim = 2,
                .shape = {2, 4},
                .stride = {4, 1},
            };
            struct geist_tensor f16_captured_out =
                tensor_1d(f16_lookup_captured_out, 4);
            int capture_token = 0;
            enum geist_status cs = v->command_sequence_begin(
                be, GEIST_COMMAND_SEQUENCE_DECODE_GREEDY_STEP,
                &capture_token);
            fails += check(cs == GEIST_OK,
                           "captured F16 embedding lookup begin OK");
            if (cs == GEIST_OK) {
                cs = v->embedding_lookup(be, &f16_table, 1,
                                         &f16_captured_out);
                fails += check(cs == GEIST_OK,
                               "captured F16 embedding lookup records OK");
            }
            if (capture_token != 0) {
                const bool submit = cs == GEIST_OK;
                enum geist_status end_s =
                    v->command_sequence_end(be, capture_token, submit);
                fails += check(end_s == GEIST_OK,
                               "captured F16 embedding lookup end OK");
                if (cs == GEIST_OK) { cs = end_s; }
            }
            ls = cs;
        }
        if (ls == GEIST_OK) {
            struct geist_tensor f16_table = {
                .buffer = f16_lookup_w,
                .offset = 0,
                .dtype = GEIST_DTYPE_F16,
                .layout = GEIST_LAYOUT_DENSE,
                .ndim = 2,
                .shape = {2, 4},
                .stride = {4, 1},
            };
            struct geist_tensor f16_out = tensor_1d(f16_lookup_out, 4);
            ls = v->embedding_lookup_scaled(be, &f16_table, 1, 2.0f,
                                            &f16_out);
            fails += check(ls == GEIST_OK,
                           "F16 scaled embedding lookup succeeds");
        }
        if (ls == GEIST_OK) {
            struct geist_tensor bf16_table = {
                .buffer = bf16_lookup_w,
                .offset = 0,
                .dtype = GEIST_DTYPE_BF16,
                .layout = GEIST_LAYOUT_DENSE,
                .ndim = 2,
                .shape = {2, 4},
                .stride = {4, 1},
            };
            struct geist_tensor bf16_out = tensor_1d(bf16_lookup_out, 4);
            ls = v->embedding_lookup_scaled(be, &bf16_table, 1, 2.0f,
                                            &bf16_out);
            fails += check(ls == GEIST_OK,
                           "BF16 scaled embedding lookup succeeds");
        }
        if (ls == GEIST_OK) {
            ls = v->buffer_download(sizeof(f16_lookup_got),
                                    (uint8_t *) f16_lookup_got,
                                    f16_lookup_out);
            fails += check(ls == GEIST_OK, "F16 lookup output download OK");
        }
        if (ls == GEIST_OK) {
            ls = v->buffer_download(sizeof(f16_lookup_unscaled_got),
                                    (uint8_t *) f16_lookup_unscaled_got,
                                    f16_lookup_unscaled_out);
            fails += check(ls == GEIST_OK,
                           "F16 unscaled lookup output download OK");
        }
        if (ls == GEIST_OK) {
            ls = v->buffer_download(sizeof(f16_lookup_captured_got),
                                    (uint8_t *) f16_lookup_captured_got,
                                    f16_lookup_captured_out);
            fails += check(ls == GEIST_OK,
                           "F16 captured lookup output download OK");
        }
        if (ls == GEIST_OK) {
            ls = v->buffer_download(sizeof(bf16_lookup_got),
                                    (uint8_t *) bf16_lookup_got,
                                    bf16_lookup_out);
            fails += check(ls == GEIST_OK, "BF16 lookup output download OK");
        }
        if (ls == GEIST_OK) {
            ls = v->buffer_download(sizeof(bf16_lookup_unscaled_got),
                                    (uint8_t *) bf16_lookup_unscaled_got,
                                    bf16_lookup_unscaled_out);
            fails += check(ls == GEIST_OK,
                           "BF16 unscaled lookup output download OK");
        }
        if (ls == GEIST_OK) {
            fails += compare_f32_tol("F16 embedding lookup", 4,
                                     f16_lookup_unscaled_got,
                                     lookup_unscaled_ref,
                                     1e-6f, 1e-6f);
            fails += compare_f32_tol("BF16 embedding lookup", 4,
                                     bf16_lookup_unscaled_got,
                                     lookup_unscaled_ref,
                                     1e-6f, 1e-6f);
            fails += compare_f32_tol("captured F16 embedding lookup", 4,
                                     f16_lookup_captured_got,
                                     lookup_unscaled_ref,
                                     1e-6f, 1e-6f);
            fails += compare_f32_tol("F16 scaled embedding lookup", 4,
                                     f16_lookup_got, lookup_ref,
                                     1e-6f, 1e-6f);
            fails += compare_f32_tol("BF16 scaled embedding lookup", 4,
                                     bf16_lookup_got, lookup_ref,
                                     1e-6f, 1e-6f);
        }
        if (f16_lookup_w != nullptr) { v->buffer_destroy(be, f16_lookup_w); }
        if (bf16_lookup_w != nullptr) { v->buffer_destroy(be, bf16_lookup_w); }
        if (f16_lookup_unscaled_out != nullptr) {
            v->buffer_destroy(be, f16_lookup_unscaled_out);
        }
        if (bf16_lookup_unscaled_out != nullptr) {
            v->buffer_destroy(be, bf16_lookup_unscaled_out);
        }
        if (f16_lookup_captured_out != nullptr) {
            v->buffer_destroy(be, f16_lookup_captured_out);
        }
        if (f16_lookup_out != nullptr) { v->buffer_destroy(be, f16_lookup_out); }
        if (bf16_lookup_out != nullptr) { v->buffer_destroy(be, bf16_lookup_out); }
    }

    {
        constexpr size_t q5_d = 256;
        constexpr size_t q5_row_bytes = 176;
        uint8_t q5_rows[2 * q5_row_bytes];
        float q5_lookup_got[q5_d];
        float q5_lookup_ref[q5_d];
        memset(q5_rows, 0, sizeof(q5_rows));
        memset(q5_lookup_got, 0, sizeof(q5_lookup_got));

        uint8_t *row = q5_rows + q5_row_bytes;
        const uint16_t one_f16 = f32_to_f16(1.0f);
        memcpy(row, &one_f16, sizeof(one_f16));
        row[4] = 1u; row[5] = 1u; row[6] = 1u; row[7] = 1u;
        row[12] = 1u; row[13] = 1u; row[14] = 1u; row[15] = 1u;
        for (size_t sub = 0; sub < 8; sub++) {
            for (size_t idx = 0; idx < 32; idx++) {
                const size_t k = sub * 32u + idx;
                const uint8_t low = (uint8_t) ((idx + sub) & 15u);
                const uint8_t high = (uint8_t) ((idx + sub) & 1u);
                uint8_t *qs = row + 48u + (sub / 2u) * 32u + idx;
                if ((sub & 1u) == 0u) {
                    *qs = (uint8_t) ((*qs & 0xf0u) | low);
                } else {
                    *qs = (uint8_t) ((*qs & 0x0fu) | (low << 4u));
                }
                row[16u + idx] =
                    (uint8_t) (row[16u + idx] | (high << sub));
                q5_lookup_ref[k] = 2.0f * (float) (low + high * 16u);
            }
        }

        struct geist_buffer *q5_lookup_w = nullptr;
        struct geist_buffer *q5_lookup_out = nullptr;
        enum geist_status qs = v->buffer_create(
            be, sizeof(q5_rows), GEIST_BUFFER_WEIGHT, GEIST_MEMORY_DEVICE,
            &q5_lookup_w);
        fails += check(qs == GEIST_OK, "Q5_K lookup weight create OK");
        if (qs == GEIST_OK) {
            qs = v->buffer_create(be, sizeof(q5_lookup_got),
                                  GEIST_BUFFER_ACTIVATION,
                                  GEIST_MEMORY_DEVICE, &q5_lookup_out);
            fails += check(qs == GEIST_OK, "Q5_K lookup output create OK");
        }
        if (qs == GEIST_OK) {
            qs = v->buffer_upload(q5_lookup_w, sizeof(q5_rows), q5_rows);
            fails += check(qs == GEIST_OK, "Q5_K lookup weight upload OK");
        }
        if (qs == GEIST_OK) {
            struct geist_tensor q5_table = {
                .buffer = q5_lookup_w,
                .offset = 0,
                .dtype = GEIST_DTYPE_Q5_K,
                .layout = GEIST_LAYOUT_BLOCK_QUANTIZED,
                .ndim = 2,
                .shape = {2, q5_d},
            };
            struct geist_tensor q5_out = tensor_1d(q5_lookup_out, q5_d);
            qs = v->embedding_lookup_scaled(be, &q5_table, 1, 2.0f,
                                            &q5_out);
            fails += check(qs == GEIST_OK,
                           "Q5_K scaled embedding lookup succeeds");
        }
        if (qs == GEIST_OK) {
            qs = v->buffer_download(sizeof(q5_lookup_got),
                                    (uint8_t *) q5_lookup_got,
                                    q5_lookup_out);
            fails += check(qs == GEIST_OK, "Q5_K lookup output download OK");
        }
        if (qs == GEIST_OK) {
            fails += compare_f32_tol("Q5_K scaled embedding lookup", q5_d,
                                     q5_lookup_got, q5_lookup_ref,
                                     1e-6f, 1e-6f);
        }
        if (q5_lookup_w != nullptr) { v->buffer_destroy(be, q5_lookup_w); }
        if (q5_lookup_out != nullptr) {
            v->buffer_destroy(be, q5_lookup_out);
        }
    }

    {
        constexpr size_t ple_d = 4;
        constexpr size_t ple_layers = 2;
        constexpr size_t ple_hidden_per_layer = 2;
        constexpr size_t ple_out = ple_layers * ple_hidden_per_layer;
        const float ple_h[ple_d] = {1.0f, 2.0f, -1.0f, 0.5f};
        const uint16_t ple_table_bf16[2 * ple_out] = {
            0x3f80u, 0xc000u, 0x3e80u, 0x4080u,
            0x3fc0u, 0xbf00u, 0x4000u, 0xbf80u,
        };
        const float ple_model_proj[ple_out * ple_d] = {
            1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 1.0f,
        };
        const float ple_norm[ple_hidden_per_layer] = {1.0f, 1.0f};
        float ple_ref[ple_out];
        float ple_proj_ref[ple_out];
        float ple_normed0[ple_hidden_per_layer];
        float ple_normed1[ple_hidden_per_layer];
        float ple_got[ple_out] = {0};
        ref_matvec(ple_d, ple_out, ple_h, ple_model_proj, ple_proj_ref);
        ref_rmsnorm(ple_hidden_per_layer, ple_proj_ref, ple_norm, 1.0e-6f,
                    ple_normed0);
        ref_rmsnorm(ple_hidden_per_layer,
                    ple_proj_ref + ple_hidden_per_layer, ple_norm, 1.0e-6f,
                    ple_normed1);
        ple_ref[0] = ple_normed0[0] + 3.0f;
        ple_ref[1] = ple_normed0[1] - 1.0f;
        ple_ref[2] = ple_normed1[0] + 4.0f;
        ple_ref[3] = ple_normed1[1] - 2.0f;

        struct geist_buffer *ple_h_b = nullptr;
        struct geist_buffer *ple_table_b = nullptr;
        struct geist_buffer *ple_model_proj_b = nullptr;
        struct geist_buffer *ple_norm_b = nullptr;
        struct geist_buffer *ple_lookup_b = nullptr;
        struct geist_buffer *ple_out_b = nullptr;
        enum geist_status ps = v->buffer_create(
            be, sizeof(ple_h), GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_DEVICE,
            &ple_h_b);
        fails += check(ps == GEIST_OK, "captured PLE h create OK");
        if (ps == GEIST_OK) {
            ps = v->buffer_create(be, sizeof(ple_table_bf16),
                                  GEIST_BUFFER_WEIGHT, GEIST_MEMORY_DEVICE,
                                  &ple_table_b);
            fails += check(ps == GEIST_OK, "captured PLE table create OK");
        }
        if (ps == GEIST_OK) {
            ps = v->buffer_create(be, sizeof(ple_model_proj),
                                  GEIST_BUFFER_WEIGHT, GEIST_MEMORY_DEVICE,
                                  &ple_model_proj_b);
            fails += check(ps == GEIST_OK, "captured PLE model_proj create OK");
        }
        if (ps == GEIST_OK) {
            ps = v->buffer_create(be, sizeof(ple_norm),
                                  GEIST_BUFFER_WEIGHT, GEIST_MEMORY_DEVICE,
                                  &ple_norm_b);
            fails += check(ps == GEIST_OK, "captured PLE norm create OK");
        }
        if (ps == GEIST_OK) {
            ps = v->buffer_create(be, ple_out * sizeof(float),
                                  GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_DEVICE,
                                  &ple_lookup_b);
            fails += check(ps == GEIST_OK, "captured PLE lookup create OK");
        }
        if (ps == GEIST_OK) {
            ps = v->buffer_create(be, ple_out * sizeof(float),
                                  GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_DEVICE,
                                  &ple_out_b);
            fails += check(ps == GEIST_OK, "captured PLE output create OK");
        }
        if (ps == GEIST_OK) {
            ps = v->buffer_upload(ple_h_b, sizeof(ple_h),
                                  (const uint8_t *) ple_h);
            fails += check(ps == GEIST_OK, "captured PLE h upload OK");
        }
        if (ps == GEIST_OK) {
            ps = v->buffer_upload(ple_table_b, sizeof(ple_table_bf16),
                                  (const uint8_t *) ple_table_bf16);
            fails += check(ps == GEIST_OK, "captured PLE table upload OK");
        }
        if (ps == GEIST_OK) {
            ps = v->buffer_upload(ple_model_proj_b, sizeof(ple_model_proj),
                                  (const uint8_t *) ple_model_proj);
            fails += check(ps == GEIST_OK, "captured PLE model_proj upload OK");
        }
        if (ps == GEIST_OK) {
            ps = v->buffer_upload(ple_norm_b, sizeof(ple_norm),
                                  (const uint8_t *) ple_norm);
            fails += check(ps == GEIST_OK, "captured PLE norm upload OK");
        }
        if (ps == GEIST_OK) {
            struct transformer_arch_session ple_sess = {
                .backend_command_sequence_active = true,
                .scratch_ple_lookup = ple_lookup_b,
            };
            struct transformer_arch_state ple_state = {
                .backend = be,
                .sess = &ple_sess,
                .config = {
                    .rms_eps = 1.0e-6f,
                    .ple_table_scale = 2.0f,
                    .ple_model_proj_scale = 1.0f,
                    .ple_input_scale = 1.0f,
                },
                .n_layers = ple_layers,
                .d_model = ple_d,
                .vocab_size = 2,
                .hidden_per_layer = ple_hidden_per_layer,
                .ple_out = ple_out,
                .ple_table = {
                    .buffer = ple_table_b,
                    .offset = 0,
                    .dtype = GEIST_DTYPE_BF16,
                    .layout = GEIST_LAYOUT_DENSE,
                    .ndim = 2,
                    .shape = {2, (int64_t) ple_out},
                    .stride = {(int64_t) ple_out, 1},
                },
                .model_proj = {
                    .buffer = ple_model_proj_b,
                    .offset = 0,
                    .dtype = GEIST_DTYPE_F32,
                    .layout = GEIST_LAYOUT_DENSE,
                    .ndim = 2,
                    .shape = {(int64_t) ple_out, (int64_t) ple_d},
                    .stride = {(int64_t) ple_d, 1},
                },
                .model_proj_norm = {
                    .buffer = ple_norm_b,
                    .offset = 0,
                    .dtype = GEIST_DTYPE_F32,
                    .layout = GEIST_LAYOUT_DENSE,
                    .ndim = 1,
                    .shape = {(int64_t) ple_hidden_per_layer},
                    .stride = {1},
                },
            };
            int token = 0;
            ps = v->command_sequence_begin(
                be, GEIST_COMMAND_SEQUENCE_DECODE_GREEDY_STEP, &token);
            fails += check(ps == GEIST_OK,
                           "captured PLE command sequence begin OK");
            if (ps == GEIST_OK) {
                ps = transformer_compute_per_layer_input(
                    &ple_state, 1, ple_h_b, ple_out_b);
                fails += check(ps == GEIST_OK,
                               "captured PLE precompute records OK");
            }
            if (token != 0) {
                const bool submit = ps == GEIST_OK;
                enum geist_status end_s =
                    v->command_sequence_end(be, token, submit);
                fails += check(end_s == GEIST_OK,
                               "captured PLE command sequence end OK");
                if (ps == GEIST_OK) {
                    ps = end_s;
                }
            }
            ple_sess.backend_command_sequence_active = false;
        }
        if (ps == GEIST_OK) {
            ps = v->buffer_download(sizeof(ple_got), (uint8_t *) ple_got,
                                    ple_out_b);
            fails += check(ps == GEIST_OK, "captured PLE output download OK");
        }
        if (ps == GEIST_OK) {
            fails += compare_f32_tol("captured PLE precompute", ple_out,
                                     ple_got, ple_ref, 1e-5f, 1e-5f);
        }
        if (ple_h_b != nullptr) { v->buffer_destroy(be, ple_h_b); }
        if (ple_table_b != nullptr) { v->buffer_destroy(be, ple_table_b); }
        if (ple_model_proj_b != nullptr) {
            v->buffer_destroy(be, ple_model_proj_b);
        }
        if (ple_norm_b != nullptr) { v->buffer_destroy(be, ple_norm_b); }
        if (ple_lookup_b != nullptr) { v->buffer_destroy(be, ple_lookup_b); }
        if (ple_out_b != nullptr) { v->buffer_destroy(be, ple_out_b); }
    }

    {
        const float capture_x[4] = {1.0f, -2.0f, 0.5f, 4.0f};
        const float capture_bias[4] = {0.25f, 0.5f, -1.0f, 2.0f};
        const float capture_ref[4] = {2.25f, -3.5f, 0.0f, 10.0f};
        float capture_got[4] = {0};
        struct geist_buffer *capture_x_b = nullptr;
        struct geist_buffer *capture_bias_b = nullptr;
        struct geist_buffer *capture_mid_b = nullptr;
        struct geist_buffer *capture_out_b = nullptr;
        enum geist_status cs = v->buffer_create(
            be, sizeof(capture_x), GEIST_BUFFER_ACTIVATION,
            GEIST_MEMORY_DEVICE, &capture_x_b);
        fails += check(cs == GEIST_OK, "capture x buffer create OK");
        if (cs == GEIST_OK) {
            cs = v->buffer_create(be, sizeof(capture_bias),
                                  GEIST_BUFFER_ACTIVATION,
                                  GEIST_MEMORY_DEVICE, &capture_bias_b);
            fails += check(cs == GEIST_OK, "capture bias buffer create OK");
        }
        if (cs == GEIST_OK) {
            cs = v->buffer_create(be, sizeof(capture_x),
                                  GEIST_BUFFER_ACTIVATION,
                                  GEIST_MEMORY_DEVICE, &capture_mid_b);
            fails += check(cs == GEIST_OK, "capture mid buffer create OK");
        }
        if (cs == GEIST_OK) {
            cs = v->buffer_create(be, sizeof(capture_x),
                                  GEIST_BUFFER_ACTIVATION,
                                  GEIST_MEMORY_DEVICE, &capture_out_b);
            fails += check(cs == GEIST_OK, "capture out buffer create OK");
        }
        if (cs == GEIST_OK) {
            cs = v->buffer_upload(capture_x_b, sizeof(capture_x),
                                  (const uint8_t *) capture_x);
            fails += check(cs == GEIST_OK, "capture x upload OK");
        }
        if (cs == GEIST_OK) {
            cs = v->buffer_upload(capture_bias_b, sizeof(capture_bias),
                                  (const uint8_t *) capture_bias);
            fails += check(cs == GEIST_OK, "capture bias upload OK");
        }
        if (cs == GEIST_OK) {
            struct geist_tensor t_x = tensor_1d(capture_x_b, 4);
            struct geist_tensor t_bias = tensor_1d(capture_bias_b, 4);
            struct geist_tensor t_mid = tensor_1d(capture_mid_b, 4);
            struct geist_tensor t_out = tensor_1d(capture_out_b, 4);
            int token = 0;
            cs = v->command_sequence_begin(
                be, GEIST_COMMAND_SEQUENCE_DECODE_GREEDY_STEP, &token);
            fails += check(cs == GEIST_OK, "command sequence begin OK");
            if (cs == GEIST_OK) {
                cs = v->scale_f32(be, &t_x, 2.0f, &t_mid);
                fails += check(cs == GEIST_OK,
                               "captured scale_f32 records OK");
            }
            if (cs == GEIST_OK) {
                cs = v->add(be, &t_mid, &t_bias, &t_out);
                fails += check(cs == GEIST_OK, "captured add records OK");
            }
            if (token != 0) {
                const bool submit = cs == GEIST_OK;
                enum geist_status end_s =
                    v->command_sequence_end(be, token, submit);
                fails += check(end_s == GEIST_OK,
                               "command sequence end OK");
                if (cs == GEIST_OK) {
                    cs = end_s;
                }
            }
        }
        if (cs == GEIST_OK) {
            cs = v->buffer_download(sizeof(capture_got),
                                    (uint8_t *) capture_got,
                                    capture_out_b);
            fails += check(cs == GEIST_OK, "capture output download OK");
        }
        if (cs == GEIST_OK) {
            fails += compare_f32_tol("command sequence scale+add", 4,
                                     capture_got, capture_ref,
                                     1e-6f, 1e-6f);
        }
        if (capture_x_b != nullptr) { v->buffer_destroy(be, capture_x_b); }
        if (capture_bias_b != nullptr) { v->buffer_destroy(be, capture_bias_b); }
        if (capture_mid_b != nullptr) { v->buffer_destroy(be, capture_mid_b); }
        if (capture_out_b != nullptr) { v->buffer_destroy(be, capture_out_b); }
    }

    {
        const float long_capture_x[4] = {0.5f, -1.25f, 3.0f, 8.0f};
        float long_capture_got[4] = {0};
        struct geist_buffer *long_capture_x_b = nullptr;
        struct geist_buffer *long_capture_out_b = nullptr;
        enum geist_status cs = v->buffer_create(
            be, sizeof(long_capture_x), GEIST_BUFFER_ACTIVATION,
            GEIST_MEMORY_DEVICE, &long_capture_x_b);
        fails += check(cs == GEIST_OK, "long capture x buffer create OK");
        if (cs == GEIST_OK) {
            cs = v->buffer_create(be, sizeof(long_capture_x),
                                  GEIST_BUFFER_ACTIVATION,
                                  GEIST_MEMORY_DEVICE,
                                  &long_capture_out_b);
            fails += check(cs == GEIST_OK,
                           "long capture out buffer create OK");
        }
        if (cs == GEIST_OK) {
            cs = v->buffer_upload(long_capture_x_b, sizeof(long_capture_x),
                                  (const uint8_t *) long_capture_x);
            fails += check(cs == GEIST_OK, "long capture x upload OK");
        }
        if (cs == GEIST_OK) {
            struct geist_tensor t_x = tensor_1d(long_capture_x_b, 4);
            struct geist_tensor t_out = tensor_1d(long_capture_out_b, 4);
            int token = 0;
            cs = v->command_sequence_begin(
                be, GEIST_COMMAND_SEQUENCE_DECODE_GREEDY_STEP, &token);
            fails += check(cs == GEIST_OK,
                           "long command sequence begin OK");
            constexpr size_t op_count = 300;
            for (size_t i = 0; i < op_count && cs == GEIST_OK; i++) {
                cs = v->scale_f32(be, &t_x, 1.0f, &t_out);
            }
            fails += check(cs == GEIST_OK,
                           "long command sequence records 300 ops");
            if (token != 0) {
                const bool submit = cs == GEIST_OK;
                enum geist_status end_s =
                    v->command_sequence_end(be, token, submit);
                fails += check(end_s == GEIST_OK,
                               "long command sequence end OK");
                if (cs == GEIST_OK) {
                    cs = end_s;
                }
            }
        }
        if (cs == GEIST_OK) {
            cs = v->buffer_download(sizeof(long_capture_got),
                                    (uint8_t *) long_capture_got,
                                    long_capture_out_b);
            fails += check(cs == GEIST_OK,
                           "long capture output download OK");
        }
        if (cs == GEIST_OK) {
            fails += compare_f32_tol("long command sequence scale", 4,
                                     long_capture_got, long_capture_x,
                                     1e-6f, 1e-6f);
        }
        if (long_capture_x_b != nullptr) {
            v->buffer_destroy(be, long_capture_x_b);
        }
        if (long_capture_out_b != nullptr) {
            v->buffer_destroy(be, long_capture_out_b);
        }
    }

    fails += check(transformer_head_copy_rows(v, nullptr, nullptr,
                                              0, 1, 16) == GEIST_E_INVALID_ARG,
                   "head row copy rejects null buffers");

    constexpr size_t copy_row_bytes = 16;
    constexpr size_t copy_rows = 4;
    constexpr size_t copy_out_rows = 2;
    uint8_t copy_src_data[copy_rows * copy_row_bytes];
    uint8_t copy_got[copy_out_rows * copy_row_bytes];
    for (size_t i = 0; i < sizeof(copy_src_data); i++) {
        copy_src_data[i] = (uint8_t) (17u + i * 3u);
    }
    memset(copy_got, 0, sizeof(copy_got));

    struct geist_buffer *copy_src = nullptr;
    struct geist_buffer *copy_dst = nullptr;
    if (s == GEIST_OK) {
        s = v->buffer_create(be, sizeof(copy_src_data),
                             GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_DEVICE,
                             &copy_src);
        fails += check(s == GEIST_OK, "device head-copy src create OK");
    }
    if (s == GEIST_OK) {
        s = v->buffer_create(be, sizeof(copy_got),
                             GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_DEVICE,
                             &copy_dst);
        fails += check(s == GEIST_OK, "device head-copy dst create OK");
    }
    if (s == GEIST_OK) {
        fails += check(v->buffer_map(copy_src) == nullptr,
                       "head-copy src is not host-mappable");
        s = v->buffer_upload(copy_src, sizeof(copy_src_data), copy_src_data);
        fails += check(s == GEIST_OK, "upload head-copy src OK");
    }
    if (s == GEIST_OK) {
        s = transformer_head_copy_rows(v, copy_dst, copy_src,
                                       2, 1, copy_row_bytes);
        fails += check(s == GEIST_OK,
                       "head row copy uses device buffer_copy");
    }
    if (s == GEIST_OK) {
        s = v->buffer_download(copy_row_bytes, copy_got, copy_dst);
        fails += check(s == GEIST_OK, "download single copied head row OK");
    }
    if (s == GEIST_OK) {
        fails += compare_bytes("single copied head row",
                               copy_row_bytes, copy_got,
                               copy_src_data + 2u * copy_row_bytes);
    }
    if (s == GEIST_OK) {
        s = transformer_head_copy_rows(v, copy_dst, copy_src,
                                       1, copy_out_rows, copy_row_bytes);
        fails += check(s == GEIST_OK,
                       "head multi-row copy uses device buffer_copy");
    }
    if (s == GEIST_OK) {
        memset(copy_got, 0, sizeof(copy_got));
        s = v->buffer_download(sizeof(copy_got), copy_got, copy_dst);
        fails += check(s == GEIST_OK, "download copied head rows OK");
    }
    if (s == GEIST_OK) {
        fails += compare_bytes("copied head rows",
                               sizeof(copy_got), copy_got,
                               copy_src_data + copy_row_bytes);
    }
    if (s == GEIST_OK) {
        s = transformer_head_copy_rows(v, copy_dst, copy_src,
                                       0, 0, copy_row_bytes);
        fails += check(s == GEIST_E_INVALID_ARG,
                       "head row copy rejects zero row count");
        s = GEIST_OK;
    }
    if (copy_src != nullptr) { v->buffer_destroy(be, copy_src); }
    if (copy_dst != nullptr) { v->buffer_destroy(be, copy_dst); }

    constexpr size_t n_in = 5;
    constexpr size_t n0 = 4;
    constexpr size_t n1 = 3;
    constexpr size_t n2 = 2;
    const float x_data[n_in] = {0.25f, -1.0f, 0.5f, 2.0f, -0.75f};
    const float w0_data[n0 * n_in] = {
        1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
        0.5f, -0.25f, 0.75f, 1.0f, -0.5f,
    };
    const float w1_data[n1 * n_in] = {
        -1.0f, 0.25f, 0.0f, 0.5f, 1.0f,
        0.0f, -0.5f, 0.5f, 0.0f, 0.25f,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    };
    const float w2_data[n2 * n_in] = {
        0.125f, 0.25f, 0.5f, 1.0f, 2.0f,
        -0.75f, 0.0f, 0.25f, 0.0f, 0.5f,
    };
    float y0_ref[n0], y1_ref[n1], y2_ref[n2];
    ref_matvec(n_in, n0, x_data, w0_data, y0_ref);
    ref_matvec(n_in, n1, x_data, w1_data, y1_ref);
    ref_matvec(n_in, n2, x_data, w2_data, y2_ref);

    struct geist_buffer *x = nullptr;
    struct geist_buffer *w0 = nullptr;
    struct geist_buffer *w1 = nullptr;
    struct geist_buffer *w2 = nullptr;
    struct geist_buffer *y0 = nullptr;
    struct geist_buffer *y1 = nullptr;
    struct geist_buffer *y2 = nullptr;

    s = v->buffer_create(be, n_in * sizeof(float), GEIST_BUFFER_ACTIVATION,
                         GEIST_MEMORY_DEVICE, &x);
    fails += check(s == GEIST_OK, "device x buffer_create OK");
    if (s == GEIST_OK) {
        s = v->buffer_create(be, n0 * n_in * sizeof(float), GEIST_BUFFER_WEIGHT,
                             GEIST_MEMORY_DEVICE, &w0);
        fails += check(s == GEIST_OK, "device w0 buffer_create OK");
    }
    if (s == GEIST_OK) {
        s = v->buffer_create(be, n1 * n_in * sizeof(float), GEIST_BUFFER_WEIGHT,
                             GEIST_MEMORY_DEVICE, &w1);
        fails += check(s == GEIST_OK, "device w1 buffer_create OK");
    }
    if (s == GEIST_OK) {
        s = v->buffer_create(be, n2 * n_in * sizeof(float), GEIST_BUFFER_WEIGHT,
                             GEIST_MEMORY_DEVICE, &w2);
        fails += check(s == GEIST_OK, "device w2 buffer_create OK");
    }
    if (s == GEIST_OK) {
        s = v->buffer_create(be, n0 * sizeof(float), GEIST_BUFFER_ACTIVATION,
                             GEIST_MEMORY_DEVICE, &y0);
        fails += check(s == GEIST_OK, "device y0 buffer_create OK");
    }
    if (s == GEIST_OK) {
        s = v->buffer_create(be, n1 * sizeof(float), GEIST_BUFFER_ACTIVATION,
                             GEIST_MEMORY_DEVICE, &y1);
        fails += check(s == GEIST_OK, "device y1 buffer_create OK");
    }
    if (s == GEIST_OK) {
        s = v->buffer_create(be, n2 * sizeof(float), GEIST_BUFFER_ACTIVATION,
                             GEIST_MEMORY_DEVICE, &y2);
        fails += check(s == GEIST_OK, "device y2 buffer_create OK");
    }
    if (s == GEIST_OK) {
        fails += check(v->buffer_map(x) == nullptr,
                       "device activation buffer is not host-mappable");
        s = v->buffer_upload(x, sizeof(x_data), (const uint8_t *) x_data);
        fails += check(s == GEIST_OK, "upload x OK");
    }
    if (s == GEIST_OK) {
        s = v->buffer_upload(w0, sizeof(w0_data), (const uint8_t *) w0_data);
        fails += check(s == GEIST_OK, "upload w0 OK");
    }
    if (s == GEIST_OK) {
        s = v->buffer_upload(w1, sizeof(w1_data), (const uint8_t *) w1_data);
        fails += check(s == GEIST_OK, "upload w1 OK");
    }
    if (s == GEIST_OK) {
        s = v->buffer_upload(w2, sizeof(w2_data), (const uint8_t *) w2_data);
        fails += check(s == GEIST_OK, "upload w2 OK");
    }

    struct geist_tensor tx = tensor_2d(x, 1, n_in);
    struct geist_tensor tw0 = tensor_2d(w0, n0, n_in);
    struct geist_tensor tw1 = tensor_2d(w1, n1, n_in);
    struct geist_tensor tw2 = tensor_2d(w2, n2, n_in);
    struct geist_tensor ty0 = tensor_2d(y0, 1, n0);
    struct geist_tensor ty1 = tensor_2d(y1, 1, n1);
    struct geist_tensor ty2 = tensor_2d(y2, 1, n2);
    const struct geist_weight unresolved = {
        .n_in = (int32_t) n_in,
        .dtype = (uint16_t) GEIST_DTYPE_F32,
    };

    if (s == GEIST_OK) {
        s = linear_w_or_legacy(be, v, x, y0, &unresolved, 1,
                               &tx, &tw0, &ty0);
        fails += check(s == GEIST_OK,
                       "single linear helper uses Vulkan device matvec");
    }

    if (s == GEIST_OK) {
        s = linear_w_pair_or_legacy(be, v, x, y0, y1,
                                    &unresolved, &unresolved, 1,
                                    &tx, &tw0, &tw1, &ty0, &ty1);
        fails += check(s == GEIST_OK,
                       "pair linear helper uses Vulkan device matvec");
    }

    if (s == GEIST_OK) {
        s = linear_w_triple_or_legacy(be, v, x, y0, y1, y2,
                                      &unresolved, &unresolved, &unresolved,
                                      1, &tx, &tw0, &tw1, &tw2,
                                      &ty0, &ty1, &ty2);
        fails += check(s == GEIST_OK,
                       "triple linear helper uses Vulkan device matvec");
    }

    float y0_got[n0] = {0};
    float y1_got[n1] = {0};
    float y2_got[n2] = {0};
    if (s == GEIST_OK) {
        s = v->buffer_download(sizeof(y0_got), (uint8_t *) y0_got, y0);
        fails += check(s == GEIST_OK, "download y0 OK");
    }
    if (s == GEIST_OK) {
        s = v->buffer_download(sizeof(y1_got), (uint8_t *) y1_got, y1);
        fails += check(s == GEIST_OK, "download y1 OK");
    }
    if (s == GEIST_OK) {
        s = v->buffer_download(sizeof(y2_got), (uint8_t *) y2_got, y2);
        fails += check(s == GEIST_OK, "download y2 OK");
    }
    if (s == GEIST_OK) {
        fails += compare_f32("y0", n0, y0_got, y0_ref);
        fails += compare_f32("y1", n1, y1_got, y1_ref);
        fails += compare_f32("y2", n2, y2_got, y2_ref);
    }

    {
        const float x_seq_data[2 * n_in] = {
            1.0f, -2.0f, 0.5f, 3.0f,
            -0.25f, 0.75f, 2.0f, -1.0f,
        };
        float y_seq_ref[2 * n0];
        float y_seq_got[2 * n0];
        ref_matvec(n_in, n0, x_seq_data, w0_data, y_seq_ref);
        ref_matvec(n_in, n0, x_seq_data + n_in, w0_data,
                   y_seq_ref + n0);
        memset(y_seq_got, 0, sizeof(y_seq_got));

        struct geist_buffer *x_seq = nullptr;
        struct geist_buffer *y_seq = nullptr;
        if (s == GEIST_OK) {
            s = v->buffer_create(be, sizeof(x_seq_data),
                                 GEIST_BUFFER_ACTIVATION,
                                 GEIST_MEMORY_DEVICE, &x_seq);
            fails += check(s == GEIST_OK,
                           "device seq F32 x buffer_create OK");
        }
        if (s == GEIST_OK) {
            s = v->buffer_create(be, sizeof(y_seq_got),
                                 GEIST_BUFFER_ACTIVATION,
                                 GEIST_MEMORY_DEVICE, &y_seq);
            fails += check(s == GEIST_OK,
                           "device seq F32 y buffer_create OK");
        }
        if (s == GEIST_OK) {
            s = v->buffer_upload(x_seq, sizeof(x_seq_data),
                                 (const uint8_t *) x_seq_data);
            fails += check(s == GEIST_OK, "upload seq F32 x OK");
        }
        if (s == GEIST_OK) {
            struct geist_tensor tx_seq = tensor_2d(x_seq, 2, n_in);
            struct geist_tensor ty_seq = tensor_2d(y_seq, 2, n0);
            s = linear_w_or_legacy(be, v, x_seq, y_seq, &unresolved, 2,
                                   &tx_seq, &tw0, &ty_seq);
            fails += check(s == GEIST_OK,
                           "seq linear helper uses Vulkan device matmul");
        }
        if (s == GEIST_OK) {
            s = v->buffer_download(sizeof(y_seq_got),
                                   (uint8_t *) y_seq_got, y_seq);
            fails += check(s == GEIST_OK, "download seq F32 y OK");
        }
        if (s == GEIST_OK) {
            fails += compare_f32("seq F32 y", 2 * n0,
                                 y_seq_got, y_seq_ref);
        }
        if (x_seq != nullptr) { v->buffer_destroy(be, x_seq); }
        if (y_seq != nullptr) { v->buffer_destroy(be, y_seq); }
    }

    constexpr size_t q6_in = 256;
    constexpr size_t q6_out = 3;
    float q6_x[q6_in];
    uint8_t q6_w_data[q6_out * 210u];
    float q6_ref[q6_out], q6_got[q6_out];
    for (size_t i = 0; i < q6_in; i++) {
        q6_x[i] = (float) ((int) (i % 17u) - 8) * 0.03125f;
    }
    pack_q6k(q6_out, q6_w_data);
    ref_q6k(q6_out, q6_x, q6_ref);
    memset(q6_got, 0, sizeof(q6_got));

    struct geist_buffer *q6_xb = nullptr;
    struct geist_buffer *q6_wb = nullptr;
    struct geist_buffer *q6_yb = nullptr;
    if (s == GEIST_OK) {
        s = v->buffer_create(be, q6_in * sizeof(float),
                             GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_DEVICE,
                             &q6_xb);
        fails += check(s == GEIST_OK, "device Q6_K x buffer_create OK");
    }
    if (s == GEIST_OK) {
        s = v->buffer_create(be, sizeof(q6_w_data),
                             GEIST_BUFFER_WEIGHT, GEIST_MEMORY_DEVICE,
                             &q6_wb);
        fails += check(s == GEIST_OK, "device Q6_K w buffer_create OK");
    }
    if (s == GEIST_OK) {
        s = v->buffer_create(be, q6_out * sizeof(float),
                             GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_DEVICE,
                             &q6_yb);
        fails += check(s == GEIST_OK, "device Q6_K y buffer_create OK");
    }
    if (s == GEIST_OK) {
        s = v->buffer_upload(q6_xb, sizeof(q6_x), (const uint8_t *) q6_x);
        fails += check(s == GEIST_OK, "upload Q6_K x OK");
    }
    if (s == GEIST_OK) {
        s = v->buffer_upload(q6_wb, sizeof(q6_w_data), q6_w_data);
        fails += check(s == GEIST_OK, "upload Q6_K weight OK");
    }
    if (s == GEIST_OK) {
        struct geist_tensor q6_tx = tensor_2d(q6_xb, 1, q6_in);
        struct geist_tensor q6_tw = tensor_q6k_2d(q6_wb, q6_out, q6_in);
        struct geist_tensor q6_ty = tensor_2d(q6_yb, 1, q6_out);
        const struct geist_weight unresolved_q6 = {
            .n_in = (int32_t) q6_in,
            .dtype = (uint16_t) GEIST_DTYPE_Q6_K,
        };
        s = linear_w_or_legacy(be, v, q6_xb, q6_yb, &unresolved_q6,
                               1, &q6_tx, &q6_tw, &q6_ty);
        fails += check(s == GEIST_OK,
                       "single Q6_K linear helper uses Vulkan device matvec");
    }
    if (s == GEIST_OK) {
        s = v->buffer_download(sizeof(q6_got), (uint8_t *) q6_got, q6_yb);
        fails += check(s == GEIST_OK, "download Q6_K y OK");
    }
    if (s == GEIST_OK) {
        fails += compare_f32("q6_y", q6_out, q6_got, q6_ref);
    }
    if (q6_xb != nullptr) { v->buffer_destroy(be, q6_xb); }
    if (q6_wb != nullptr) { v->buffer_destroy(be, q6_wb); }
    if (q6_yb != nullptr) { v->buffer_destroy(be, q6_yb); }

    constexpr size_t head_d = 4;
    constexpr size_t head_vocab = 3;
    const float head_rows[2 * head_d] = {
        0.0f, 0.0f, 0.0f, 0.0f,
        1.0f, 2.0f, 3.0f, 4.0f,
    };
    const float head_norm[head_d] = {1.0f, 1.0f, 1.0f, 1.0f};
    const float head_w[head_vocab * head_d] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 2.0f,
        -1.0f, -1.0f, -1.0f, -1.0f,
    };
    const uint16_t head_w_f16[head_vocab * head_d] = {
        0x3c00u, 0x0000u, 0x0000u, 0x0000u,
        0x0000u, 0x0000u, 0x0000u, 0x4000u,
        0xbc00u, 0xbc00u, 0xbc00u, 0xbc00u,
    };
    const uint16_t head_w_bf16[head_vocab * head_d] = {
        0x3f80u, 0x0000u, 0x0000u, 0x0000u,
        0x0000u, 0x0000u, 0x0000u, 0x4000u,
        0xbf80u, 0xbf80u, 0xbf80u, 0xbf80u,
    };
    struct geist_buffer *head_h_a = nullptr;
    struct geist_buffer *head_h_b = nullptr;
    struct geist_buffer *head_norm_b = nullptr;
    struct geist_buffer *head_w_b = nullptr;
    struct geist_buffer *head_w_f16_b = nullptr;
    struct geist_buffer *head_w_bf16_b = nullptr;
    struct geist_buffer *head_logits = nullptr;
    if (s == GEIST_OK) {
        s = v->buffer_create(be, head_d * sizeof(float),
                             GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_DEVICE,
                             &head_h_a);
        fails += check(s == GEIST_OK, "device head scratch_h_a create OK");
    }
    if (s == GEIST_OK) {
        s = v->buffer_create(be, sizeof(head_rows),
                             GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_DEVICE,
                             &head_h_b);
        fails += check(s == GEIST_OK, "device head scratch_h_b create OK");
    }
    if (s == GEIST_OK) {
        s = v->buffer_create(be, sizeof(head_norm),
                             GEIST_BUFFER_WEIGHT, GEIST_MEMORY_DEVICE,
                             &head_norm_b);
        fails += check(s == GEIST_OK, "device head norm create OK");
    }
    if (s == GEIST_OK) {
        s = v->buffer_create(be, sizeof(head_w),
                             GEIST_BUFFER_WEIGHT, GEIST_MEMORY_DEVICE,
                             &head_w_b);
        fails += check(s == GEIST_OK, "device head lm weight create OK");
    }
    if (s == GEIST_OK) {
        s = v->buffer_create(be, sizeof(head_w_f16),
                             GEIST_BUFFER_WEIGHT, GEIST_MEMORY_DEVICE,
                             &head_w_f16_b);
        fails += check(s == GEIST_OK, "device head F16 lm weight create OK");
    }
    if (s == GEIST_OK) {
        s = v->buffer_create(be, sizeof(head_w_bf16),
                             GEIST_BUFFER_WEIGHT, GEIST_MEMORY_DEVICE,
                             &head_w_bf16_b);
        fails += check(s == GEIST_OK, "device head BF16 lm weight create OK");
    }
    if (s == GEIST_OK) {
        s = v->buffer_create(be, head_vocab * sizeof(float),
                             GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_DEVICE,
                             &head_logits);
        fails += check(s == GEIST_OK, "device head logits create OK");
    }
    if (s == GEIST_OK) {
        s = v->buffer_upload(head_h_b, sizeof(head_rows),
                             (const uint8_t *) head_rows);
        fails += check(s == GEIST_OK, "upload head residual rows OK");
    }
    if (s == GEIST_OK) {
        s = v->buffer_upload(head_norm_b, sizeof(head_norm),
                             (const uint8_t *) head_norm);
        fails += check(s == GEIST_OK, "upload head norm OK");
    }
    if (s == GEIST_OK) {
        s = v->buffer_upload(head_w_b, sizeof(head_w),
                             (const uint8_t *) head_w);
        fails += check(s == GEIST_OK, "upload head lm weight OK");
    }
    if (s == GEIST_OK) {
        s = v->buffer_upload(head_w_f16_b, sizeof(head_w_f16),
                             (const uint8_t *) head_w_f16);
        fails += check(s == GEIST_OK, "upload head F16 lm weight OK");
    }
    if (s == GEIST_OK) {
        s = v->buffer_upload(head_w_bf16_b, sizeof(head_w_bf16),
                             (const uint8_t *) head_w_bf16);
        fails += check(s == GEIST_OK, "upload head BF16 lm weight OK");
    }
    if (s == GEIST_OK) {
        fails += check(v->greedy_head != nullptr,
                       "vulkan exposes direct greedy_head op");
        struct geist_tensor direct_hidden = {
            .buffer = head_h_b,
            .offset = head_d * sizeof(float),
            .dtype = GEIST_DTYPE_F32,
            .layout = GEIST_LAYOUT_DENSE,
            .ndim = 1,
            .shape = {(int64_t) head_d},
            .stride = {1},
        };
        struct geist_tensor direct_norm = {
            .buffer = head_norm_b,
            .offset = 0,
            .dtype = GEIST_DTYPE_F32,
            .layout = GEIST_LAYOUT_DENSE,
            .ndim = 1,
            .shape = {(int64_t) head_d},
            .stride = {1},
        };
        struct geist_tensor direct_weight = {
            .buffer = head_w_b,
            .offset = 0,
            .dtype = GEIST_DTYPE_F32,
            .layout = GEIST_LAYOUT_DENSE,
            .ndim = 2,
            .shape = {(int64_t) head_vocab, (int64_t) head_d},
            .stride = {(int64_t) head_d, 1},
        };
        struct geist_tensor direct_normed = {
            .buffer = head_h_a,
            .offset = 0,
            .dtype = GEIST_DTYPE_F32,
            .layout = GEIST_LAYOUT_DENSE,
            .ndim = 1,
            .shape = {(int64_t) head_d},
            .stride = {1},
        };
        struct geist_tensor direct_logits = {
            .buffer = head_logits,
            .offset = 0,
            .dtype = GEIST_DTYPE_F32,
            .layout = GEIST_LAYOUT_DENSE,
            .ndim = 1,
            .shape = {(int64_t) head_vocab},
            .stride = {1},
        };
        const struct geist_backend_greedy_head direct_head = {
            .struct_size = sizeof(direct_head),
            .d_model = head_d,
            .vocab_size = head_vocab,
            .eps = 1.0e-5f,
            .hidden = &direct_hidden,
            .norm_weight = &direct_norm,
            .lm_head_weight = &direct_weight,
            .normed_scratch = &direct_normed,
            .logits = &direct_logits,
        };
        geist_token_t direct_token = -1;
        s = v->greedy_head(be, &direct_head, &direct_token);
        fails += check(s == GEIST_OK, "direct greedy_head succeeds");
        fails += check(direct_token == 1, "direct greedy_head returns argmax");
        {
            struct geist_tensor direct_weight_f16 = direct_weight;
            direct_weight_f16.buffer = head_w_f16_b;
            direct_weight_f16.dtype = GEIST_DTYPE_F16;
            const struct geist_backend_greedy_head f16_head = {
                .struct_size = sizeof(f16_head),
                .d_model = head_d,
                .vocab_size = head_vocab,
                .eps = 1.0e-5f,
                .hidden = &direct_hidden,
                .norm_weight = &direct_norm,
                .lm_head_weight = &direct_weight_f16,
                .normed_scratch = &direct_normed,
                .logits = &direct_logits,
            };
            geist_token_t f16_token = -1;
            enum geist_status hs = v->greedy_head(be, &f16_head, &f16_token);
            fails += check(hs == GEIST_OK,
                           "direct F16 greedy_head succeeds");
            fails += check(f16_token == 1,
                           "direct F16 greedy_head returns argmax");

            struct geist_tensor direct_weight_bf16 = direct_weight;
            direct_weight_bf16.buffer = head_w_bf16_b;
            direct_weight_bf16.dtype = GEIST_DTYPE_BF16;
            const struct geist_backend_greedy_head bf16_head = {
                .struct_size = sizeof(bf16_head),
                .d_model = head_d,
                .vocab_size = head_vocab,
                .eps = 1.0e-5f,
                .hidden = &direct_hidden,
                .norm_weight = &direct_norm,
                .lm_head_weight = &direct_weight_bf16,
                .normed_scratch = &direct_normed,
                .logits = &direct_logits,
            };
            geist_token_t bf16_token = -1;
            hs = v->greedy_head(be, &bf16_head, &bf16_token);
            fails += check(hs == GEIST_OK,
                           "direct BF16 greedy_head succeeds");
            fails += check(bf16_token == 1,
                           "direct BF16 greedy_head returns argmax");
        }
        if (s == GEIST_OK) {
            int argmax_capture_token = 0;
            geist_token_t captured_argmax_token = -2;
            enum geist_status argmax_cs = v->command_sequence_begin(
                be, GEIST_COMMAND_SEQUENCE_DECODE_GREEDY_STEP,
                &argmax_capture_token);
            fails += check(argmax_cs == GEIST_OK,
                           "captured argmax begin OK");
            if (argmax_cs == GEIST_OK) {
                argmax_cs = v->argmax_f32(be, &direct_logits,
                                          &captured_argmax_token);
                fails += check(argmax_cs == GEIST_E_UNSUPPORTED,
                               "captured standalone argmax is unsupported");
                fails += check(captured_argmax_token == 0,
                               "captured standalone argmax keeps default token");
            }
            if (argmax_capture_token != 0) {
                enum geist_status end_s =
                    v->command_sequence_end(be, argmax_capture_token, false);
                fails += check(end_s == GEIST_OK,
                               "captured standalone argmax discard OK");
            }

            int capture_token = 0;
            geist_token_t captured_direct_token = -2;
            geist_token_t captured_read_token = -1;
            enum geist_status cs = v->command_sequence_begin(
                be, GEIST_COMMAND_SEQUENCE_DECODE_GREEDY_STEP,
                &capture_token);
            fails += check(cs == GEIST_OK,
                           "captured greedy_head begin OK");
            if (cs == GEIST_OK) {
                cs = v->greedy_head(be, &direct_head,
                                    &captured_direct_token);
                fails += check(cs == GEIST_OK,
                               "captured greedy_head records OK");
                fails += check(captured_direct_token == -1,
                               "captured greedy_head defers token");
            }
            if (capture_token != 0) {
                const bool submit = cs == GEIST_OK;
                enum geist_status end_s =
                    v->command_sequence_end(be, capture_token, submit);
                fails += check(end_s == GEIST_OK,
                               "captured greedy_head end OK");
                if (cs == GEIST_OK) { cs = end_s; }
            }
            if (cs == GEIST_OK) {
                cs = v->command_sequence_read_token(be,
                                                    &captured_read_token);
                fails += check(cs == GEIST_OK,
                               "captured greedy_head token read OK");
                fails += check(captured_read_token == 1,
                               "captured greedy_head token matches argmax");
            }
        }
    }
    if (s == GEIST_OK) {
        struct transformer_arch_session head_sess = {
            .scratch_h_a = head_h_a,
            .scratch_h_b = head_h_b,
            .scratch_logits = head_logits,
            .temperature = 0.0f,
            .top_p = 1.0f,
            .top_k = 0,
            .logits_valid = false,
            .logits_on_device = false,
            .logits_host_valid = false,
            .next_token_pending = -1,
        };
        struct transformer_arch_state head_state = {
            .backend = be,
            .sess = &head_sess,
            .d_model = (int64_t) head_d,
            .vocab_size = (int64_t) head_vocab,
            .output_norm = {
                .buffer = head_norm_b,
                .offset = 0,
                .dtype = GEIST_DTYPE_F32,
                .layout = GEIST_LAYOUT_DENSE,
                .ndim = 1,
                .shape = {(int64_t) head_d},
                .stride = {1},
            },
            .embed_table = {
                .buffer = head_w_b,
                .offset = 0,
                .dtype = GEIST_DTYPE_F32,
                .layout = GEIST_LAYOUT_DENSE,
                .ndim = 2,
                .shape = {(int64_t) head_vocab, (int64_t) head_d},
                .stride = {(int64_t) head_d, 1},
            },
            .embed_table_w = {
                .n_in = (int32_t) head_d,
                .dtype = (uint16_t) GEIST_DTYPE_F32,
            },
        };
        geist_token_t head_token = -1;
        s = finalize_logits_one_row(&head_state, 1, &head_token);
        fails += check(s == GEIST_OK,
                       "device greedy finalize_logits_one_row succeeds");
        fails += check(head_token == 1,
                       "device greedy finalize_logits_one_row returns argmax");
        fails += check(head_sess.next_token_pending == 1,
                       "device greedy head stores pending token");
        fails += check(head_sess.logits_valid,
                       "device greedy head marks logits valid");
        fails += check(head_sess.logits_on_device,
                       "device greedy head marks logits device-resident");
        fails += check(!head_sess.logits_host_valid,
                       "device greedy head does not claim host logits");
        s = GEIST_OK;
    }
    if (head_h_a != nullptr) { v->buffer_destroy(be, head_h_a); }
    if (head_h_b != nullptr) { v->buffer_destroy(be, head_h_b); }
    if (head_norm_b != nullptr) { v->buffer_destroy(be, head_norm_b); }
    if (head_w_b != nullptr) { v->buffer_destroy(be, head_w_b); }
    if (head_w_f16_b != nullptr) { v->buffer_destroy(be, head_w_f16_b); }
    if (head_w_bf16_b != nullptr) { v->buffer_destroy(be, head_w_bf16_b); }
    if (head_logits != nullptr) { v->buffer_destroy(be, head_logits); }

    constexpr size_t ffn_d = 4;
    constexpr size_t ffn_inter = 4;
    const float ffn_residual[ffn_d] = {0.75f, -1.25f, 0.5f, 2.0f};
    const float ffn_norm[ffn_d] = {1.0f, 0.875f, 1.125f, 0.75f};
    const float ffn_gate_w[ffn_inter * ffn_d] = {
        0.5f, -0.25f, 0.125f, 0.75f,
        -0.375f, 0.625f, 0.25f, -0.5f,
        0.875f, 0.0f, -0.625f, 0.375f,
        -0.125f, 0.25f, -0.375f, 0.5f,
    };
    const float ffn_up_w[ffn_inter * ffn_d] = {
        -0.25f, 0.5f, 0.75f, -0.125f,
        0.625f, -0.75f, 0.125f, 0.25f,
        -0.5f, 0.375f, 0.875f, -0.625f,
        0.25f, -0.375f, 0.5f, -0.625f,
    };
    const float ffn_down_w[ffn_d * ffn_inter] = {
        0.25f, -0.5f, 0.75f, -0.125f,
        -0.875f, 0.125f, 0.5f, 0.25f,
        0.625f, -0.25f, -0.375f, 0.75f,
        -0.125f, 0.875f, 0.25f, -0.5f,
    };
    const float ffn_post_norm[ffn_d] = {1.0f, 0.95f, 1.05f, 0.8f};
    float ffn_ref[ffn_d];
    float ffn_f16_ref[ffn_d];
    float ffn_bf16_ref[ffn_d];
    float ffn_got[ffn_d] = {0};
    float ffn_f16_got[ffn_d] = {0};
    float ffn_bf16_got[ffn_d] = {0};
    uint16_t ffn_gate_w_f16[ffn_inter * ffn_d];
    uint16_t ffn_up_w_f16[ffn_inter * ffn_d];
    uint16_t ffn_down_w_f16[ffn_d * ffn_inter];
    uint16_t ffn_gate_w_bf16[ffn_inter * ffn_d];
    uint16_t ffn_up_w_bf16[ffn_inter * ffn_d];
    uint16_t ffn_down_w_bf16[ffn_d * ffn_inter];
    float ffn_gate_w_f16_ref[ffn_inter * ffn_d];
    float ffn_up_w_f16_ref[ffn_inter * ffn_d];
    float ffn_down_w_f16_ref[ffn_d * ffn_inter];
    float ffn_gate_w_bf16_ref[ffn_inter * ffn_d];
    float ffn_up_w_bf16_ref[ffn_inter * ffn_d];
    float ffn_down_w_bf16_ref[ffn_d * ffn_inter];
    ref_ffn_geglu_block(ffn_d, ffn_inter, ffn_residual, ffn_norm,
                        ffn_gate_w, ffn_up_w, ffn_down_w, ffn_post_norm,
                        1e-5f, ffn_ref);
    encode_f16_array(ffn_inter * ffn_d, ffn_gate_w,
                     ffn_gate_w_f16, ffn_gate_w_f16_ref);
    encode_f16_array(ffn_inter * ffn_d, ffn_up_w,
                     ffn_up_w_f16, ffn_up_w_f16_ref);
    encode_f16_array(ffn_d * ffn_inter, ffn_down_w,
                     ffn_down_w_f16, ffn_down_w_f16_ref);
    encode_bf16_array(ffn_inter * ffn_d, ffn_gate_w,
                      ffn_gate_w_bf16, ffn_gate_w_bf16_ref);
    encode_bf16_array(ffn_inter * ffn_d, ffn_up_w,
                      ffn_up_w_bf16, ffn_up_w_bf16_ref);
    encode_bf16_array(ffn_d * ffn_inter, ffn_down_w,
                      ffn_down_w_bf16, ffn_down_w_bf16_ref);
    ref_ffn_geglu_block(ffn_d, ffn_inter, ffn_residual, ffn_norm,
                        ffn_gate_w_f16_ref, ffn_up_w_f16_ref,
                        ffn_down_w_f16_ref, ffn_post_norm, 1e-5f,
                        ffn_f16_ref);
    ref_ffn_geglu_block(ffn_d, ffn_inter, ffn_residual, ffn_norm,
                        ffn_gate_w_bf16_ref, ffn_up_w_bf16_ref,
                        ffn_down_w_bf16_ref, ffn_post_norm, 1e-5f,
                        ffn_bf16_ref);

    struct geist_buffer *ffn_residual_b = nullptr;
    struct geist_buffer *ffn_norm_b = nullptr;
    struct geist_buffer *ffn_gate_w_b = nullptr;
    struct geist_buffer *ffn_up_w_b = nullptr;
    struct geist_buffer *ffn_down_w_b = nullptr;
    struct geist_buffer *ffn_gate_w_f16_b = nullptr;
    struct geist_buffer *ffn_up_w_f16_b = nullptr;
    struct geist_buffer *ffn_down_w_f16_b = nullptr;
    struct geist_buffer *ffn_gate_w_bf16_b = nullptr;
    struct geist_buffer *ffn_up_w_bf16_b = nullptr;
    struct geist_buffer *ffn_down_w_bf16_b = nullptr;
    struct geist_buffer *ffn_post_norm_b = nullptr;
    struct geist_buffer *ffn_pre_b = nullptr;
    struct geist_buffer *ffn_gate_b = nullptr;
    struct geist_buffer *ffn_up_b = nullptr;
    struct geist_buffer *ffn_out_b = nullptr;
    struct geist_buffer *ffn_post_b = nullptr;
    struct geist_buffer *ffn_y_b = nullptr;
    struct geist_buffer *ffn_m2_residual_b = nullptr;
    struct geist_buffer *ffn_m2_pre_b = nullptr;
    struct geist_buffer *ffn_m2_gate_b = nullptr;
    struct geist_buffer *ffn_m2_up_b = nullptr;
    struct geist_buffer *ffn_m2_out_b = nullptr;
    struct geist_buffer *ffn_m2_post_b = nullptr;
    struct geist_buffer *ffn_m2_y_b = nullptr;

    if (s == GEIST_OK) {
        const struct geist_backend_ffn_geglu_block bad_block = {
            .struct_size = 0,
        };
        fails += check(v->ffn_geglu_block(be, &bad_block) == GEIST_E_INVALID_ARG,
                       "FFN GEGLU block rejects short parameter struct");
    }
    if (s == GEIST_OK) {
        s = v->buffer_create(be, sizeof(ffn_residual),
                             GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_DEVICE,
                             &ffn_residual_b);
        fails += check(s == GEIST_OK, "device FFN residual create OK");
    }
    if (s == GEIST_OK) {
        s = v->buffer_create(be, sizeof(ffn_norm),
                             GEIST_BUFFER_WEIGHT, GEIST_MEMORY_DEVICE,
                             &ffn_norm_b);
        fails += check(s == GEIST_OK, "device FFN norm create OK");
    }
    if (s == GEIST_OK) {
        s = v->buffer_create(be, sizeof(ffn_gate_w),
                             GEIST_BUFFER_WEIGHT, GEIST_MEMORY_DEVICE,
                             &ffn_gate_w_b);
        fails += check(s == GEIST_OK, "device FFN gate weight create OK");
    }
    if (s == GEIST_OK) {
        s = v->buffer_create(be, sizeof(ffn_up_w),
                             GEIST_BUFFER_WEIGHT, GEIST_MEMORY_DEVICE,
                             &ffn_up_w_b);
        fails += check(s == GEIST_OK, "device FFN up weight create OK");
    }
    if (s == GEIST_OK) {
        s = v->buffer_create(be, sizeof(ffn_down_w),
                             GEIST_BUFFER_WEIGHT, GEIST_MEMORY_DEVICE,
                             &ffn_down_w_b);
        fails += check(s == GEIST_OK, "device FFN down weight create OK");
    }
    if (s == GEIST_OK) {
        s = v->buffer_create(be, sizeof(ffn_post_norm),
                             GEIST_BUFFER_WEIGHT, GEIST_MEMORY_DEVICE,
                             &ffn_post_norm_b);
        fails += check(s == GEIST_OK, "device FFN post norm create OK");
    }
    if (s == GEIST_OK) {
        s = v->buffer_create(be, ffn_d * sizeof(float),
                             GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_DEVICE,
                             &ffn_pre_b);
        fails += check(s == GEIST_OK, "device FFN pre scratch create OK");
    }
    if (s == GEIST_OK) {
        s = v->buffer_create(be, ffn_inter * sizeof(float),
                             GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_DEVICE,
                             &ffn_gate_b);
        fails += check(s == GEIST_OK, "device FFN gate scratch create OK");
    }
    if (s == GEIST_OK) {
        s = v->buffer_create(be, ffn_inter * sizeof(float),
                             GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_DEVICE,
                             &ffn_up_b);
        fails += check(s == GEIST_OK, "device FFN up scratch create OK");
    }
    if (s == GEIST_OK) {
        s = v->buffer_create(be, ffn_d * sizeof(float),
                             GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_DEVICE,
                             &ffn_out_b);
        fails += check(s == GEIST_OK, "device FFN out scratch create OK");
    }
    if (s == GEIST_OK) {
        s = v->buffer_create(be, ffn_d * sizeof(float),
                             GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_DEVICE,
                             &ffn_post_b);
        fails += check(s == GEIST_OK, "device FFN post scratch create OK");
    }
    if (s == GEIST_OK) {
        s = v->buffer_create(be, ffn_d * sizeof(float),
                             GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_DEVICE,
                             &ffn_y_b);
        fails += check(s == GEIST_OK, "device FFN y create OK");
    }
    if (s == GEIST_OK) {
        s = v->buffer_upload(ffn_residual_b, sizeof(ffn_residual),
                             (const uint8_t *) ffn_residual);
        fails += check(s == GEIST_OK, "upload FFN residual OK");
    }
    if (s == GEIST_OK) {
        s = v->buffer_upload(ffn_norm_b, sizeof(ffn_norm),
                             (const uint8_t *) ffn_norm);
        fails += check(s == GEIST_OK, "upload FFN norm OK");
    }
    if (s == GEIST_OK) {
        s = v->buffer_upload(ffn_gate_w_b, sizeof(ffn_gate_w),
                             (const uint8_t *) ffn_gate_w);
        fails += check(s == GEIST_OK, "upload FFN gate weight OK");
    }
    if (s == GEIST_OK) {
        s = v->buffer_upload(ffn_up_w_b, sizeof(ffn_up_w),
                             (const uint8_t *) ffn_up_w);
        fails += check(s == GEIST_OK, "upload FFN up weight OK");
    }
    if (s == GEIST_OK) {
        s = v->buffer_upload(ffn_down_w_b, sizeof(ffn_down_w),
                             (const uint8_t *) ffn_down_w);
        fails += check(s == GEIST_OK, "upload FFN down weight OK");
    }
    if (s == GEIST_OK) {
        s = v->buffer_upload(ffn_post_norm_b, sizeof(ffn_post_norm),
                             (const uint8_t *) ffn_post_norm);
        fails += check(s == GEIST_OK, "upload FFN post norm OK");
    }
    if (s == GEIST_OK) {
        struct geist_tensor t_ffn_residual = tensor_2d(ffn_residual_b, 1, ffn_d);
        struct geist_tensor t_ffn_norm = tensor_1d(ffn_norm_b, ffn_d);
        struct geist_tensor t_ffn_gate_w =
            tensor_2d(ffn_gate_w_b, ffn_inter, ffn_d);
        struct geist_tensor t_ffn_up_w =
            tensor_2d(ffn_up_w_b, ffn_inter, ffn_d);
        struct geist_tensor t_ffn_down_w =
            tensor_2d(ffn_down_w_b, ffn_d, ffn_inter);
        struct geist_tensor t_ffn_post_norm =
            tensor_1d(ffn_post_norm_b, ffn_d);
        struct geist_tensor t_ffn_pre = tensor_2d(ffn_pre_b, 1, ffn_d);
        struct geist_tensor t_ffn_gate = tensor_2d(ffn_gate_b, 1, ffn_inter);
        struct geist_tensor t_ffn_up = tensor_2d(ffn_up_b, 1, ffn_inter);
        struct geist_tensor t_ffn_out = tensor_2d(ffn_out_b, 1, ffn_d);
        struct geist_tensor t_ffn_post = tensor_2d(ffn_post_b, 1, ffn_d);
        struct geist_tensor t_ffn_y = tensor_2d(ffn_y_b, 1, ffn_d);
        const struct geist_backend_ffn_geglu_block block = {
            .struct_size = sizeof(block),
            .seq = 1,
            .d_model = ffn_d,
            .inter = ffn_inter,
            .eps = 1e-5f,
            .residual = &t_ffn_residual,
            .ffn_norm_weight = &t_ffn_norm,
            .gate_weight = &t_ffn_gate_w,
            .up_weight = &t_ffn_up_w,
            .down_weight = &t_ffn_down_w,
            .post_ffw_norm_weight = &t_ffn_post_norm,
            .pre_ff_scratch = &t_ffn_pre,
            .gate_scratch = &t_ffn_gate,
            .up_scratch = &t_ffn_up,
            .ffn_out_scratch = &t_ffn_out,
            .post_ff_scratch = &t_ffn_post,
            .out = &t_ffn_y,
        };
        s = v->ffn_geglu_block(be, &block);
        fails += check(s == GEIST_OK,
                       "device FFN GEGLU block succeeds");
    }
    if (s == GEIST_OK) {
        s = v->buffer_download(sizeof(ffn_got), (uint8_t *) ffn_got,
                               ffn_y_b);
        fails += check(s == GEIST_OK, "download FFN GEGLU output OK");
    }
    if (s == GEIST_OK) {
        fails += compare_f32("ffn_geglu_block", ffn_d, ffn_got, ffn_ref);
    }
    constexpr size_t ffn_m2 = 33;
    float ffn_m2_residual[ffn_m2 * ffn_d];
    float ffn_m2_ref[ffn_m2 * ffn_d];
    float ffn_m2_got[ffn_m2 * ffn_d] = {0};
    for (size_t row = 0; row < ffn_m2; row++) {
        for (size_t col = 0; col < ffn_d; col++) {
            const int v = (int) ((row + 3u * col) % 11u) - 5;
            ffn_m2_residual[row * ffn_d + col] =
                (float) v * 0.125f + (float) row * 0.0078125f;
        }
        ref_ffn_geglu_block(ffn_d, ffn_inter,
                            ffn_m2_residual + row * ffn_d, ffn_norm,
                            ffn_gate_w, ffn_up_w, ffn_down_w,
                            ffn_post_norm, 1e-5f,
                            ffn_m2_ref + row * ffn_d);
    }
#define CREATE_FFN_M2_BUF(var, elems, label) \
    do { \
        if (s == GEIST_OK) { \
            s = v->buffer_create(be, (elems) * sizeof(float), \
                                 GEIST_BUFFER_ACTIVATION, \
                                 GEIST_MEMORY_DEVICE, &(var)); \
            fails += check(s == GEIST_OK, label); \
        } \
    } while (0)
    CREATE_FFN_M2_BUF(ffn_m2_residual_b, ffn_m2 * ffn_d,
                      "device M2 FFN residual create OK");
    CREATE_FFN_M2_BUF(ffn_m2_pre_b, ffn_m2 * ffn_d,
                      "device M2 FFN pre scratch create OK");
    CREATE_FFN_M2_BUF(ffn_m2_gate_b, ffn_m2 * ffn_inter,
                      "device M2 FFN gate scratch create OK");
    CREATE_FFN_M2_BUF(ffn_m2_up_b, ffn_m2 * ffn_inter,
                      "device M2 FFN up scratch create OK");
    CREATE_FFN_M2_BUF(ffn_m2_out_b, ffn_m2 * ffn_d,
                      "device M2 FFN out scratch create OK");
    CREATE_FFN_M2_BUF(ffn_m2_post_b, ffn_m2 * ffn_d,
                      "device M2 FFN post scratch create OK");
    CREATE_FFN_M2_BUF(ffn_m2_y_b, ffn_m2 * ffn_d,
                      "device M2 FFN y create OK");
#undef CREATE_FFN_M2_BUF
    if (s == GEIST_OK) {
        s = v->buffer_upload(ffn_m2_residual_b, sizeof(ffn_m2_residual),
                             (const uint8_t *) ffn_m2_residual);
        fails += check(s == GEIST_OK, "upload M2 FFN residual OK");
    }
    if (s == GEIST_OK) {
        struct geist_tensor t_ffn_residual =
            tensor_2d(ffn_m2_residual_b, ffn_m2, ffn_d);
        struct geist_tensor t_ffn_norm = tensor_1d(ffn_norm_b, ffn_d);
        struct geist_tensor t_ffn_gate_w =
            tensor_2d(ffn_gate_w_b, ffn_inter, ffn_d);
        struct geist_tensor t_ffn_up_w =
            tensor_2d(ffn_up_w_b, ffn_inter, ffn_d);
        struct geist_tensor t_ffn_down_w =
            tensor_2d(ffn_down_w_b, ffn_d, ffn_inter);
        struct geist_tensor t_ffn_post_norm =
            tensor_1d(ffn_post_norm_b, ffn_d);
        struct geist_tensor t_ffn_pre =
            tensor_2d(ffn_m2_pre_b, ffn_m2, ffn_d);
        struct geist_tensor t_ffn_gate =
            tensor_2d(ffn_m2_gate_b, ffn_m2, ffn_inter);
        struct geist_tensor t_ffn_up =
            tensor_2d(ffn_m2_up_b, ffn_m2, ffn_inter);
        struct geist_tensor t_ffn_out =
            tensor_2d(ffn_m2_out_b, ffn_m2, ffn_d);
        struct geist_tensor t_ffn_post =
            tensor_2d(ffn_m2_post_b, ffn_m2, ffn_d);
        struct geist_tensor t_ffn_y =
            tensor_2d(ffn_m2_y_b, ffn_m2, ffn_d);
        const struct geist_backend_ffn_geglu_block block = {
            .struct_size = sizeof(block),
            .seq = ffn_m2,
            .d_model = ffn_d,
            .inter = ffn_inter,
            .eps = 1e-5f,
            .residual = &t_ffn_residual,
            .ffn_norm_weight = &t_ffn_norm,
            .gate_weight = &t_ffn_gate_w,
            .up_weight = &t_ffn_up_w,
            .down_weight = &t_ffn_down_w,
            .post_ffw_norm_weight = &t_ffn_post_norm,
            .pre_ff_scratch = &t_ffn_pre,
            .gate_scratch = &t_ffn_gate,
            .up_scratch = &t_ffn_up,
            .ffn_out_scratch = &t_ffn_out,
            .post_ff_scratch = &t_ffn_post,
            .out = &t_ffn_y,
        };
        s = v->ffn_geglu_block(be, &block);
        fails += check(s == GEIST_OK,
                       "device M2 FFN GEGLU block succeeds");
    }
    if (s == GEIST_OK) {
        s = v->buffer_download(sizeof(ffn_m2_got),
                               (uint8_t *) ffn_m2_got, ffn_m2_y_b);
        fails += check(s == GEIST_OK, "download M2 FFN GEGLU output OK");
    }
    if (s == GEIST_OK) {
        fails += compare_f32("m2_ffn_geglu_block", ffn_m2 * ffn_d,
                             ffn_m2_got, ffn_m2_ref);
    }
    if (ffn_m2_residual_b != nullptr) {
        v->buffer_destroy(be, ffn_m2_residual_b);
    }
    if (ffn_m2_pre_b != nullptr) { v->buffer_destroy(be, ffn_m2_pre_b); }
    if (ffn_m2_gate_b != nullptr) { v->buffer_destroy(be, ffn_m2_gate_b); }
    if (ffn_m2_up_b != nullptr) { v->buffer_destroy(be, ffn_m2_up_b); }
    if (ffn_m2_out_b != nullptr) { v->buffer_destroy(be, ffn_m2_out_b); }
    if (ffn_m2_post_b != nullptr) { v->buffer_destroy(be, ffn_m2_post_b); }
    if (ffn_m2_y_b != nullptr) { v->buffer_destroy(be, ffn_m2_y_b); }

    {
        constexpr size_t ffn_wide_m = 2;
        constexpr size_t ffn_wide_inter = 300;
        float residual[ffn_wide_m * ffn_d];
        float gate_w[ffn_wide_inter * ffn_d];
        float up_w[ffn_wide_inter * ffn_d];
        float down_w[ffn_d * ffn_wide_inter];
        float ref[ffn_wide_m * ffn_d];
        float got[ffn_wide_m * ffn_d] = {0};
        for (size_t i = 0; i < ffn_wide_m * ffn_d; i++) {
            residual[i] = (float) ((int) (i % 13u) - 6) * 0.0625f;
        }
        for (size_t i = 0; i < ffn_wide_inter * ffn_d; i++) {
            gate_w[i] = (float) ((int) (i % 17u) - 8) * 0.015625f;
            up_w[i] = (float) ((int) ((i * 3u) % 19u) - 9) * 0.0125f;
        }
        for (size_t i = 0; i < ffn_d * ffn_wide_inter; i++) {
            down_w[i] = (float) ((int) ((i * 5u) % 23u) - 11) * 0.01f;
        }
        for (size_t row = 0; row < ffn_wide_m; row++) {
            ref_ffn_geglu_block(ffn_d, ffn_wide_inter,
                                residual + row * ffn_d, ffn_norm,
                                gate_w, up_w, down_w, ffn_post_norm,
                                1e-5f, ref + row * ffn_d);
        }

        struct geist_buffer *residual_b = nullptr;
        struct geist_buffer *gate_w_b = nullptr;
        struct geist_buffer *up_w_b = nullptr;
        struct geist_buffer *down_w_b = nullptr;
        struct geist_buffer *pre_b = nullptr;
        struct geist_buffer *gate_b = nullptr;
        struct geist_buffer *up_b = nullptr;
        struct geist_buffer *out_b = nullptr;
        struct geist_buffer *post_b = nullptr;
        struct geist_buffer *y_b = nullptr;
#define CREATE_WIDE_FFN_BUF(var, bytes, kind, label) \
        do { \
            if (s == GEIST_OK) { \
                s = v->buffer_create(be, (bytes), (kind), \
                                     GEIST_MEMORY_DEVICE, &(var)); \
                fails += check(s == GEIST_OK, label); \
            } \
        } while (0)
        CREATE_WIDE_FFN_BUF(residual_b, sizeof(residual),
                            GEIST_BUFFER_ACTIVATION,
                            "device wide FFN residual create OK");
        CREATE_WIDE_FFN_BUF(gate_w_b, sizeof(gate_w), GEIST_BUFFER_WEIGHT,
                            "device wide FFN gate weight create OK");
        CREATE_WIDE_FFN_BUF(up_w_b, sizeof(up_w), GEIST_BUFFER_WEIGHT,
                            "device wide FFN up weight create OK");
        CREATE_WIDE_FFN_BUF(down_w_b, sizeof(down_w), GEIST_BUFFER_WEIGHT,
                            "device wide FFN down weight create OK");
        CREATE_WIDE_FFN_BUF(pre_b, sizeof(residual),
                            GEIST_BUFFER_ACTIVATION,
                            "device wide FFN pre scratch create OK");
        CREATE_WIDE_FFN_BUF(gate_b, ffn_wide_m * ffn_wide_inter * sizeof(float),
                            GEIST_BUFFER_ACTIVATION,
                            "device wide FFN gate scratch create OK");
        CREATE_WIDE_FFN_BUF(up_b, ffn_wide_m * ffn_wide_inter * sizeof(float),
                            GEIST_BUFFER_ACTIVATION,
                            "device wide FFN up scratch create OK");
        CREATE_WIDE_FFN_BUF(out_b, sizeof(got), GEIST_BUFFER_ACTIVATION,
                            "device wide FFN out scratch create OK");
        CREATE_WIDE_FFN_BUF(post_b, sizeof(got), GEIST_BUFFER_ACTIVATION,
                            "device wide FFN post scratch create OK");
        CREATE_WIDE_FFN_BUF(y_b, sizeof(got), GEIST_BUFFER_ACTIVATION,
                            "device wide FFN y create OK");
#undef CREATE_WIDE_FFN_BUF
        if (s == GEIST_OK) {
            s = v->buffer_upload(residual_b, sizeof(residual),
                                 (const uint8_t *) residual);
            fails += check(s == GEIST_OK,
                           "upload wide FFN residual OK");
        }
        if (s == GEIST_OK) {
            s = v->buffer_upload(gate_w_b, sizeof(gate_w),
                                 (const uint8_t *) gate_w);
            fails += check(s == GEIST_OK,
                           "upload wide FFN gate weight OK");
        }
        if (s == GEIST_OK) {
            s = v->buffer_upload(up_w_b, sizeof(up_w),
                                 (const uint8_t *) up_w);
            fails += check(s == GEIST_OK,
                           "upload wide FFN up weight OK");
        }
        if (s == GEIST_OK) {
            s = v->buffer_upload(down_w_b, sizeof(down_w),
                                 (const uint8_t *) down_w);
            fails += check(s == GEIST_OK,
                           "upload wide FFN down weight OK");
        }
        if (s == GEIST_OK) {
            struct geist_tensor t_residual =
                tensor_2d(residual_b, ffn_wide_m, ffn_d);
            struct geist_tensor t_norm = tensor_1d(ffn_norm_b, ffn_d);
            struct geist_tensor t_gate_w =
                tensor_2d(gate_w_b, ffn_wide_inter, ffn_d);
            struct geist_tensor t_up_w =
                tensor_2d(up_w_b, ffn_wide_inter, ffn_d);
            struct geist_tensor t_down_w =
                tensor_2d(down_w_b, ffn_d, ffn_wide_inter);
            struct geist_tensor t_post_norm =
                tensor_1d(ffn_post_norm_b, ffn_d);
            struct geist_tensor t_pre =
                tensor_2d(pre_b, ffn_wide_m, ffn_d);
            struct geist_tensor t_gate =
                tensor_2d(gate_b, ffn_wide_m, ffn_wide_inter);
            struct geist_tensor t_up =
                tensor_2d(up_b, ffn_wide_m, ffn_wide_inter);
            struct geist_tensor t_out =
                tensor_2d(out_b, ffn_wide_m, ffn_d);
            struct geist_tensor t_post =
                tensor_2d(post_b, ffn_wide_m, ffn_d);
            struct geist_tensor t_y =
                tensor_2d(y_b, ffn_wide_m, ffn_d);
            const struct geist_backend_ffn_geglu_block block = {
                .struct_size = sizeof(block),
                .seq = ffn_wide_m,
                .d_model = ffn_d,
                .inter = ffn_wide_inter,
                .eps = 1e-5f,
                .residual = &t_residual,
                .ffn_norm_weight = &t_norm,
                .gate_weight = &t_gate_w,
                .up_weight = &t_up_w,
                .down_weight = &t_down_w,
                .post_ffw_norm_weight = &t_post_norm,
                .pre_ff_scratch = &t_pre,
                .gate_scratch = &t_gate,
                .up_scratch = &t_up,
                .ffn_out_scratch = &t_out,
                .post_ff_scratch = &t_post,
                .out = &t_y,
            };
            s = v->ffn_geglu_block(be, &block);
            fails += check(s == GEIST_OK,
                           "device wide FFN GEGLU block succeeds");
        }
        if (s == GEIST_OK) {
            s = v->buffer_download(sizeof(got), (uint8_t *) got, y_b);
            fails += check(s == GEIST_OK,
                           "download wide FFN GEGLU output OK");
        }
        if (s == GEIST_OK) {
            fails += compare_f32("wide_ffn_geglu_block",
                                 ffn_wide_m * ffn_d, got, ref);
        }
        if (residual_b != nullptr) { v->buffer_destroy(be, residual_b); }
        if (gate_w_b != nullptr) { v->buffer_destroy(be, gate_w_b); }
        if (up_w_b != nullptr) { v->buffer_destroy(be, up_w_b); }
        if (down_w_b != nullptr) { v->buffer_destroy(be, down_w_b); }
        if (pre_b != nullptr) { v->buffer_destroy(be, pre_b); }
        if (gate_b != nullptr) { v->buffer_destroy(be, gate_b); }
        if (up_b != nullptr) { v->buffer_destroy(be, up_b); }
        if (out_b != nullptr) { v->buffer_destroy(be, out_b); }
        if (post_b != nullptr) { v->buffer_destroy(be, post_b); }
        if (y_b != nullptr) { v->buffer_destroy(be, y_b); }
    }

    if (s == GEIST_OK) {
        s = v->buffer_create(be, sizeof(ffn_gate_w_f16),
                             GEIST_BUFFER_WEIGHT, GEIST_MEMORY_DEVICE,
                             &ffn_gate_w_f16_b);
        fails += check(s == GEIST_OK, "device F16 FFN gate weight create OK");
    }
    if (s == GEIST_OK) {
        s = v->buffer_create(be, sizeof(ffn_up_w_f16),
                             GEIST_BUFFER_WEIGHT, GEIST_MEMORY_DEVICE,
                             &ffn_up_w_f16_b);
        fails += check(s == GEIST_OK, "device F16 FFN up weight create OK");
    }
    if (s == GEIST_OK) {
        s = v->buffer_create(be, sizeof(ffn_down_w_f16),
                             GEIST_BUFFER_WEIGHT, GEIST_MEMORY_DEVICE,
                             &ffn_down_w_f16_b);
        fails += check(s == GEIST_OK, "device F16 FFN down weight create OK");
    }
    if (s == GEIST_OK) {
        s = v->buffer_create(be, sizeof(ffn_gate_w_bf16),
                             GEIST_BUFFER_WEIGHT, GEIST_MEMORY_DEVICE,
                             &ffn_gate_w_bf16_b);
        fails += check(s == GEIST_OK, "device BF16 FFN gate weight create OK");
    }
    if (s == GEIST_OK) {
        s = v->buffer_create(be, sizeof(ffn_up_w_bf16),
                             GEIST_BUFFER_WEIGHT, GEIST_MEMORY_DEVICE,
                             &ffn_up_w_bf16_b);
        fails += check(s == GEIST_OK, "device BF16 FFN up weight create OK");
    }
    if (s == GEIST_OK) {
        s = v->buffer_create(be, sizeof(ffn_down_w_bf16),
                             GEIST_BUFFER_WEIGHT, GEIST_MEMORY_DEVICE,
                             &ffn_down_w_bf16_b);
        fails += check(s == GEIST_OK, "device BF16 FFN down weight create OK");
    }
    if (s == GEIST_OK) {
        s = v->buffer_upload(ffn_gate_w_f16_b, sizeof(ffn_gate_w_f16),
                             (const uint8_t *) ffn_gate_w_f16);
        fails += check(s == GEIST_OK, "upload F16 FFN gate weight OK");
    }
    if (s == GEIST_OK) {
        s = v->buffer_upload(ffn_up_w_f16_b, sizeof(ffn_up_w_f16),
                             (const uint8_t *) ffn_up_w_f16);
        fails += check(s == GEIST_OK, "upload F16 FFN up weight OK");
    }
    if (s == GEIST_OK) {
        s = v->buffer_upload(ffn_down_w_f16_b, sizeof(ffn_down_w_f16),
                             (const uint8_t *) ffn_down_w_f16);
        fails += check(s == GEIST_OK, "upload F16 FFN down weight OK");
    }
    if (s == GEIST_OK) {
        s = v->buffer_upload(ffn_gate_w_bf16_b, sizeof(ffn_gate_w_bf16),
                             (const uint8_t *) ffn_gate_w_bf16);
        fails += check(s == GEIST_OK, "upload BF16 FFN gate weight OK");
    }
    if (s == GEIST_OK) {
        s = v->buffer_upload(ffn_up_w_bf16_b, sizeof(ffn_up_w_bf16),
                             (const uint8_t *) ffn_up_w_bf16);
        fails += check(s == GEIST_OK, "upload BF16 FFN up weight OK");
    }
    if (s == GEIST_OK) {
        s = v->buffer_upload(ffn_down_w_bf16_b, sizeof(ffn_down_w_bf16),
                             (const uint8_t *) ffn_down_w_bf16);
        fails += check(s == GEIST_OK, "upload BF16 FFN down weight OK");
    }
    if (s == GEIST_OK) {
        struct geist_tensor t_ffn_residual = tensor_2d(ffn_residual_b, 1, ffn_d);
        struct geist_tensor t_ffn_norm = tensor_1d(ffn_norm_b, ffn_d);
        struct geist_tensor t_ffn_gate_w =
            tensor_2d_dtype(ffn_gate_w_f16_b, GEIST_DTYPE_F16, ffn_inter, ffn_d);
        struct geist_tensor t_ffn_up_w =
            tensor_2d_dtype(ffn_up_w_f16_b, GEIST_DTYPE_F16, ffn_inter, ffn_d);
        struct geist_tensor t_ffn_down_w =
            tensor_2d_dtype(ffn_down_w_f16_b, GEIST_DTYPE_F16, ffn_d, ffn_inter);
        struct geist_tensor t_ffn_post_norm =
            tensor_1d(ffn_post_norm_b, ffn_d);
        struct geist_tensor t_ffn_pre = tensor_2d(ffn_pre_b, 1, ffn_d);
        struct geist_tensor t_ffn_gate = tensor_2d(ffn_gate_b, 1, ffn_inter);
        struct geist_tensor t_ffn_up = tensor_2d(ffn_up_b, 1, ffn_inter);
        struct geist_tensor t_ffn_out = tensor_2d(ffn_out_b, 1, ffn_d);
        struct geist_tensor t_ffn_post = tensor_2d(ffn_post_b, 1, ffn_d);
        struct geist_tensor t_ffn_y = tensor_2d(ffn_y_b, 1, ffn_d);
        const struct geist_backend_ffn_geglu_block block = {
            .struct_size = sizeof(block),
            .seq = 1,
            .d_model = ffn_d,
            .inter = ffn_inter,
            .eps = 1e-5f,
            .residual = &t_ffn_residual,
            .ffn_norm_weight = &t_ffn_norm,
            .gate_weight = &t_ffn_gate_w,
            .up_weight = &t_ffn_up_w,
            .down_weight = &t_ffn_down_w,
            .post_ffw_norm_weight = &t_ffn_post_norm,
            .pre_ff_scratch = &t_ffn_pre,
            .gate_scratch = &t_ffn_gate,
            .up_scratch = &t_ffn_up,
            .ffn_out_scratch = &t_ffn_out,
            .post_ff_scratch = &t_ffn_post,
            .out = &t_ffn_y,
        };
        s = v->ffn_geglu_block(be, &block);
        fails += check(s == GEIST_OK, "device F16 FFN GEGLU block succeeds");
    }
    if (s == GEIST_OK) {
        s = v->buffer_download(sizeof(ffn_f16_got),
                               (uint8_t *) ffn_f16_got, ffn_y_b);
        fails += check(s == GEIST_OK, "download F16 FFN GEGLU output OK");
    }
    if (s == GEIST_OK) {
        fails += compare_f32_tol("f16_ffn_geglu_block", ffn_d,
                                 ffn_f16_got, ffn_f16_ref, 2e-4f, 2e-4f);
    }
    if (s == GEIST_OK) {
        struct geist_tensor t_ffn_residual = tensor_2d(ffn_residual_b, 1, ffn_d);
        struct geist_tensor t_ffn_norm = tensor_1d(ffn_norm_b, ffn_d);
        struct geist_tensor t_ffn_gate_w =
            tensor_2d_dtype(ffn_gate_w_bf16_b, GEIST_DTYPE_BF16, ffn_inter, ffn_d);
        struct geist_tensor t_ffn_up_w =
            tensor_2d_dtype(ffn_up_w_bf16_b, GEIST_DTYPE_BF16, ffn_inter, ffn_d);
        struct geist_tensor t_ffn_down_w =
            tensor_2d_dtype(ffn_down_w_bf16_b, GEIST_DTYPE_BF16, ffn_d, ffn_inter);
        struct geist_tensor t_ffn_post_norm =
            tensor_1d(ffn_post_norm_b, ffn_d);
        struct geist_tensor t_ffn_pre = tensor_2d(ffn_pre_b, 1, ffn_d);
        struct geist_tensor t_ffn_gate = tensor_2d(ffn_gate_b, 1, ffn_inter);
        struct geist_tensor t_ffn_up = tensor_2d(ffn_up_b, 1, ffn_inter);
        struct geist_tensor t_ffn_out = tensor_2d(ffn_out_b, 1, ffn_d);
        struct geist_tensor t_ffn_post = tensor_2d(ffn_post_b, 1, ffn_d);
        struct geist_tensor t_ffn_y = tensor_2d(ffn_y_b, 1, ffn_d);
        const struct geist_backend_ffn_geglu_block block = {
            .struct_size = sizeof(block),
            .seq = 1,
            .d_model = ffn_d,
            .inter = ffn_inter,
            .eps = 1e-5f,
            .residual = &t_ffn_residual,
            .ffn_norm_weight = &t_ffn_norm,
            .gate_weight = &t_ffn_gate_w,
            .up_weight = &t_ffn_up_w,
            .down_weight = &t_ffn_down_w,
            .post_ffw_norm_weight = &t_ffn_post_norm,
            .pre_ff_scratch = &t_ffn_pre,
            .gate_scratch = &t_ffn_gate,
            .up_scratch = &t_ffn_up,
            .ffn_out_scratch = &t_ffn_out,
            .post_ff_scratch = &t_ffn_post,
            .out = &t_ffn_y,
        };
        s = v->ffn_geglu_block(be, &block);
        fails += check(s == GEIST_OK, "device BF16 FFN GEGLU block succeeds");
    }
    if (s == GEIST_OK) {
        s = v->buffer_download(sizeof(ffn_bf16_got),
                               (uint8_t *) ffn_bf16_got, ffn_y_b);
        fails += check(s == GEIST_OK, "download BF16 FFN GEGLU output OK");
    }
    if (s == GEIST_OK) {
        fails += compare_f32_tol("bf16_ffn_geglu_block", ffn_d,
                                 ffn_bf16_got, ffn_bf16_ref, 2e-4f, 2e-4f);
    }
    if (ffn_residual_b != nullptr) { v->buffer_destroy(be, ffn_residual_b); }
    if (ffn_norm_b != nullptr) { v->buffer_destroy(be, ffn_norm_b); }
    if (ffn_gate_w_b != nullptr) { v->buffer_destroy(be, ffn_gate_w_b); }
    if (ffn_up_w_b != nullptr) { v->buffer_destroy(be, ffn_up_w_b); }
    if (ffn_down_w_b != nullptr) { v->buffer_destroy(be, ffn_down_w_b); }
    if (ffn_gate_w_f16_b != nullptr) { v->buffer_destroy(be, ffn_gate_w_f16_b); }
    if (ffn_up_w_f16_b != nullptr) { v->buffer_destroy(be, ffn_up_w_f16_b); }
    if (ffn_down_w_f16_b != nullptr) { v->buffer_destroy(be, ffn_down_w_f16_b); }
    if (ffn_gate_w_bf16_b != nullptr) { v->buffer_destroy(be, ffn_gate_w_bf16_b); }
    if (ffn_up_w_bf16_b != nullptr) { v->buffer_destroy(be, ffn_up_w_bf16_b); }
    if (ffn_down_w_bf16_b != nullptr) { v->buffer_destroy(be, ffn_down_w_bf16_b); }
    if (ffn_post_norm_b != nullptr) { v->buffer_destroy(be, ffn_post_norm_b); }
    if (ffn_pre_b != nullptr) { v->buffer_destroy(be, ffn_pre_b); }
    if (ffn_gate_b != nullptr) { v->buffer_destroy(be, ffn_gate_b); }
    if (ffn_up_b != nullptr) { v->buffer_destroy(be, ffn_up_b); }
    if (ffn_out_b != nullptr) { v->buffer_destroy(be, ffn_out_b); }
    if (ffn_post_b != nullptr) { v->buffer_destroy(be, ffn_post_b); }
    if (ffn_y_b != nullptr) { v->buffer_destroy(be, ffn_y_b); }

    constexpr size_t q6_ffn_n = 256;
    constexpr size_t q6_ffn_seq = 2;
    float q6_ffn_residual[q6_ffn_seq * q6_ffn_n];
    float q6_ffn_norm[q6_ffn_n];
    float q6_ffn_ref[q6_ffn_seq * q6_ffn_n];
    float q6_ffn_got[q6_ffn_seq * q6_ffn_n];
    uint8_t q6_ffn_w_data[q6_ffn_n * 210u];
    for (size_t i = 0; i < q6_ffn_seq * q6_ffn_n; i++) {
        q6_ffn_residual[i] =
            (float) ((int) (i % 23u) - 11) * 0.03125f;
        q6_ffn_got[i] = 0.0f;
    }
    for (size_t i = 0; i < q6_ffn_n; i++) {
        q6_ffn_norm[i] = 0.75f + (float) (i % 7u) * 0.03125f;
    }
    pack_q6k(q6_ffn_n, q6_ffn_w_data);
    for (size_t row = 0; row < q6_ffn_seq; row++) {
        ref_ffn_geglu_q6_block(q6_ffn_residual + row * q6_ffn_n,
                               q6_ffn_norm, 1e-5f,
                               q6_ffn_ref + row * q6_ffn_n);
    }

    struct geist_buffer *q6_ffn_residual_b = nullptr;
    struct geist_buffer *q6_ffn_norm_b = nullptr;
    struct geist_buffer *q6_ffn_gate_w_b = nullptr;
    struct geist_buffer *q6_ffn_up_w_b = nullptr;
    struct geist_buffer *q6_ffn_down_w_b = nullptr;
    struct geist_buffer *q6_ffn_pre_b = nullptr;
    struct geist_buffer *q6_ffn_gate_b = nullptr;
    struct geist_buffer *q6_ffn_up_b = nullptr;
    struct geist_buffer *q6_ffn_out_b = nullptr;
    struct geist_buffer *q6_ffn_y_b = nullptr;

    if (s == GEIST_OK) {
        s = v->buffer_create(be, sizeof(q6_ffn_residual),
                             GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_DEVICE,
                             &q6_ffn_residual_b);
        fails += check(s == GEIST_OK, "device Q6 FFN residual create OK");
    }
    if (s == GEIST_OK) {
        s = v->buffer_create(be, sizeof(q6_ffn_norm),
                             GEIST_BUFFER_WEIGHT, GEIST_MEMORY_DEVICE,
                             &q6_ffn_norm_b);
        fails += check(s == GEIST_OK, "device Q6 FFN norm create OK");
    }
    if (s == GEIST_OK) {
        s = v->buffer_create(be, sizeof(q6_ffn_w_data),
                             GEIST_BUFFER_WEIGHT, GEIST_MEMORY_DEVICE,
                             &q6_ffn_gate_w_b);
        fails += check(s == GEIST_OK, "device Q6 FFN gate weight create OK");
    }
    if (s == GEIST_OK) {
        s = v->buffer_create(be, sizeof(q6_ffn_w_data),
                             GEIST_BUFFER_WEIGHT, GEIST_MEMORY_DEVICE,
                             &q6_ffn_up_w_b);
        fails += check(s == GEIST_OK, "device Q6 FFN up weight create OK");
    }
    if (s == GEIST_OK) {
        s = v->buffer_create(be, sizeof(q6_ffn_w_data),
                             GEIST_BUFFER_WEIGHT, GEIST_MEMORY_DEVICE,
                             &q6_ffn_down_w_b);
        fails += check(s == GEIST_OK, "device Q6 FFN down weight create OK");
    }
    if (s == GEIST_OK) {
        s = v->buffer_create(be, q6_ffn_seq * q6_ffn_n * sizeof(float),
                             GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_DEVICE,
                             &q6_ffn_pre_b);
        fails += check(s == GEIST_OK, "device Q6 FFN pre scratch create OK");
    }
    if (s == GEIST_OK) {
        s = v->buffer_create(be, q6_ffn_seq * q6_ffn_n * sizeof(float),
                             GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_DEVICE,
                             &q6_ffn_gate_b);
        fails += check(s == GEIST_OK, "device Q6 FFN gate scratch create OK");
    }
    if (s == GEIST_OK) {
        s = v->buffer_create(be, q6_ffn_seq * q6_ffn_n * sizeof(float),
                             GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_DEVICE,
                             &q6_ffn_up_b);
        fails += check(s == GEIST_OK, "device Q6 FFN up scratch create OK");
    }
    if (s == GEIST_OK) {
        s = v->buffer_create(be, q6_ffn_seq * q6_ffn_n * sizeof(float),
                             GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_DEVICE,
                             &q6_ffn_out_b);
        fails += check(s == GEIST_OK, "device Q6 FFN out scratch create OK");
    }
    if (s == GEIST_OK) {
        s = v->buffer_create(be, q6_ffn_seq * q6_ffn_n * sizeof(float),
                             GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_DEVICE,
                             &q6_ffn_y_b);
        fails += check(s == GEIST_OK, "device Q6 FFN y create OK");
    }
    if (s == GEIST_OK) {
        s = v->buffer_upload(q6_ffn_residual_b, sizeof(q6_ffn_residual),
                             (const uint8_t *) q6_ffn_residual);
        fails += check(s == GEIST_OK, "upload Q6 FFN residual OK");
    }
    if (s == GEIST_OK) {
        s = v->buffer_upload(q6_ffn_norm_b, sizeof(q6_ffn_norm),
                             (const uint8_t *) q6_ffn_norm);
        fails += check(s == GEIST_OK, "upload Q6 FFN norm OK");
    }
    if (s == GEIST_OK) {
        s = v->buffer_upload(q6_ffn_gate_w_b, sizeof(q6_ffn_w_data),
                             q6_ffn_w_data);
        fails += check(s == GEIST_OK, "upload Q6 FFN gate weight OK");
    }
    if (s == GEIST_OK) {
        s = v->buffer_upload(q6_ffn_up_w_b, sizeof(q6_ffn_w_data),
                             q6_ffn_w_data);
        fails += check(s == GEIST_OK, "upload Q6 FFN up weight OK");
    }
    if (s == GEIST_OK) {
        s = v->buffer_upload(q6_ffn_down_w_b, sizeof(q6_ffn_w_data),
                             q6_ffn_w_data);
        fails += check(s == GEIST_OK, "upload Q6 FFN down weight OK");
    }
    if (s == GEIST_OK) {
        struct geist_tensor t_q6_ffn_residual =
            tensor_2d(q6_ffn_residual_b, q6_ffn_seq, q6_ffn_n);
        struct geist_tensor t_q6_ffn_norm =
            tensor_1d(q6_ffn_norm_b, q6_ffn_n);
        struct geist_tensor t_q6_ffn_gate_w =
            tensor_q6k_2d(q6_ffn_gate_w_b, q6_ffn_n, q6_ffn_n);
        struct geist_tensor t_q6_ffn_up_w =
            tensor_q6k_2d(q6_ffn_up_w_b, q6_ffn_n, q6_ffn_n);
        struct geist_tensor t_q6_ffn_down_w =
            tensor_q6k_2d(q6_ffn_down_w_b, q6_ffn_n, q6_ffn_n);
        struct geist_tensor t_q6_ffn_pre =
            tensor_2d(q6_ffn_pre_b, q6_ffn_seq, q6_ffn_n);
        struct geist_tensor t_q6_ffn_gate =
            tensor_2d(q6_ffn_gate_b, q6_ffn_seq, q6_ffn_n);
        struct geist_tensor t_q6_ffn_up =
            tensor_2d(q6_ffn_up_b, q6_ffn_seq, q6_ffn_n);
        struct geist_tensor t_q6_ffn_out =
            tensor_2d(q6_ffn_out_b, q6_ffn_seq, q6_ffn_n);
        struct geist_tensor t_q6_ffn_y =
            tensor_2d(q6_ffn_y_b, q6_ffn_seq, q6_ffn_n);
        const struct geist_backend_ffn_geglu_block q6_block = {
            .struct_size = sizeof(q6_block),
            .seq = q6_ffn_seq,
            .d_model = q6_ffn_n,
            .inter = q6_ffn_n,
            .eps = 1e-5f,
            .residual = &t_q6_ffn_residual,
            .ffn_norm_weight = &t_q6_ffn_norm,
            .gate_weight = &t_q6_ffn_gate_w,
            .up_weight = &t_q6_ffn_up_w,
            .down_weight = &t_q6_ffn_down_w,
            .post_ffw_norm_weight = nullptr,
            .pre_ff_scratch = &t_q6_ffn_pre,
            .gate_scratch = &t_q6_ffn_gate,
            .up_scratch = &t_q6_ffn_up,
            .ffn_out_scratch = &t_q6_ffn_out,
            .post_ff_scratch = nullptr,
            .out = &t_q6_ffn_y,
        };
        s = v->ffn_geglu_block(be, &q6_block);
        fails += check(s == GEIST_OK,
                       "device Q6 FFN GEGLU block succeeds");
    }
    if (s == GEIST_OK) {
        s = v->buffer_download(sizeof(q6_ffn_got),
                               (uint8_t *) q6_ffn_got, q6_ffn_y_b);
        fails += check(s == GEIST_OK, "download Q6 FFN output OK");
    }
    if (s == GEIST_OK) {
        fails += compare_f32_tol("q6_ffn_geglu_block",
                                 q6_ffn_seq * q6_ffn_n,
                                 q6_ffn_got, q6_ffn_ref, 1e-4f, 1e-2f);
    }
    if (q6_ffn_residual_b != nullptr) {
        v->buffer_destroy(be, q6_ffn_residual_b);
    }
    if (q6_ffn_norm_b != nullptr) { v->buffer_destroy(be, q6_ffn_norm_b); }
    if (q6_ffn_gate_w_b != nullptr) {
        v->buffer_destroy(be, q6_ffn_gate_w_b);
    }
    if (q6_ffn_up_w_b != nullptr) { v->buffer_destroy(be, q6_ffn_up_w_b); }
    if (q6_ffn_down_w_b != nullptr) {
        v->buffer_destroy(be, q6_ffn_down_w_b);
    }
    if (q6_ffn_pre_b != nullptr) { v->buffer_destroy(be, q6_ffn_pre_b); }
    if (q6_ffn_gate_b != nullptr) { v->buffer_destroy(be, q6_ffn_gate_b); }
    if (q6_ffn_up_b != nullptr) { v->buffer_destroy(be, q6_ffn_up_b); }
    if (q6_ffn_out_b != nullptr) { v->buffer_destroy(be, q6_ffn_out_b); }
    if (q6_ffn_y_b != nullptr) { v->buffer_destroy(be, q6_ffn_y_b); }

    constexpr size_t attn_d = 4;
    constexpr size_t attn_q_out = 4;
    constexpr size_t attn_kv_out = 2;
    constexpr size_t attn_kv_len = 2;
    const float attn_residual[4] = {0.30f, -0.20f, 0.10f, 0.40f};
    const float attn_norm[4] = {1.00f, 0.90f, 1.10f, 0.80f};
    const float attn_q_w[16] = {
        0.20f, -0.10f, 0.05f, 0.30f,
       -0.15f, 0.25f, 0.10f, -0.05f,
        0.12f, 0.07f, -0.18f, 0.22f,
        0.03f, -0.20f, 0.28f, 0.11f,
    };
    const float attn_k_w[8] = {
        0.16f, -0.08f, 0.12f, 0.04f,
       -0.06f, 0.18f, 0.09f, -0.14f,
    };
    const float attn_v_w[8] = {
       -0.11f, 0.21f, 0.05f, 0.13f,
        0.17f, -0.04f, 0.19f, 0.02f,
    };
    const float attn_q_norm[2] = {0.95f, 1.05f};
    const float attn_k_norm[2] = {1.10f, 0.85f};
    const float attn_v_norm[2] = {1.00f, 1.00f};
    const float attn_cos[2] = {1.00f, 1.00f};
    const float attn_sin[2] = {0.00f, 0.00f};
    float attn_k_cache[4] = {0.07f, -0.04f, 0.0f, 0.0f};
    float attn_v_cache[4] = {0.03f, 0.08f, 0.0f, 0.0f};
    const float attn_o_w[16] = {
        0.25f, -0.05f, 0.18f, 0.07f,
       -0.12f, 0.20f, 0.09f, -0.03f,
        0.04f, 0.11f, -0.16f, 0.24f,
        0.19f, -0.02f, 0.13f, 0.08f,
    };
    const float attn_post_norm[4] = {0.90f, 1.00f, 1.10f, 0.95f};
    float attn_ref[4];
    float attn_ref_k[2];
    float attn_ref_v[2];
    float attn_f16_ref[4];
    float attn_f16_ref_k[2];
    float attn_f16_ref_v[2];
    float attn_bf16_ref[4];
    float attn_bf16_ref_k[2];
    float attn_bf16_ref_v[2];
    float attn_got[4] = {0};
    float attn_query_got[4] = {0};
    float attn_f16_got[4] = {0};
    float attn_f16_query_got[4] = {0};
    float attn_bf16_got[4] = {0};
    float attn_bf16_query_got[4] = {0};
    float attn_k_cache_got[4] = {0};
    float attn_v_cache_got[4] = {0};
    uint16_t attn_q_w_f16[16];
    uint16_t attn_k_w_f16[8];
    uint16_t attn_v_w_f16[8];
    uint16_t attn_o_w_f16[16];
    uint16_t attn_q_w_bf16[16];
    uint16_t attn_k_w_bf16[8];
    uint16_t attn_v_w_bf16[8];
    uint16_t attn_o_w_bf16[16];
    float attn_q_w_f16_ref[16];
    float attn_k_w_f16_ref[8];
    float attn_v_w_f16_ref[8];
    float attn_o_w_f16_ref[16];
    float attn_q_w_bf16_ref[16];
    float attn_k_w_bf16_ref[8];
    float attn_v_w_bf16_ref[8];
    float attn_o_w_bf16_ref[16];
    ref_attention_block(attn_ref, attn_ref_k, attn_ref_v);
    encode_f16_array(16, attn_q_w, attn_q_w_f16, attn_q_w_f16_ref);
    encode_f16_array(8, attn_k_w, attn_k_w_f16, attn_k_w_f16_ref);
    encode_f16_array(8, attn_v_w, attn_v_w_f16, attn_v_w_f16_ref);
    encode_f16_array(16, attn_o_w, attn_o_w_f16, attn_o_w_f16_ref);
    encode_bf16_array(16, attn_q_w, attn_q_w_bf16, attn_q_w_bf16_ref);
    encode_bf16_array(8, attn_k_w, attn_k_w_bf16, attn_k_w_bf16_ref);
    encode_bf16_array(8, attn_v_w, attn_v_w_bf16, attn_v_w_bf16_ref);
    encode_bf16_array(16, attn_o_w, attn_o_w_bf16, attn_o_w_bf16_ref);
    ref_attention_block_weights(attn_q_w_f16_ref, attn_k_w_f16_ref,
                                attn_v_w_f16_ref, attn_o_w_f16_ref,
                                attn_f16_ref, attn_f16_ref_k,
                                attn_f16_ref_v);
    ref_attention_block_weights(attn_q_w_bf16_ref, attn_k_w_bf16_ref,
                                attn_v_w_bf16_ref, attn_o_w_bf16_ref,
                                attn_bf16_ref, attn_bf16_ref_k,
                                attn_bf16_ref_v);

    struct geist_buffer *attn_residual_b = nullptr;
    struct geist_buffer *attn_norm_b = nullptr;
    struct geist_buffer *attn_q_w_b = nullptr;
    struct geist_buffer *attn_k_w_b = nullptr;
    struct geist_buffer *attn_v_w_b = nullptr;
    struct geist_buffer *attn_q_w_f16_b = nullptr;
    struct geist_buffer *attn_k_w_f16_b = nullptr;
    struct geist_buffer *attn_v_w_f16_b = nullptr;
    struct geist_buffer *attn_o_w_f16_b = nullptr;
    struct geist_buffer *attn_q_w_bf16_b = nullptr;
    struct geist_buffer *attn_k_w_bf16_b = nullptr;
    struct geist_buffer *attn_v_w_bf16_b = nullptr;
    struct geist_buffer *attn_o_w_bf16_b = nullptr;
    struct geist_buffer *attn_q_norm_b = nullptr;
    struct geist_buffer *attn_k_norm_b = nullptr;
    struct geist_buffer *attn_v_norm_b = nullptr;
    struct geist_buffer *attn_cos_b = nullptr;
    struct geist_buffer *attn_sin_b = nullptr;
    struct geist_buffer *attn_k_cache_b = nullptr;
    struct geist_buffer *attn_v_cache_b = nullptr;
    struct geist_buffer *attn_o_w_b = nullptr;
    struct geist_buffer *attn_post_norm_b = nullptr;
    struct geist_buffer *attn_normed_b = nullptr;
    struct geist_buffer *attn_q_b = nullptr;
    struct geist_buffer *attn_k_b = nullptr;
    struct geist_buffer *attn_v_b = nullptr;
    struct geist_buffer *attn_mid_b = nullptr;
    struct geist_buffer *attn_o_b = nullptr;
    struct geist_buffer *attn_post_b = nullptr;
    struct geist_buffer *attn_out_b = nullptr;

#define CREATE_ATTN_BUF(var, bytes, role, label) \
    do { \
        if (s == GEIST_OK) { \
            s = v->buffer_create(be, (bytes), (role), GEIST_MEMORY_DEVICE, &(var)); \
            fails += check(s == GEIST_OK, label); \
        } \
    } while (0)
#define UPLOAD_ATTN_BUF(var, data, label) \
    do { \
        if (s == GEIST_OK) { \
            s = v->buffer_upload((var), sizeof(data), (const uint8_t *) (data)); \
            fails += check(s == GEIST_OK, label); \
        } \
    } while (0)

    CREATE_ATTN_BUF(attn_residual_b, sizeof(attn_residual),
                    GEIST_BUFFER_ACTIVATION, "device attention residual create OK");
    CREATE_ATTN_BUF(attn_norm_b, sizeof(attn_norm),
                    GEIST_BUFFER_WEIGHT, "device attention norm create OK");
    CREATE_ATTN_BUF(attn_q_w_b, sizeof(attn_q_w),
                    GEIST_BUFFER_WEIGHT, "device attention q weight create OK");
    CREATE_ATTN_BUF(attn_k_w_b, sizeof(attn_k_w),
                    GEIST_BUFFER_WEIGHT, "device attention k weight create OK");
    CREATE_ATTN_BUF(attn_v_w_b, sizeof(attn_v_w),
                    GEIST_BUFFER_WEIGHT, "device attention v weight create OK");
    CREATE_ATTN_BUF(attn_q_w_f16_b, sizeof(attn_q_w_f16),
                    GEIST_BUFFER_WEIGHT, "device F16 attention q weight create OK");
    CREATE_ATTN_BUF(attn_k_w_f16_b, sizeof(attn_k_w_f16),
                    GEIST_BUFFER_WEIGHT, "device F16 attention k weight create OK");
    CREATE_ATTN_BUF(attn_v_w_f16_b, sizeof(attn_v_w_f16),
                    GEIST_BUFFER_WEIGHT, "device F16 attention v weight create OK");
    CREATE_ATTN_BUF(attn_o_w_f16_b, sizeof(attn_o_w_f16),
                    GEIST_BUFFER_WEIGHT, "device F16 attention o weight create OK");
    CREATE_ATTN_BUF(attn_q_w_bf16_b, sizeof(attn_q_w_bf16),
                    GEIST_BUFFER_WEIGHT, "device BF16 attention q weight create OK");
    CREATE_ATTN_BUF(attn_k_w_bf16_b, sizeof(attn_k_w_bf16),
                    GEIST_BUFFER_WEIGHT, "device BF16 attention k weight create OK");
    CREATE_ATTN_BUF(attn_v_w_bf16_b, sizeof(attn_v_w_bf16),
                    GEIST_BUFFER_WEIGHT, "device BF16 attention v weight create OK");
    CREATE_ATTN_BUF(attn_o_w_bf16_b, sizeof(attn_o_w_bf16),
                    GEIST_BUFFER_WEIGHT, "device BF16 attention o weight create OK");
    CREATE_ATTN_BUF(attn_q_norm_b, sizeof(attn_q_norm),
                    GEIST_BUFFER_WEIGHT, "device attention q norm create OK");
    CREATE_ATTN_BUF(attn_k_norm_b, sizeof(attn_k_norm),
                    GEIST_BUFFER_WEIGHT, "device attention k norm create OK");
    CREATE_ATTN_BUF(attn_v_norm_b, sizeof(attn_v_norm),
                    GEIST_BUFFER_WEIGHT, "device attention v norm create OK");
    CREATE_ATTN_BUF(attn_cos_b, sizeof(attn_cos),
                    GEIST_BUFFER_WEIGHT, "device attention cos create OK");
    CREATE_ATTN_BUF(attn_sin_b, sizeof(attn_sin),
                    GEIST_BUFFER_WEIGHT, "device attention sin create OK");
    CREATE_ATTN_BUF(attn_k_cache_b, sizeof(attn_k_cache),
                    GEIST_BUFFER_KV_CACHE, "device attention k cache create OK");
    CREATE_ATTN_BUF(attn_v_cache_b, sizeof(attn_v_cache),
                    GEIST_BUFFER_KV_CACHE, "device attention v cache create OK");
    CREATE_ATTN_BUF(attn_o_w_b, sizeof(attn_o_w),
                    GEIST_BUFFER_WEIGHT, "device attention o weight create OK");
    CREATE_ATTN_BUF(attn_post_norm_b, sizeof(attn_post_norm),
                    GEIST_BUFFER_WEIGHT, "device attention post norm create OK");
    CREATE_ATTN_BUF(attn_normed_b, attn_d * sizeof(float),
                    GEIST_BUFFER_ACTIVATION, "device attention normed scratch create OK");
    CREATE_ATTN_BUF(attn_q_b, attn_q_out * sizeof(float),
                    GEIST_BUFFER_ACTIVATION, "device attention q scratch create OK");
    CREATE_ATTN_BUF(attn_k_b, attn_kv_out * sizeof(float),
                    GEIST_BUFFER_ACTIVATION, "device attention k scratch create OK");
    CREATE_ATTN_BUF(attn_v_b, attn_kv_out * sizeof(float),
                    GEIST_BUFFER_ACTIVATION, "device attention v scratch create OK");
    CREATE_ATTN_BUF(attn_mid_b, attn_q_out * sizeof(float),
                    GEIST_BUFFER_ACTIVATION, "device attention mid scratch create OK");
    CREATE_ATTN_BUF(attn_o_b, attn_d * sizeof(float),
                    GEIST_BUFFER_ACTIVATION, "device attention o scratch create OK");
    CREATE_ATTN_BUF(attn_post_b, attn_d * sizeof(float),
                    GEIST_BUFFER_ACTIVATION, "device attention post scratch create OK");
    CREATE_ATTN_BUF(attn_out_b, attn_d * sizeof(float),
                    GEIST_BUFFER_ACTIVATION, "device attention out create OK");

    UPLOAD_ATTN_BUF(attn_residual_b, attn_residual,
                    "upload attention residual OK");
    UPLOAD_ATTN_BUF(attn_norm_b, attn_norm, "upload attention norm OK");
    UPLOAD_ATTN_BUF(attn_q_w_b, attn_q_w, "upload attention q weight OK");
    UPLOAD_ATTN_BUF(attn_k_w_b, attn_k_w, "upload attention k weight OK");
    UPLOAD_ATTN_BUF(attn_v_w_b, attn_v_w, "upload attention v weight OK");
    UPLOAD_ATTN_BUF(attn_q_w_f16_b, attn_q_w_f16,
                    "upload F16 attention q weight OK");
    UPLOAD_ATTN_BUF(attn_k_w_f16_b, attn_k_w_f16,
                    "upload F16 attention k weight OK");
    UPLOAD_ATTN_BUF(attn_v_w_f16_b, attn_v_w_f16,
                    "upload F16 attention v weight OK");
    UPLOAD_ATTN_BUF(attn_o_w_f16_b, attn_o_w_f16,
                    "upload F16 attention o weight OK");
    UPLOAD_ATTN_BUF(attn_q_w_bf16_b, attn_q_w_bf16,
                    "upload BF16 attention q weight OK");
    UPLOAD_ATTN_BUF(attn_k_w_bf16_b, attn_k_w_bf16,
                    "upload BF16 attention k weight OK");
    UPLOAD_ATTN_BUF(attn_v_w_bf16_b, attn_v_w_bf16,
                    "upload BF16 attention v weight OK");
    UPLOAD_ATTN_BUF(attn_o_w_bf16_b, attn_o_w_bf16,
                    "upload BF16 attention o weight OK");
    UPLOAD_ATTN_BUF(attn_q_norm_b, attn_q_norm,
                    "upload attention q norm OK");
    UPLOAD_ATTN_BUF(attn_k_norm_b, attn_k_norm,
                    "upload attention k norm OK");
    UPLOAD_ATTN_BUF(attn_v_norm_b, attn_v_norm,
                    "upload attention v norm OK");
    UPLOAD_ATTN_BUF(attn_cos_b, attn_cos, "upload attention cos OK");
    UPLOAD_ATTN_BUF(attn_sin_b, attn_sin, "upload attention sin OK");
    UPLOAD_ATTN_BUF(attn_k_cache_b, attn_k_cache,
                    "upload attention k cache OK");
    UPLOAD_ATTN_BUF(attn_v_cache_b, attn_v_cache,
                    "upload attention v cache OK");
    UPLOAD_ATTN_BUF(attn_o_w_b, attn_o_w, "upload attention o weight OK");
    UPLOAD_ATTN_BUF(attn_post_norm_b, attn_post_norm,
                    "upload attention post norm OK");

    if (s == GEIST_OK) {
        struct geist_tensor t_attn_residual =
            tensor_2d(attn_residual_b, 1, attn_d);
        struct geist_tensor t_attn_norm = tensor_1d(attn_norm_b, attn_d);
        struct geist_tensor t_attn_q_w =
            tensor_2d(attn_q_w_b, attn_q_out, attn_d);
        struct geist_tensor t_attn_k_w =
            tensor_2d(attn_k_w_b, attn_kv_out, attn_d);
        struct geist_tensor t_attn_v_w =
            tensor_2d(attn_v_w_b, attn_kv_out, attn_d);
        struct geist_tensor t_attn_q_norm = tensor_1d(attn_q_norm_b, 2);
        struct geist_tensor t_attn_k_norm = tensor_1d(attn_k_norm_b, 2);
        struct geist_tensor t_attn_v_norm = tensor_1d(attn_v_norm_b, 2);
        struct geist_tensor t_attn_cos = tensor_2d(attn_cos_b, 1, 2);
        struct geist_tensor t_attn_sin = tensor_2d(attn_sin_b, 1, 2);
        struct geist_tensor t_attn_k_cache =
            tensor_3d(attn_k_cache_b, attn_kv_len, 1, 2);
        struct geist_tensor t_attn_v_cache =
            tensor_3d(attn_v_cache_b, attn_kv_len, 1, 2);
        struct geist_tensor t_attn_o_w =
            tensor_2d(attn_o_w_b, attn_d, attn_q_out);
        struct geist_tensor t_attn_post_norm =
            tensor_1d(attn_post_norm_b, attn_d);
        struct geist_tensor t_attn_normed =
            tensor_2d(attn_normed_b, 1, attn_d);
        struct geist_tensor t_attn_q = tensor_2d(attn_q_b, 1, attn_q_out);
        struct geist_tensor t_attn_k = tensor_2d(attn_k_b, 1, attn_kv_out);
        struct geist_tensor t_attn_v = tensor_2d(attn_v_b, 1, attn_kv_out);
        struct geist_tensor t_attn_mid =
            tensor_2d(attn_mid_b, 1, attn_q_out);
        struct geist_tensor t_attn_o = tensor_2d(attn_o_b, 1, attn_d);
        struct geist_tensor t_attn_post = tensor_2d(attn_post_b, 1, attn_d);
        struct geist_tensor t_attn_out = tensor_2d(attn_out_b, 1, attn_d);
        const struct geist_backend_attention_block attn_block = {
            .struct_size = sizeof(attn_block),
            .q_position = 1,
            .kv_len = attn_kv_len,
            .d_model = attn_d,
            .q_heads = 2,
            .kv_heads = 1,
            .head_dim = 2,
            .sliding_window = 0,
            .eps = 1e-5f,
            .residual = &t_attn_residual,
            .attn_norm_weight = &t_attn_norm,
            .q_proj_weight = &t_attn_q_w,
            .k_proj_weight = &t_attn_k_w,
            .v_proj_weight = &t_attn_v_w,
            .q_norm_weight = &t_attn_q_norm,
            .k_norm_weight = &t_attn_k_norm,
            .v_norm_weight = &t_attn_v_norm,
            .cos = &t_attn_cos,
            .sin = &t_attn_sin,
            .k_cache = &t_attn_k_cache,
            .v_cache = &t_attn_v_cache,
            .o_proj_weight = &t_attn_o_w,
            .post_attn_norm_weight = &t_attn_post_norm,
            .normed_scratch = &t_attn_normed,
            .q_scratch = &t_attn_q,
            .k_scratch = &t_attn_k,
            .v_scratch = &t_attn_v,
            .attn_scratch = &t_attn_mid,
            .o_scratch = &t_attn_o,
            .post_attn_scratch = &t_attn_post,
            .out = &t_attn_out,
        };
        int block_capture_token = 0;
        enum geist_status bs = v->command_sequence_begin(
            be, GEIST_COMMAND_SEQUENCE_DECODE_GREEDY_STEP,
            &block_capture_token);
        fails += check(bs == GEIST_OK,
                       "captured attention block begin OK");
        if (bs == GEIST_OK) {
            bs = v->attention_block(be, &attn_block);
            fails += check(bs == GEIST_OK,
                           "captured attention block records OK");
        }
        if (block_capture_token != 0) {
            const bool submit = bs == GEIST_OK;
            enum geist_status end_s =
                v->command_sequence_end(be, block_capture_token, submit);
            fails += check(end_s == GEIST_OK,
                           "captured attention block end OK");
            if (bs == GEIST_OK) { bs = end_s; }
        }
        s = bs;
        fails += check(s == GEIST_OK,
                       "captured attention block succeeds");
        if (s == GEIST_OK) {
            const struct geist_backend_attention_query_block query_block = {
                .struct_size = sizeof(query_block),
                .q_position = 1,
                .kv_len = attn_kv_len,
                .d_model = attn_d,
                .q_heads = 2,
                .kv_heads = 1,
                .head_dim = 2,
                .sliding_window = 0,
                .eps = 1e-5f,
                .residual = &t_attn_residual,
                .attn_norm_weight = &t_attn_norm,
                .q_proj_weight = &t_attn_q_w,
                .q_norm_weight = &t_attn_q_norm,
                .cos = &t_attn_cos,
                .sin = &t_attn_sin,
                .k_cache = &t_attn_k_cache,
                .v_cache = &t_attn_v_cache,
                .o_proj_weight = &t_attn_o_w,
                .post_attn_norm_weight = &t_attn_post_norm,
                .normed_scratch = &t_attn_normed,
                .q_scratch = &t_attn_q,
                .attn_scratch = &t_attn_mid,
                .o_scratch = &t_attn_o,
                .post_attn_scratch = &t_attn_post,
                .out = &t_attn_out,
            };
            int query_capture_token = 0;
            enum geist_status qs = v->command_sequence_begin(
                be, GEIST_COMMAND_SEQUENCE_DECODE_GREEDY_STEP,
                &query_capture_token);
            fails += check(qs == GEIST_OK,
                           "captured attention query begin OK");
            if (qs == GEIST_OK) {
                qs = v->attention_query_block(be, &query_block);
                fails += check(qs == GEIST_OK,
                               "captured attention query records OK");
            }
            if (query_capture_token != 0) {
                const bool submit = qs == GEIST_OK;
                enum geist_status end_s =
                    v->command_sequence_end(be, query_capture_token, submit);
                fails += check(end_s == GEIST_OK,
                               "captured attention query end OK");
                if (qs == GEIST_OK) { qs = end_s; }
            }
            s = qs;
            fails += check(s == GEIST_OK,
                           "captured attention query block succeeds");
        }
    }
    if (s == GEIST_OK) {
        s = v->buffer_download(sizeof(attn_got),
                               (uint8_t *) attn_got, attn_out_b);
        fails += check(s == GEIST_OK, "download attention block output OK");
        memcpy(attn_query_got, attn_got, sizeof(attn_query_got));
    }
    if (s == GEIST_OK) {
        s = v->buffer_download(sizeof(attn_k_cache_got),
                               (uint8_t *) attn_k_cache_got, attn_k_cache_b);
        fails += check(s == GEIST_OK, "download attention k cache OK");
    }
    if (s == GEIST_OK) {
        s = v->buffer_download(sizeof(attn_v_cache_got),
                               (uint8_t *) attn_v_cache_got, attn_v_cache_b);
        fails += check(s == GEIST_OK, "download attention v cache OK");
    }
    if (s == GEIST_OK) {
        fails += compare_f32_tol("attention_block", attn_d,
                                 attn_got, attn_ref, 1e-4f, 2e-4f);
        fails += compare_f32_tol("attention_query_block", attn_d,
                                 attn_query_got, attn_ref, 1e-4f, 2e-4f);
        fails += compare_f32_tol("attention_block k append", 2,
                                 attn_k_cache_got + 2, attn_ref_k,
                                 1e-4f, 2e-4f);
        fails += compare_f32_tol("attention_block v append", 2,
                                 attn_v_cache_got + 2, attn_ref_v,
                                 1e-4f, 2e-4f);
    }

#define RUN_ATTN_HALF_VARIANT(label, dtype_value, qbuf, kbuf, vbuf, obuf, \
                              out_arr, query_arr, ref_arr, ref_k_arr, ref_v_arr) \
    do { \
        if (s == GEIST_OK) { \
            s = v->buffer_upload(attn_k_cache_b, sizeof(attn_k_cache), \
                                 (const uint8_t *) attn_k_cache); \
            fails += check(s == GEIST_OK, label " reset k cache OK"); \
        } \
        if (s == GEIST_OK) { \
            s = v->buffer_upload(attn_v_cache_b, sizeof(attn_v_cache), \
                                 (const uint8_t *) attn_v_cache); \
            fails += check(s == GEIST_OK, label " reset v cache OK"); \
        } \
        if (s == GEIST_OK) { \
            struct geist_tensor t_attn_residual = \
                tensor_2d(attn_residual_b, 1, attn_d); \
            struct geist_tensor t_attn_norm = tensor_1d(attn_norm_b, attn_d); \
            struct geist_tensor t_attn_q_w = \
                tensor_2d_dtype((qbuf), (dtype_value), attn_q_out, attn_d); \
            struct geist_tensor t_attn_k_w = \
                tensor_2d_dtype((kbuf), (dtype_value), attn_kv_out, attn_d); \
            struct geist_tensor t_attn_v_w = \
                tensor_2d_dtype((vbuf), (dtype_value), attn_kv_out, attn_d); \
            struct geist_tensor t_attn_q_norm = tensor_1d(attn_q_norm_b, 2); \
            struct geist_tensor t_attn_k_norm = tensor_1d(attn_k_norm_b, 2); \
            struct geist_tensor t_attn_v_norm = tensor_1d(attn_v_norm_b, 2); \
            struct geist_tensor t_attn_cos = tensor_2d(attn_cos_b, 1, 2); \
            struct geist_tensor t_attn_sin = tensor_2d(attn_sin_b, 1, 2); \
            struct geist_tensor t_attn_k_cache = \
                tensor_3d(attn_k_cache_b, attn_kv_len, 1, 2); \
            struct geist_tensor t_attn_v_cache = \
                tensor_3d(attn_v_cache_b, attn_kv_len, 1, 2); \
            struct geist_tensor t_attn_o_w = \
                tensor_2d_dtype((obuf), (dtype_value), attn_d, attn_q_out); \
            struct geist_tensor t_attn_post_norm = \
                tensor_1d(attn_post_norm_b, attn_d); \
            struct geist_tensor t_attn_normed = \
                tensor_2d(attn_normed_b, 1, attn_d); \
            struct geist_tensor t_attn_q = \
                tensor_2d(attn_q_b, 1, attn_q_out); \
            struct geist_tensor t_attn_k = \
                tensor_2d(attn_k_b, 1, attn_kv_out); \
            struct geist_tensor t_attn_v = \
                tensor_2d(attn_v_b, 1, attn_kv_out); \
            struct geist_tensor t_attn_mid = \
                tensor_2d(attn_mid_b, 1, attn_q_out); \
            struct geist_tensor t_attn_o = tensor_2d(attn_o_b, 1, attn_d); \
            struct geist_tensor t_attn_post = \
                tensor_2d(attn_post_b, 1, attn_d); \
            struct geist_tensor t_attn_out = \
                tensor_2d(attn_out_b, 1, attn_d); \
            const struct geist_backend_attention_block block = { \
                .struct_size = sizeof(block), \
                .q_position = 1, \
                .kv_len = attn_kv_len, \
                .d_model = attn_d, \
                .q_heads = 2, \
                .kv_heads = 1, \
                .head_dim = 2, \
                .sliding_window = 0, \
                .eps = 1e-5f, \
                .residual = &t_attn_residual, \
                .attn_norm_weight = &t_attn_norm, \
                .q_proj_weight = &t_attn_q_w, \
                .k_proj_weight = &t_attn_k_w, \
                .v_proj_weight = &t_attn_v_w, \
                .q_norm_weight = &t_attn_q_norm, \
                .k_norm_weight = &t_attn_k_norm, \
                .v_norm_weight = &t_attn_v_norm, \
                .cos = &t_attn_cos, \
                .sin = &t_attn_sin, \
                .k_cache = &t_attn_k_cache, \
                .v_cache = &t_attn_v_cache, \
                .o_proj_weight = &t_attn_o_w, \
                .post_attn_norm_weight = &t_attn_post_norm, \
                .normed_scratch = &t_attn_normed, \
                .q_scratch = &t_attn_q, \
                .k_scratch = &t_attn_k, \
                .v_scratch = &t_attn_v, \
                .attn_scratch = &t_attn_mid, \
                .o_scratch = &t_attn_o, \
                .post_attn_scratch = &t_attn_post, \
                .out = &t_attn_out, \
            }; \
            s = v->attention_block(be, &block); \
            fails += check(s == GEIST_OK, label " attention block succeeds"); \
            if (s == GEIST_OK) { \
                const struct geist_backend_attention_query_block query = { \
                    .struct_size = sizeof(query), \
                    .q_position = 1, \
                    .kv_len = attn_kv_len, \
                    .d_model = attn_d, \
                    .q_heads = 2, \
                    .kv_heads = 1, \
                    .head_dim = 2, \
                    .sliding_window = 0, \
                    .eps = 1e-5f, \
                    .residual = &t_attn_residual, \
                    .attn_norm_weight = &t_attn_norm, \
                    .q_proj_weight = &t_attn_q_w, \
                    .q_norm_weight = &t_attn_q_norm, \
                    .cos = &t_attn_cos, \
                    .sin = &t_attn_sin, \
                    .k_cache = &t_attn_k_cache, \
                    .v_cache = &t_attn_v_cache, \
                    .o_proj_weight = &t_attn_o_w, \
                    .post_attn_norm_weight = &t_attn_post_norm, \
                    .normed_scratch = &t_attn_normed, \
                    .q_scratch = &t_attn_q, \
                    .attn_scratch = &t_attn_mid, \
                    .o_scratch = &t_attn_o, \
                    .post_attn_scratch = &t_attn_post, \
                    .out = &t_attn_out, \
                }; \
                s = v->buffer_download(sizeof(out_arr), (uint8_t *) (out_arr), \
                                       attn_out_b); \
                fails += check(s == GEIST_OK, label " attention output download OK"); \
                if (s == GEIST_OK) { \
                    s = v->attention_query_block(be, &query); \
                    fails += check(s == GEIST_OK, \
                                   label " attention query block succeeds"); \
                } \
                if (s == GEIST_OK) { \
                    s = v->buffer_download(sizeof(query_arr), \
                                           (uint8_t *) (query_arr), attn_out_b); \
                    fails += check(s == GEIST_OK, \
                                   label " attention query output download OK"); \
                } \
            } \
        } \
        if (s == GEIST_OK) { \
            s = v->buffer_download(sizeof(attn_k_cache_got), \
                                   (uint8_t *) attn_k_cache_got, \
                                   attn_k_cache_b); \
            fails += check(s == GEIST_OK, label " k cache download OK"); \
        } \
        if (s == GEIST_OK) { \
            s = v->buffer_download(sizeof(attn_v_cache_got), \
                                   (uint8_t *) attn_v_cache_got, \
                                   attn_v_cache_b); \
            fails += check(s == GEIST_OK, label " v cache download OK"); \
        } \
        if (s == GEIST_OK) { \
            fails += compare_f32_tol(label " attention_block", attn_d, \
                                     (out_arr), (ref_arr), 3e-4f, 3e-4f); \
            fails += compare_f32_tol(label " attention_query_block", attn_d, \
                                     (query_arr), (ref_arr), 3e-4f, 3e-4f); \
            fails += compare_f32_tol(label " attention_block k append", 2, \
                                     attn_k_cache_got + 2, (ref_k_arr), \
                                     3e-4f, 3e-4f); \
            fails += compare_f32_tol(label " attention_block v append", 2, \
                                     attn_v_cache_got + 2, (ref_v_arr), \
                                     3e-4f, 3e-4f); \
        } \
    } while (0)

    RUN_ATTN_HALF_VARIANT("F16", GEIST_DTYPE_F16, attn_q_w_f16_b,
                          attn_k_w_f16_b, attn_v_w_f16_b, attn_o_w_f16_b,
                          attn_f16_got, attn_f16_query_got, attn_f16_ref,
                          attn_f16_ref_k, attn_f16_ref_v);
    RUN_ATTN_HALF_VARIANT("BF16", GEIST_DTYPE_BF16, attn_q_w_bf16_b,
                          attn_k_w_bf16_b, attn_v_w_bf16_b, attn_o_w_bf16_b,
                          attn_bf16_got, attn_bf16_query_got, attn_bf16_ref,
                          attn_bf16_ref_k, attn_bf16_ref_v);

#undef RUN_ATTN_HALF_VARIANT

    if (attn_residual_b != nullptr) { v->buffer_destroy(be, attn_residual_b); }
    if (attn_norm_b != nullptr) { v->buffer_destroy(be, attn_norm_b); }
    if (attn_q_w_b != nullptr) { v->buffer_destroy(be, attn_q_w_b); }
    if (attn_k_w_b != nullptr) { v->buffer_destroy(be, attn_k_w_b); }
    if (attn_v_w_b != nullptr) { v->buffer_destroy(be, attn_v_w_b); }
    if (attn_q_w_f16_b != nullptr) { v->buffer_destroy(be, attn_q_w_f16_b); }
    if (attn_k_w_f16_b != nullptr) { v->buffer_destroy(be, attn_k_w_f16_b); }
    if (attn_v_w_f16_b != nullptr) { v->buffer_destroy(be, attn_v_w_f16_b); }
    if (attn_o_w_f16_b != nullptr) { v->buffer_destroy(be, attn_o_w_f16_b); }
    if (attn_q_w_bf16_b != nullptr) { v->buffer_destroy(be, attn_q_w_bf16_b); }
    if (attn_k_w_bf16_b != nullptr) { v->buffer_destroy(be, attn_k_w_bf16_b); }
    if (attn_v_w_bf16_b != nullptr) { v->buffer_destroy(be, attn_v_w_bf16_b); }
    if (attn_o_w_bf16_b != nullptr) { v->buffer_destroy(be, attn_o_w_bf16_b); }
    if (attn_q_norm_b != nullptr) { v->buffer_destroy(be, attn_q_norm_b); }
    if (attn_k_norm_b != nullptr) { v->buffer_destroy(be, attn_k_norm_b); }
    if (attn_v_norm_b != nullptr) { v->buffer_destroy(be, attn_v_norm_b); }
    if (attn_cos_b != nullptr) { v->buffer_destroy(be, attn_cos_b); }
    if (attn_sin_b != nullptr) { v->buffer_destroy(be, attn_sin_b); }
    if (attn_k_cache_b != nullptr) { v->buffer_destroy(be, attn_k_cache_b); }
    if (attn_v_cache_b != nullptr) { v->buffer_destroy(be, attn_v_cache_b); }
    if (attn_o_w_b != nullptr) { v->buffer_destroy(be, attn_o_w_b); }
    if (attn_post_norm_b != nullptr) {
        v->buffer_destroy(be, attn_post_norm_b);
    }
    if (attn_normed_b != nullptr) { v->buffer_destroy(be, attn_normed_b); }
    if (attn_q_b != nullptr) { v->buffer_destroy(be, attn_q_b); }
    if (attn_k_b != nullptr) { v->buffer_destroy(be, attn_k_b); }
    if (attn_v_b != nullptr) { v->buffer_destroy(be, attn_v_b); }
    if (attn_mid_b != nullptr) { v->buffer_destroy(be, attn_mid_b); }
    if (attn_o_b != nullptr) { v->buffer_destroy(be, attn_o_b); }
    if (attn_post_b != nullptr) { v->buffer_destroy(be, attn_post_b); }
    if (attn_out_b != nullptr) { v->buffer_destroy(be, attn_out_b); }

#undef UPLOAD_ATTN_BUF
#undef CREATE_ATTN_BUF

    if (x != nullptr) { v->buffer_destroy(be, x); }
    if (w0 != nullptr) { v->buffer_destroy(be, w0); }
    if (w1 != nullptr) { v->buffer_destroy(be, w1); }
    if (w2 != nullptr) { v->buffer_destroy(be, w2); }
    if (y0 != nullptr) { v->buffer_destroy(be, y0); }
    if (y1 != nullptr) { v->buffer_destroy(be, y1); }
    if (y2 != nullptr) { v->buffer_destroy(be, y2); }
    geist_backend_destroy(be);

    if (fails == 0) {
        printf("PASS: transformer Vulkan linear helper device matvec\n");
        return GEIST_TEST_PASS;
    }
    fprintf(stderr, "FAILED: %d check(s)\n", fails);
    return GEIST_TEST_FAIL;
}
