/*
 * test_backend_metal_q4k_unit - Metal Q4_K decode matvec correctness.
 *
 * This is the first performance-oriented Metal primitive: a device-resident
 * m=1 Q4_K matvec over GGUF's raw Q4_K block layout.
 */
#include "test_helpers.h"

#include <geist.h>
#include <geist_util.h>
#include <geist_backend.h>
#include "heap.h"

#include <dirent.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int check(bool cond, const char *what) {
    if (!cond) {
        fprintf(stderr, "FAIL: %s\n", what);
        return 1;
    }
    return 0;
}

static int check_status(enum geist_status got,
                        enum geist_status want,
                        const char *what) {
    if (got != want) {
        fprintf(stderr, "FAIL: %s: got %s, want %s\n",
                what, geist_status_to_string(got),
                geist_status_to_string(want));
        return 1;
    }
    return 0;
}

struct cache_file_counts {
    size_t total;
    size_t q4k;
    size_t q6k;
};

static struct cache_file_counts count_and_remove_cache_files(const char *dir) {
    struct cache_file_counts counts = {0};
    if (dir == nullptr) {
        return counts;
    }
    DIR *d = opendir(dir);
    if (d == nullptr) {
        return counts;
    }
    struct dirent *ent = nullptr;
    while ((ent = readdir(d)) != nullptr) {
        if (strcmp(ent->d_name, ".") == 0 ||
            strcmp(ent->d_name, "..") == 0) {
            continue;
        }
        char path[1024];
        const int wrote = snprintf(path, sizeof(path), "%s/%s", dir,
                                   ent->d_name);
        if (wrote <= 0 || (size_t) wrote >= sizeof(path)) {
            continue;
        }
        counts.total++;
        if (strstr(ent->d_name, "q4k-nt4") != nullptr) {
            counts.q4k++;
        }
        if (strstr(ent->d_name, "q6k-nt4") != nullptr) {
            counts.q6k++;
        }
        (void) unlink(path);
    }
    (void) closedir(d);
    return counts;
}

static struct geist_tensor tensor_f32_1d(struct geist_buffer *buf, size_t n) {
    return (struct geist_tensor){
        .buffer = buf,
        .dtype = GEIST_DTYPE_F32,
        .layout = GEIST_LAYOUT_DENSE,
        .ndim = 1,
        .shape = {(int64_t) n},
        .stride = {1},
    };
}

static struct geist_tensor tensor_f32_2d(struct geist_buffer *buf,
                                         size_t rows,
                                         size_t cols) {
    return (struct geist_tensor){
        .buffer = buf,
        .dtype = GEIST_DTYPE_F32,
        .layout = GEIST_LAYOUT_DENSE,
        .ndim = 2,
        .shape = {(int64_t) rows, (int64_t) cols},
        .stride = {(int64_t) cols, 1},
    };
}

static struct geist_tensor tensor_f32_3d(struct geist_buffer *buf,
                                         size_t d0,
                                         size_t d1,
                                         size_t d2) {
    return (struct geist_tensor){
        .buffer = buf,
        .dtype = GEIST_DTYPE_F32,
        .layout = GEIST_LAYOUT_DENSE,
        .ndim = 3,
        .shape = {(int64_t) d0, (int64_t) d1, (int64_t) d2},
        .stride = {(int64_t) (d1 * d2), (int64_t) d2, 1},
    };
}

static struct geist_tensor tensor_f16_3d(struct geist_buffer *buf,
                                         size_t d0,
                                         size_t d1,
                                         size_t d2) {
    return (struct geist_tensor){
        .buffer = buf,
        .dtype = GEIST_DTYPE_F16,
        .layout = GEIST_LAYOUT_DENSE,
        .ndim = 3,
        .shape = {(int64_t) d0, (int64_t) d1, (int64_t) d2},
        .stride = {(int64_t) (d1 * d2), (int64_t) d2, 1},
    };
}

static uint16_t f32_to_f16(float x) {
    uint32_t u = 0;
    memcpy(&u, &x, sizeof(u));
    const uint32_t sign = (u >> 16) & 0x8000u;
    const uint32_t exp = (u >> 23) & 0xffu;
    const uint32_t mant = u & 0x7fffffu;
    if (exp == 0xffu) {
        return (uint16_t) (sign | 0x7c00u | (mant != 0u ? 0x0200u : 0u));
    }
    const int32_t half_exp = (int32_t) exp - 127 + 15;
    if (half_exp >= 0x1f) {
        return (uint16_t) (sign | 0x7c00u);
    }
    if (half_exp <= 0) {
        if (half_exp < -10) {
            return (uint16_t) sign;
        }
        uint32_t m = mant | 0x800000u;
        const uint32_t shift = (uint32_t) (14 - half_exp);
        uint32_t half_mant = m >> shift;
        if (((m >> (shift - 1u)) & 1u) != 0u) {
            half_mant++;
        }
        return (uint16_t) (sign | half_mant);
    }
    uint32_t half = sign | ((uint32_t) half_exp << 10) | (mant >> 13);
    if ((mant & 0x00001000u) != 0u) {
        half++;
    }
    return (uint16_t) half;
}

static float f16_to_f32(uint16_t h) {
    const uint32_t sign = ((uint32_t) h & 0x8000u) << 16;
    uint32_t exp = ((uint32_t) h >> 10) & 0x1fu;
    uint32_t mant = (uint32_t) h & 0x03ffu;
    uint32_t out = 0;
    if (exp == 0u) {
        if (mant == 0u) {
            out = sign;
        } else {
            exp = 1u;
            while ((mant & 0x0400u) == 0u) {
                mant <<= 1u;
                exp--;
            }
            mant &= 0x03ffu;
            out = sign | ((exp + 127u - 15u) << 23) | (mant << 13);
        }
    } else if (exp == 0x1fu) {
        out = sign | 0x7f800000u | (mant << 13);
    } else {
        out = sign | ((exp + 127u - 15u) << 23) | (mant << 13);
    }
    float f = 0.0f;
    memcpy(&f, &out, sizeof(f));
    return f;
}

static void encode_f16_array(size_t n, const float src[static n],
                             uint16_t dst[static n]) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = f32_to_f16(src[i]);
    }
}

static void decode_f16_array(size_t n, const uint16_t src[static n],
                             float dst[static n]) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = f16_to_f32(src[i]);
    }
}

static struct geist_tensor tensor_q4k_2d(struct geist_buffer *buf,
                                         size_t rows,
                                         size_t cols) {
    return (struct geist_tensor){
        .buffer = buf,
        .dtype = GEIST_DTYPE_Q4_K,
        .layout = GEIST_LAYOUT_BLOCK_QUANTIZED,
        .ndim = 2,
        .shape = {(int64_t) rows, (int64_t) cols},
        .stride = {0, 0},
    };
}

static struct geist_tensor tensor_q6k_2d(struct geist_buffer *buf,
                                         size_t rows,
                                         size_t cols) {
    return (struct geist_tensor){
        .buffer = buf,
        .dtype = GEIST_DTYPE_Q6_K,
        .layout = GEIST_LAYOUT_BLOCK_QUANTIZED,
        .ndim = 2,
        .shape = {(int64_t) rows, (int64_t) cols},
        .stride = {0, 0},
    };
}

static uint8_t q4k_value(size_t row,
                         size_t block,
                         size_t sub,
                         size_t idx) {
    return (uint8_t) ((row * 3u + block * 5u + sub * 7u + idx) & 15u);
}

static void pack_q4k_matrix(size_t n_in, size_t n_out, uint8_t *dst) {
    const size_t blocks_per_row = n_in / 256u;
    for (size_t row = 0; row < n_out; row++) {
        for (size_t block = 0; block < blocks_per_row; block++) {
            uint8_t *b = dst + (row * blocks_per_row + block) * 144u;
            memset(b, 0, 144u);
            b[0] = 0x00u;
            b[1] = 0x3cu; /* fp16(1.0) */
            b[2] = 0x00u;
            b[3] = 0x00u; /* fp16(0.0) */
            b[4] = 1u;
            b[5] = 1u;
            b[6] = 1u;
            b[7] = 1u;
            b[8] = 0u;
            b[9] = 0u;
            b[10] = 0u;
            b[11] = 0u;
            b[12] = 1u;
            b[13] = 1u;
            b[14] = 1u;
            b[15] = 1u;
            for (size_t pair = 0; pair < 4u; pair++) {
                for (size_t idx = 0; idx < 32u; idx++) {
                    const uint8_t lo =
                        q4k_value(row, block, pair * 2u, idx);
                    const uint8_t hi =
                        q4k_value(row, block, pair * 2u + 1u, idx);
                    b[16u + pair * 32u + idx] =
                        (uint8_t) (lo | (hi << 4u));
                }
            }
        }
    }
}

static void ref_q4k_matvec(const float *x,
                           size_t n_in,
                           size_t n_out,
                           float *y) {
    const size_t blocks_per_row = n_in / 256u;
    for (size_t row = 0; row < n_out; row++) {
        double acc = 0.0;
        for (size_t block = 0; block < blocks_per_row; block++) {
            for (size_t sub = 0; sub < 8u; sub++) {
                for (size_t idx = 0; idx < 32u; idx++) {
                    const size_t k = block * 256u + sub * 32u + idx;
                    acc += (double) x[k] *
                           (double) q4k_value(row, block, sub, idx);
                }
            }
        }
        y[row] = (float) acc;
    }
}

static int8_t q6k_value(size_t row, size_t stream, size_t idx) {
    return (int8_t) ((int) ((row * 5u + stream * 3u + idx) & 15u) - 8);
}

static void pack_q6k_matrix(size_t n_in, size_t n_out, uint8_t *dst) {
    const size_t blocks_per_row = n_in / 256u;
    for (size_t row = 0; row < n_out; row++) {
        for (size_t block = 0; block < blocks_per_row; block++) {
            uint8_t *b = dst + (row * blocks_per_row + block) * 210u;
            memset(b, 0, 210u);
            for (size_t i = 0; i < 16u; i++) {
                b[192u + i] = 1u;
            }
            b[208] = 0x00u;
            b[209] = 0x3cu; /* fp16(1.0) */
            for (size_t half_idx = 0; half_idx < 2u; half_idx++) {
                uint8_t *ql = b + half_idx * 64u;
                uint8_t *qh = b + 128u + half_idx * 32u;
                for (size_t idx = 0; idx < 32u; idx++) {
                    const uint8_t q0 =
                        (uint8_t) (q6k_value(row + block, 0u, idx) + 32);
                    const uint8_t q1 =
                        (uint8_t) (q6k_value(row + block, 1u, idx) + 32);
                    const uint8_t q2 =
                        (uint8_t) (q6k_value(row + block, 2u, idx) + 32);
                    const uint8_t q3 =
                        (uint8_t) (q6k_value(row + block, 3u, idx) + 32);
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
}

static void ref_q6k_matmul(const float *x,
                           size_t rows,
                           size_t n_in,
                           size_t n_out,
                           float *y) {
    const size_t blocks_per_row = n_in / 256u;
    for (size_t r = 0; r < rows; r++) {
        for (size_t row = 0; row < n_out; row++) {
            double acc = 0.0;
            for (size_t block = 0; block < blocks_per_row; block++) {
                for (size_t half_idx = 0; half_idx < 2u; half_idx++) {
                    for (size_t stream = 0; stream < 4u; stream++) {
                        for (size_t idx = 0; idx < 32u; idx++) {
                            const size_t k = block * 256u +
                                             half_idx * 128u +
                                             stream * 32u + idx;
                            acc += (double) x[r * n_in + k] *
                                   (double) q6k_value(row + block, stream, idx);
                        }
                    }
                }
            }
            y[r * n_out + row] = (float) acc;
        }
    }
}

static void ref_q4k_matmul(const float *x,
                           size_t rows,
                           size_t n_in,
                           size_t n_out,
                           float *y) {
    for (size_t r = 0; r < rows; r++) {
        ref_q4k_matvec(x + r * n_in, n_in, n_out, y + r * n_out);
    }
}

static void ref_rmsnorm_rows(const float *x,
                             const float *w,
                             size_t rows,
                             size_t cols,
                             float eps,
                             float *y) {
    for (size_t r = 0; r < rows; r++) {
        double ss = 0.0;
        for (size_t c = 0; c < cols; c++) {
            const double v = (double) x[r * cols + c];
            ss += v * v;
        }
        const float inv = 1.0f / sqrtf((float) (ss / (double) cols) + eps);
        for (size_t c = 0; c < cols; c++) {
            y[r * cols + c] = x[r * cols + c] * inv * w[c];
        }
    }
}

static float ref_gelu_tanh(float x) {
    constexpr float k0 = 0.7978845608028654f;
    constexpr float k1 = 0.044715f;
    return 0.5f * x * (1.0f + tanhf(k0 * (x + k1 * x * x * x)));
}

static void ref_f32_matmul(const float *x,
                           const float *w,
                           size_t rows,
                           size_t n_in,
                           size_t n_out,
                           float *y) {
    for (size_t r = 0; r < rows; r++) {
        for (size_t o = 0; o < n_out; o++) {
            float sum = 0.0f;
            for (size_t k = 0; k < n_in; k++) {
                sum += x[r * n_in + k] * w[o * n_in + k];
            }
            y[r * n_out + o] = sum;
        }
    }
}

static void ref_ple_f32_block(const float *hidden,
                              const float *ple,
                              const float *gate_w,
                              const float *proj_w,
                              const float *post_norm,
                              size_t rows,
                              size_t d_model,
                              size_t hpl,
                              float eps,
                              float *gate,
                              float *proj,
                              float *post,
                              float *out) {
    ref_f32_matmul(hidden, gate_w, rows, d_model, hpl, gate);
    for (size_t i = 0; i < rows * hpl; i++) {
        gate[i] = ref_gelu_tanh(gate[i]) * ple[i];
    }
    ref_f32_matmul(gate, proj_w, rows, hpl, d_model, proj);
    ref_rmsnorm_rows(proj, post_norm, rows, d_model, eps, post);
    for (size_t i = 0; i < rows * d_model; i++) {
        out[i] = hidden[i] + post[i];
    }
}

static void ref_ffn_geglu_q4k(const float *residual,
                              const float *ffn_norm,
                              const uint8_t *gate_w,
                              const uint8_t *up_w,
                              const uint8_t *down_w,
                              const float *post_norm,
                              size_t rows,
                              size_t d_model,
                              size_t inter,
                              float eps,
                              float *pre,
                              float *gate,
                              float *up,
                              float *mid,
                              float *ffn_out,
                              float *post,
                              float *out) {
    ref_rmsnorm_rows(residual, ffn_norm, rows, d_model, eps, pre);

    (void) gate_w;
    (void) up_w;
    (void) down_w;
    ref_q4k_matmul(pre, rows, d_model, inter, gate);
    ref_q4k_matmul(pre, rows, d_model, inter, up);
    for (size_t i = 0; i < rows * inter; i++) {
        mid[i] = ref_gelu_tanh(gate[i]) * up[i];
    }
    ref_q4k_matmul(mid, rows, inter, d_model, ffn_out);
    ref_rmsnorm_rows(ffn_out, post_norm, rows, d_model, eps, post);
    for (size_t i = 0; i < rows * d_model; i++) {
        out[i] = residual[i] + post[i];
    }
}

static void ref_ffn_geglu_q4k_q6down(const float *residual,
                                     const float *ffn_norm,
                                     const float *post_norm,
                                     size_t rows,
                                     size_t d_model,
                                     size_t inter,
                                     float eps,
                                     float *pre,
                                     float *gate,
                                     float *up,
                                     float *mid,
                                     float *ffn_out,
                                     float *post,
                                     float *out) {
    ref_rmsnorm_rows(residual, ffn_norm, rows, d_model, eps, pre);
    ref_q4k_matmul(pre, rows, d_model, inter, gate);
    ref_q4k_matmul(pre, rows, d_model, inter, up);
    for (size_t i = 0; i < rows * inter; i++) {
        mid[i] = ref_gelu_tanh(gate[i]) * up[i];
    }
    ref_q6k_matmul(mid, rows, inter, d_model, ffn_out);
    ref_rmsnorm_rows(ffn_out, post_norm, rows, d_model, eps, post);
    for (size_t i = 0; i < rows * d_model; i++) {
        out[i] = residual[i] + post[i];
    }
}

static void ref_rope_identity(size_t n, float *x) {
    (void) n;
    (void) x;
}

static void ref_rope_rows(float *x,
                          const float *cos,
                          const float *sin,
                          size_t rows,
                          size_t heads,
                          size_t head_dim) {
    const size_t half = head_dim / 2u;
    for (size_t r = 0; r < rows; r++) {
        for (size_t h = 0; h < heads; h++) {
            float *xh = x + (r * heads + h) * head_dim;
            for (size_t i = 0; i < half; i++) {
                const float x0 = xh[i];
                const float x1 = xh[i + half];
                const float co = cos[r * head_dim + i];
                const float si = sin[r * head_dim + i];
                xh[i] = x0 * co - x1 * si;
                xh[i + half] = x0 * si + x1 * co;
            }
        }
    }
}

static void ref_attention_mqa(const float *q,
                              const float *k,
                              const float *v,
                              size_t rows,
                              size_t kv_len,
                              size_t q_heads,
                              size_t kv_heads,
                              size_t head_dim,
                              size_t q_offset,
                              size_t sliding_window,
                              float *out) {
    for (size_t r = 0; r < rows; r++) {
        const size_t qpos = q_offset + r;
        const size_t start = sliding_window > 0u && qpos + 1u > sliding_window
                                 ? qpos + 1u - sliding_window
                                 : 0u;
        const size_t end = qpos < kv_len ? qpos : kv_len - 1u;
        for (size_t h = 0; h < q_heads; h++) {
            const size_t kvh = h / (q_heads / kv_heads);
            float max_score = -INFINITY;
            float scores[16];
            for (size_t s = start; s <= end; s++) {
                float dot = 0.0f;
                for (size_t i = 0; i < head_dim; i++) {
                    dot += q[(r * q_heads + h) * head_dim + i] *
                           k[(s * kv_heads + kvh) * head_dim + i];
                }
                scores[s - start] = dot;
                if (dot > max_score) {
                    max_score = dot;
                }
            }
            float sum = 0.0f;
            for (size_t s = start; s <= end; s++) {
                const float e = expf(scores[s - start] - max_score);
                scores[s - start] = e;
                sum += e;
            }
            for (size_t i = 0; i < head_dim; i++) {
                float acc = 0.0f;
                for (size_t s = start; s <= end; s++) {
                    acc += (scores[s - start] / sum) *
                           v[(s * kv_heads + kvh) * head_dim + i];
                }
                out[(r * q_heads + h) * head_dim + i] = acc;
            }
        }
    }
}

static void ref_attention_q4k(const float *residual,
                              const float *attn_norm,
                              const uint8_t *q_w,
                              const uint8_t *k_w,
                              const uint8_t *v_w,
                              const float *q_norm,
                              const float *k_norm,
                              const float *v_norm,
                              const uint8_t *o_w,
                              const float *post_norm,
                              float *k_cache,
                              float *v_cache,
                              size_t q_position,
                              size_t kv_len,
                              size_t d_model,
                              size_t q_heads,
                              size_t kv_heads,
                              size_t head_dim,
                              float eps,
                              float *normed,
                              float *q,
                              float *k,
                              float *v,
                              float *attn,
                              float *o,
                              float *post,
                              float *out,
                              float *appended_k,
                              float *appended_v) {
    (void) q_w;
    (void) k_w;
    (void) v_w;
    (void) o_w;
    const size_t q_out = q_heads * head_dim;
    const size_t kv_out = kv_heads * head_dim;
    ref_rmsnorm_rows(residual, attn_norm, 1, d_model, eps, normed);
    ref_q4k_matmul(normed, 1, d_model, q_out, q);
    ref_q4k_matmul(normed, 1, d_model, kv_out, k);
    ref_q4k_matmul(normed, 1, d_model, kv_out, v);
    ref_rmsnorm_rows(q, q_norm, q_heads, head_dim, eps, q);
    ref_rmsnorm_rows(k, k_norm, kv_heads, head_dim, eps, k);
    ref_rmsnorm_rows(v, v_norm, kv_heads, head_dim, eps, v);
    ref_rope_identity(q_out, q);
    ref_rope_identity(kv_out, k);
    memcpy(k_cache + q_position * kv_out, k, kv_out * sizeof(float));
    memcpy(v_cache + q_position * kv_out, v, kv_out * sizeof(float));
    memcpy(appended_k, k, kv_out * sizeof(float));
    memcpy(appended_v, v, kv_out * sizeof(float));

    for (size_t h = 0; h < q_heads; h++) {
        const size_t kv_h = h / (q_heads / kv_heads);
        const float *qh = q + h * head_dim;
        float scores[16];
        float max_score = -INFINITY;
        for (size_t s = 0; s < kv_len; s++) {
            const float *kh = k_cache + (s * kv_heads + kv_h) * head_dim;
            float dot = 0.0f;
            for (size_t i = 0; i < head_dim; i++) {
                dot += qh[i] * kh[i];
            }
            scores[s] = dot;
            if (dot > max_score) {
                max_score = dot;
            }
        }
        float sum = 0.0f;
        for (size_t s = 0; s < kv_len; s++) {
            scores[s] = expf(scores[s] - max_score);
            sum += scores[s];
        }
        for (size_t i = 0; i < head_dim; i++) {
            float acc = 0.0f;
            for (size_t s = 0; s < kv_len; s++) {
                const float *vh = v_cache + (s * kv_heads + kv_h) * head_dim;
                acc += (scores[s] / sum) * vh[i];
            }
            attn[h * head_dim + i] = acc;
        }
    }
    ref_q4k_matmul(attn, 1, q_out, d_model, o);
    ref_rmsnorm_rows(o, post_norm, 1, d_model, eps, post);
    for (size_t i = 0; i < d_model; i++) {
        out[i] = residual[i] + post[i];
    }
}

static int create_metal_or_skip(struct geist_backend **out) {
    *out = nullptr;
    enum geist_status s = geist_backend_create("metal", nullptr, nullptr, out);
#if defined(GEIST_BACKEND_METAL) && GEIST_BACKEND_METAL
    if (s == GEIST_E_UNSUPPORTED) {
        printf("SKIP: metal runtime unavailable: %s\n",
               geist_last_create_error());
        return GEIST_TEST_SKIP;
    }
    if (s != GEIST_OK) {
        fprintf(stderr, "FAIL: metal create failed: %s: %s\n",
                geist_status_to_string(s), geist_last_create_error());
        return GEIST_TEST_FAIL;
    }
    return GEIST_TEST_PASS;
#else
    int fails = 0;
    fails += check_status(s, GEIST_E_NOT_FOUND,
                          "metal is absent from non-metal builds");
    fails += check(*out == nullptr,
                   "failed metal create leaves output null");
    return fails == 0 ? GEIST_TEST_SKIP : GEIST_TEST_FAIL;
#endif
}

int main(void) {
    char pack_cache_dir[] = "/tmp/geist-metal-pack-cache-XXXXXX";
    char *pack_cache_path = mkdtemp(pack_cache_dir);
    if (pack_cache_path != nullptr) {
        (void) setenv("GEIST_METAL_PACK_CACHE_DIR", pack_cache_path, 1);
    }
    (void) setenv("GEIST_METAL_Q4K_NT4", "1", 1);
    (void) setenv("GEIST_METAL_Q4K_NT8", "1", 1);
    (void) setenv("GEIST_METAL_Q4K_NT4_LARGE", "1", 1);
    (void) unsetenv("GEIST_METAL_Q4K_MM_SG");
    (void) unsetenv("GEIST_METAL_Q4K_MM_SG_UNSAFE");
    (void) setenv("GEIST_METAL_Q6K_NT4", "1", 1);
    (void) setenv("GEIST_METAL_ATTENTION_FUSED_QK_NT4", "1", 1);
    (void) setenv("GEIST_METAL_PLE_BLOCK", "1", 1);

    struct geist_backend *be = nullptr;
    int create_result = create_metal_or_skip(&be);
    if (create_result != GEIST_TEST_PASS) {
        if (pack_cache_path != nullptr) {
            (void) count_and_remove_cache_files(pack_cache_path);
            (void) rmdir(pack_cache_path);
        }
        return create_result;
    }

    int fails = 0;
    const struct geist_backend_vtbl *v = be->desc->vtbl;
    fails += check(v->matvec_f32_dense != nullptr,
                   "metal F32 dense matvec present");
    fails += check(v->matmul_f32_dense != nullptr,
                   "metal F32 dense matmul present");
    fails += check(v->rmsnorm != nullptr, "metal rmsnorm present");
    fails += check(v->add != nullptr, "metal add present");
    fails += check(v->mul != nullptr, "metal mul present");
    fails += check(v->scale_f32 != nullptr, "metal scale_f32 present");
    fails += check(v->gelu_tanh != nullptr, "metal gelu_tanh present");
    fails += check(v->gelu_tanh_mul != nullptr,
                   "metal gelu_tanh_mul present");
    fails += check(v->rope_apply != nullptr, "metal rope_apply present");
    fails += check(v->attention != nullptr, "metal attention present");
    fails += check(v->argmax_f32 != nullptr, "metal argmax_f32 present");
    fails += check(v->argmax_f32_batch != nullptr,
                   "metal argmax_f32_batch present");
    fails += check(v->greedy_head != nullptr, "metal greedy_head present");
    fails += check(v->greedy_head_batch != nullptr,
                   "metal greedy_head_batch present");
    fails += check(v->matvec_q4k != nullptr, "metal Q4_K matvec present");
    fails += check(v->matmul_q4k != nullptr, "metal Q4_K matmul present");
    fails += check(v->matvec_q6k != nullptr, "metal Q6_K matvec present");
    fails += check(v->matmul_q6k != nullptr, "metal Q6_K matmul present");
    fails += check(v->embedding_lookup_scaled != nullptr,
                   "metal scaled embedding lookup present");
    fails += check(v->ffn_geglu_block != nullptr,
                   "metal GEGLU FFN block present");
    fails += check(v->ple_block != nullptr,
                   "metal PLE block present");
    fails += check(v->attention_block != nullptr,
                   "metal attention block present");
    fails += check(v->command_sequence_begin != nullptr,
                   "metal command sequence begin present");
    fails += check(v->command_sequence_end != nullptr,
                   "metal command sequence end present");
    fails += check(v->command_sequence_read_token != nullptr,
                   "metal command sequence token read present");
    fails += check(v->command_sequence_read_tokens != nullptr,
                   "metal command sequence token batch read present");
    fails += check(v->prepare_weight_layout != nullptr,
                   "metal prepare_weight_layout present");
    fails += check(v->prepare_weight_layout_from_host != nullptr,
                   "metal prepare_weight_layout_from_host present");
    if (fails != 0) {
        geist_backend_destroy(be);
        return GEIST_TEST_FAIL;
    }

    struct geist_op_support_query q = {
        .op = GEIST_OP_LINEAR,
        .input_count = 2,
        .inputs = {
            {.dtype = GEIST_DTYPE_F32, .layout = GEIST_LAYOUT_DENSE},
            {.dtype = GEIST_DTYPE_Q4_K,
             .layout = GEIST_LAYOUT_BLOCK_QUANTIZED},
        },
        .output_count = 1,
        .outputs = {{.dtype = GEIST_DTYPE_F32,
                     .layout = GEIST_LAYOUT_DENSE}},
    };
    fails += check(geist_backend_supports_op(be, &q) == GEIST_SUPPORT_NATIVE,
                   "metal advertises native Q4_K decode linear");
    q.inputs[1].dtype = GEIST_DTYPE_Q6_K;
    fails += check(geist_backend_supports_op(be, &q) == GEIST_SUPPORT_NATIVE,
                   "metal advertises native Q6_K decode linear");
    q.inputs[1].dtype = GEIST_DTYPE_Q4_K;

    {
        const float logits[] = {-1.0f, 0.5f, 3.25f, 2.0f, 3.25f, -0.25f};
        struct geist_buffer *lb = nullptr;
        enum geist_status as = v->buffer_create(
            be, sizeof(logits), GEIST_BUFFER_ACTIVATION,
            GEIST_MEMORY_DEVICE, &lb);
        if (as == GEIST_OK) {
            as = v->buffer_upload(lb, sizeof(logits),
                                  (const uint8_t *) logits);
        }
        struct geist_tensor tl =
            tensor_f32_1d(lb, sizeof(logits) / sizeof(logits[0]));
        geist_token_t token = -1;
        if (as == GEIST_OK) {
            as = v->argmax_f32(be, &tl, &token);
        }
        fails += check_status(as, GEIST_OK, "Metal argmax_f32 runs");
        if (as == GEIST_OK) {
            fails += check(token == 2,
                           "Metal argmax_f32 returns lowest max index");
        }
        if (lb != nullptr) { v->buffer_destroy(be, lb); }
    }

    {
        constexpr size_t head_d = 4;
        constexpr size_t head_vocab = 3;
        const float hidden[head_d] = {1.0f, 2.0f, -1.0f, 0.5f};
        const float norm[head_d] = {1.0f, 0.75f, 1.25f, 1.5f};
        const float weight[head_vocab * head_d] = {
            -0.4f, 0.1f, 0.2f, 0.0f,
             0.1f, 0.2f, 0.3f, 0.4f,
             0.2f, 1.0f, -0.2f, 0.8f,
        };
        float normed_ref[head_d] = {0};
        float logits_ref[head_vocab] = {0};
        ref_rmsnorm_rows(hidden, norm, 1, head_d, 1.0e-5f, normed_ref);
        for (size_t row = 0; row < head_vocab; row++) {
            double acc = 0.0;
            for (size_t col = 0; col < head_d; col++) {
                acc += (double) weight[row * head_d + col] *
                       (double) normed_ref[col];
            }
            logits_ref[row] = (float) acc;
        }
        geist_token_t expected = 0;
        for (size_t i = 1; i < head_vocab; i++) {
            if (logits_ref[i] > logits_ref[(size_t) expected]) {
                expected = (geist_token_t) i;
            }
        }

        struct geist_buffer *hb = nullptr;
        struct geist_buffer *nb = nullptr;
        struct geist_buffer *wb = nullptr;
        struct geist_buffer *normed_b = nullptr;
        struct geist_buffer *logits_b = nullptr;
        enum geist_status hs = v->buffer_create(
            be, sizeof(hidden), GEIST_BUFFER_ACTIVATION,
            GEIST_MEMORY_DEVICE, &hb);
        if (hs == GEIST_OK) {
            hs = v->buffer_create(be, sizeof(norm), GEIST_BUFFER_WEIGHT,
                                  GEIST_MEMORY_DEVICE, &nb);
        }
        if (hs == GEIST_OK) {
            hs = v->buffer_create(be, sizeof(weight), GEIST_BUFFER_WEIGHT,
                                  GEIST_MEMORY_DEVICE, &wb);
        }
        if (hs == GEIST_OK) {
            hs = v->buffer_create(be, sizeof(normed_ref),
                                  GEIST_BUFFER_ACTIVATION,
                                  GEIST_MEMORY_DEVICE, &normed_b);
        }
        if (hs == GEIST_OK) {
            hs = v->buffer_create(be, sizeof(logits_ref),
                                  GEIST_BUFFER_ACTIVATION,
                                  GEIST_MEMORY_DEVICE, &logits_b);
        }
        if (hs == GEIST_OK) {
            hs = v->buffer_upload(hb, sizeof(hidden),
                                  (const uint8_t *) hidden);
        }
        if (hs == GEIST_OK) {
            hs = v->buffer_upload(nb, sizeof(norm),
                                  (const uint8_t *) norm);
        }
        if (hs == GEIST_OK) {
            hs = v->buffer_upload(wb, sizeof(weight),
                                  (const uint8_t *) weight);
        }

        struct geist_tensor th = tensor_f32_1d(hb, head_d);
        struct geist_tensor tn = tensor_f32_1d(nb, head_d);
        struct geist_tensor tw = tensor_f32_2d(wb, head_vocab, head_d);
        struct geist_tensor ts = tensor_f32_1d(normed_b, head_d);
        struct geist_tensor tl = tensor_f32_1d(logits_b, head_vocab);
        const struct geist_backend_greedy_head head = {
            .struct_size = sizeof(head),
            .d_model = head_d,
            .vocab_size = head_vocab,
            .token_output_offset = 0,
            .eps = 1.0e-5f,
            .hidden = &th,
            .norm_weight = &tn,
            .lm_head_weight = &tw,
            .normed_scratch = &ts,
            .logits = &tl,
        };
        geist_token_t token = -1;
        if (hs == GEIST_OK) {
            hs = v->greedy_head(be, &head, &token);
        }
        fails += check_status(hs, GEIST_OK, "Metal greedy_head runs");
        if (hs == GEIST_OK) {
            fails += check(token == expected,
                           "Metal greedy_head returns argmax");
        }

        int capture_token = 0;
        geist_token_t captured_direct = -2;
        geist_token_t captured_read = -1;
        if (hs == GEIST_OK) {
            hs = v->command_sequence_begin(
                be, GEIST_COMMAND_SEQUENCE_VERIFY_GREEDY, &capture_token);
        }
        if (hs == GEIST_OK) {
            hs = v->greedy_head(be, &head, &captured_direct);
        }
        if (capture_token != 0) {
            enum geist_status end_s =
                v->command_sequence_end(be, capture_token,
                                        hs == GEIST_OK);
            if (hs == GEIST_OK) {
                hs = end_s;
            }
        }
        if (hs == GEIST_OK) {
            hs = v->command_sequence_read_token(be, &captured_read);
        }
        fails += check_status(hs, GEIST_OK,
                              "Metal captured greedy_head reads token");
        if (hs == GEIST_OK) {
            fails += check(captured_direct == -1,
                           "captured greedy_head defers direct token");
            fails += check(captured_read == expected,
                           "captured greedy_head token matches argmax");
        }

        {
            constexpr size_t replay_vocab = 3;
            constexpr size_t replay_d = 4;
            const float replay_embed[replay_vocab * replay_d] = {
                1.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 2.0f, 0.0f,
                0.0f, 1.0f, 0.0f, 0.0f,
            };
            const float replay_norm[replay_d] = {
                1.0f, 1.0f, 1.0f, 1.0f,
            };
            const float replay_head_w[replay_vocab * replay_d] = {
                0.1f, 0.0f, 0.0f, 0.0f,
                2.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 3.0f, 0.0f,
            };
            struct geist_buffer *replay_eb = nullptr;
            struct geist_buffer *replay_hb = nullptr;
            struct geist_buffer *replay_nb = nullptr;
            struct geist_buffer *replay_wb = nullptr;
            struct geist_buffer *replay_sb = nullptr;
            struct geist_buffer *replay_lb = nullptr;
            enum geist_status rs = v->buffer_create(
                be, sizeof(replay_embed), GEIST_BUFFER_WEIGHT,
                GEIST_MEMORY_DEVICE, &replay_eb);
            if (rs == GEIST_OK) {
                rs = v->buffer_create(be, replay_d * sizeof(float),
                                      GEIST_BUFFER_ACTIVATION,
                                      GEIST_MEMORY_DEVICE, &replay_hb);
            }
            if (rs == GEIST_OK) {
                rs = v->buffer_create(be, sizeof(replay_norm),
                                      GEIST_BUFFER_WEIGHT,
                                      GEIST_MEMORY_DEVICE, &replay_nb);
            }
            if (rs == GEIST_OK) {
                rs = v->buffer_create(be, sizeof(replay_head_w),
                                      GEIST_BUFFER_WEIGHT,
                                      GEIST_MEMORY_DEVICE, &replay_wb);
            }
            if (rs == GEIST_OK) {
                rs = v->buffer_create(be, replay_d * sizeof(float),
                                      GEIST_BUFFER_ACTIVATION,
                                      GEIST_MEMORY_DEVICE, &replay_sb);
            }
            if (rs == GEIST_OK) {
                rs = v->buffer_create(be, replay_vocab * sizeof(float),
                                      GEIST_BUFFER_ACTIVATION,
                                      GEIST_MEMORY_DEVICE, &replay_lb);
            }
            if (rs == GEIST_OK) {
                rs = v->buffer_upload(replay_eb, sizeof(replay_embed),
                                      (const uint8_t *) replay_embed);
            }
            if (rs == GEIST_OK) {
                rs = v->buffer_upload(replay_nb, sizeof(replay_norm),
                                      (const uint8_t *) replay_norm);
            }
            if (rs == GEIST_OK) {
                rs = v->buffer_upload(replay_wb, sizeof(replay_head_w),
                                      (const uint8_t *) replay_head_w);
            }

            struct geist_tensor replay_te =
                tensor_f32_2d(replay_eb, replay_vocab, replay_d);
            struct geist_tensor replay_th =
                tensor_f32_1d(replay_hb, replay_d);
            struct geist_tensor replay_tn =
                tensor_f32_1d(replay_nb, replay_d);
            struct geist_tensor replay_tw =
                tensor_f32_2d(replay_wb, replay_vocab, replay_d);
            struct geist_tensor replay_ts =
                tensor_f32_1d(replay_sb, replay_d);
            struct geist_tensor replay_tl =
                tensor_f32_1d(replay_lb, replay_vocab);
            const struct geist_backend_greedy_head replay_head = {
                .struct_size = sizeof(replay_head),
                .d_model = replay_d,
                .vocab_size = replay_vocab,
                .token_output_offset = 0,
                .eps = 1.0e-5f,
                .hidden = &replay_th,
                .norm_weight = &replay_tn,
                .lm_head_weight = &replay_tw,
                .normed_scratch = &replay_ts,
                .logits = &replay_tl,
            };

            int replay_capture = 0;
            geist_token_t replay_direct = -2;
            geist_token_t replay_first = -1;
            geist_token_t replay_second = -1;
            if (rs == GEIST_OK) {
                rs = v->command_sequence_begin(
                    be, GEIST_COMMAND_SEQUENCE_DECODE_GREEDY_STEP,
                    &replay_capture);
            }
            if (rs == GEIST_OK) {
                rs = v->embedding_lookup_scaled(be, &replay_te, 0, 1.0f,
                                                &replay_th);
            }
            if (rs == GEIST_OK) {
                rs = v->greedy_head(be, &replay_head, &replay_direct);
            }
            if (replay_capture != 0) {
                enum geist_status end_s =
                    v->command_sequence_end(be, replay_capture,
                                            rs == GEIST_OK);
                if (rs == GEIST_OK) {
                    rs = end_s;
                }
            }
            if (rs == GEIST_OK) {
                rs = v->command_sequence_read_token(be, &replay_first);
            }
            if (rs == GEIST_OK) {
                rs = v->command_sequence_replay_decode_greedy_step(
                    be, 1, 0, &replay_second);
            }
            fails += check_status(rs, GEIST_OK,
                                  "Metal decode greedy replay runs");
            if (rs == GEIST_OK) {
                fails += check(replay_direct == -1,
                               "decode replay capture defers direct token");
                fails += check(replay_first == 1,
                               "decode replay first capture token matches");
                fails += check(replay_second == 2,
                               "decode replay patches token id");
            }

            if (replay_eb != nullptr) { v->buffer_destroy(be, replay_eb); }
            if (replay_hb != nullptr) { v->buffer_destroy(be, replay_hb); }
            if (replay_nb != nullptr) { v->buffer_destroy(be, replay_nb); }
            if (replay_wb != nullptr) { v->buffer_destroy(be, replay_wb); }
            if (replay_sb != nullptr) { v->buffer_destroy(be, replay_sb); }
            if (replay_lb != nullptr) { v->buffer_destroy(be, replay_lb); }
        }

        struct geist_backend_greedy_head bad_head = head;
        bad_head.token_output_offset = 1;
        geist_token_t bad_token = -1;
        enum geist_status bad_s =
            v->greedy_head(be, &bad_head, &bad_token);
        fails += check_status(bad_s, GEIST_E_INVALID_ARG,
                              "Metal greedy_head rejects token offset");

        const float hidden_batch[2 * head_d] = {
            1.0f, 2.0f, -1.0f, 0.5f,
            -0.25f, 0.75f, 1.5f, -2.0f,
        };
        float normed_batch_ref[2 * head_d] = {0};
        float logits_batch_ref[2 * head_vocab] = {0};
        geist_token_t expected_batch[2] = {-1, -1};
        ref_rmsnorm_rows(hidden_batch, norm, 2, head_d, 1.0e-5f,
                         normed_batch_ref);
        for (size_t row = 0; row < 2; row++) {
            for (size_t vocab = 0; vocab < head_vocab; vocab++) {
                double acc = 0.0;
                for (size_t col = 0; col < head_d; col++) {
                    acc +=
                        (double) weight[vocab * head_d + col] *
                        (double) normed_batch_ref[row * head_d + col];
                }
                logits_batch_ref[row * head_vocab + vocab] = (float) acc;
            }
            expected_batch[row] = 0;
            for (size_t vocab = 1; vocab < head_vocab; vocab++) {
                if (logits_batch_ref[row * head_vocab + vocab] >
                    logits_batch_ref[row * head_vocab +
                                     (size_t) expected_batch[row]]) {
                    expected_batch[row] = (geist_token_t) vocab;
                }
            }
        }

        struct geist_buffer *hb_batch = nullptr;
        struct geist_buffer *normed_batch_b = nullptr;
        struct geist_buffer *logits_batch_b = nullptr;
        if (hs == GEIST_OK) {
            hs = v->buffer_create(be, sizeof(hidden_batch),
                                  GEIST_BUFFER_ACTIVATION,
                                  GEIST_MEMORY_DEVICE, &hb_batch);
        }
        if (hs == GEIST_OK) {
            hs = v->buffer_create(be, sizeof(normed_batch_ref),
                                  GEIST_BUFFER_ACTIVATION,
                                  GEIST_MEMORY_DEVICE, &normed_batch_b);
        }
        if (hs == GEIST_OK) {
            hs = v->buffer_create(be, sizeof(logits_batch_ref),
                                  GEIST_BUFFER_ACTIVATION,
                                  GEIST_MEMORY_DEVICE, &logits_batch_b);
        }
        if (hs == GEIST_OK) {
            hs = v->buffer_upload(hb_batch, sizeof(hidden_batch),
                                  (const uint8_t *) hidden_batch);
        }

        struct geist_tensor th_batch =
            tensor_f32_2d(hb_batch, 2, head_d);
        struct geist_tensor ts_batch =
            tensor_f32_2d(normed_batch_b, 2, head_d);
        struct geist_tensor tl_batch =
            tensor_f32_2d(logits_batch_b, 2, head_vocab);
        const struct geist_backend_greedy_head_batch head_batch = {
            .struct_size = sizeof(head_batch),
            .d_model = head_d,
            .vocab_size = head_vocab,
            .row_count = 2,
            .token_output_offset = 0,
            .eps = 1.0e-5f,
            .hidden = &th_batch,
            .norm_weight = &tn,
            .lm_head_weight = &tw,
            .normed_scratch = &ts_batch,
            .logits = &tl_batch,
        };
        geist_token_t batch_tokens[2] = {-1, -1};
        if (hs == GEIST_OK) {
            hs = v->greedy_head_batch(be, &head_batch, batch_tokens);
        }
        fails += check_status(hs, GEIST_OK,
                              "Metal greedy_head_batch runs");
        if (hs == GEIST_OK) {
            fails += check(batch_tokens[0] == expected_batch[0] &&
                           batch_tokens[1] == expected_batch[1],
                           "Metal greedy_head_batch returns argmaxes");
        }

        int batch_capture_token = 0;
        geist_token_t batch_captured_direct[2] = {-2, -2};
        geist_token_t batch_captured_read[2] = {-1, -1};
        if (hs == GEIST_OK) {
            hs = v->command_sequence_begin(
                be, GEIST_COMMAND_SEQUENCE_VERIFY_GREEDY,
                &batch_capture_token);
        }
        if (hs == GEIST_OK) {
            hs = v->greedy_head_batch(be, &head_batch,
                                      batch_captured_direct);
        }
        if (batch_capture_token != 0) {
            enum geist_status end_s =
                v->command_sequence_end(be, batch_capture_token,
                                        hs == GEIST_OK);
            if (hs == GEIST_OK) {
                hs = end_s;
            }
        }
        if (hs == GEIST_OK) {
            hs = v->command_sequence_read_tokens(
                be, 2, batch_captured_read);
        }
        fails += check_status(hs, GEIST_OK,
                              "Metal captured greedy_head_batch reads tokens");
        if (hs == GEIST_OK) {
            fails += check(batch_captured_direct[0] == -1 &&
                           batch_captured_direct[1] == -1,
                           "captured greedy_head_batch defers direct tokens");
            fails += check(batch_captured_read[0] == expected_batch[0] &&
                           batch_captured_read[1] == expected_batch[1],
                           "captured greedy_head_batch tokens match argmaxes");
        }

        if (hb != nullptr) { v->buffer_destroy(be, hb); }
        if (nb != nullptr) { v->buffer_destroy(be, nb); }
        if (wb != nullptr) { v->buffer_destroy(be, wb); }
        if (normed_b != nullptr) { v->buffer_destroy(be, normed_b); }
        if (logits_b != nullptr) { v->buffer_destroy(be, logits_b); }
        if (hb_batch != nullptr) { v->buffer_destroy(be, hb_batch); }
        if (normed_batch_b != nullptr) {
            v->buffer_destroy(be, normed_batch_b);
        }
        if (logits_batch_b != nullptr) {
            v->buffer_destroy(be, logits_batch_b);
        }
    }

    {
        constexpr size_t rope_rows = 2;
        constexpr size_t rope_heads = 2;
        constexpr size_t rope_hd = 4;
        float rope_x[rope_rows * rope_heads * rope_hd];
        float rope_ref[rope_rows * rope_heads * rope_hd];
        float rope_cos[rope_rows * rope_hd];
        float rope_sin[rope_rows * rope_hd];
        float rope_got[rope_rows * rope_heads * rope_hd];
        for (size_t i = 0; i < rope_rows * rope_heads * rope_hd; i++) {
            rope_x[i] = -0.7f + 0.13f * (float) i;
            rope_ref[i] = rope_x[i];
        }
        for (size_t i = 0; i < rope_rows * rope_hd; i++) {
            const float angle = 0.05f + 0.17f * (float) i;
            rope_cos[i] = cosf(angle);
            rope_sin[i] = sinf(angle);
        }
        ref_rope_rows(rope_ref, rope_cos, rope_sin, rope_rows, rope_heads,
                      rope_hd);

        struct geist_buffer *xb = nullptr;
        struct geist_buffer *cb = nullptr;
        struct geist_buffer *sb = nullptr;
        enum geist_status rs = v->buffer_create(
            be, sizeof(rope_x), GEIST_BUFFER_ACTIVATION,
            GEIST_MEMORY_DEVICE, &xb);
        if (rs == GEIST_OK) {
            rs = v->buffer_create(be, sizeof(rope_cos), GEIST_BUFFER_WEIGHT,
                                  GEIST_MEMORY_DEVICE, &cb);
        }
        if (rs == GEIST_OK) {
            rs = v->buffer_create(be, sizeof(rope_sin), GEIST_BUFFER_WEIGHT,
                                  GEIST_MEMORY_DEVICE, &sb);
        }
        if (rs == GEIST_OK) {
            rs = v->buffer_upload(xb, sizeof(rope_x),
                                  (const uint8_t *) rope_x);
        }
        if (rs == GEIST_OK) {
            rs = v->buffer_upload(cb, sizeof(rope_cos),
                                  (const uint8_t *) rope_cos);
        }
        if (rs == GEIST_OK) {
            rs = v->buffer_upload(sb, sizeof(rope_sin),
                                  (const uint8_t *) rope_sin);
        }
        struct geist_tensor tx =
            tensor_f32_3d(xb, rope_rows, rope_heads, rope_hd);
        struct geist_tensor tc = tensor_f32_2d(cb, rope_rows, rope_hd);
        struct geist_tensor ts = tensor_f32_2d(sb, rope_rows, rope_hd);
        if (rs == GEIST_OK) {
            rs = v->rope_apply(be, &tx, &tc, &ts);
        }
        if (rs == GEIST_OK) {
            rs = v->buffer_download(sizeof(rope_got),
                                    (uint8_t *) rope_got, xb);
        }
        fails += check_status(rs, GEIST_OK, "Metal rope_apply runs");
        if (rs == GEIST_OK) {
            const ptrdiff_t bad = geist_fp32_close_array(
                rope_got, rope_ref, rope_rows * rope_heads * rope_hd,
                2e-5f, 2e-6f);
            if (bad >= 0) {
                fprintf(stderr,
                        "FAIL: rope[%td]: got %.7f want %.7f\n",
                        bad, (double) rope_got[bad],
                        (double) rope_ref[bad]);
                fails++;
            }
        }
        if (xb != nullptr) { v->buffer_destroy(be, xb); }
        if (cb != nullptr) { v->buffer_destroy(be, cb); }
        if (sb != nullptr) { v->buffer_destroy(be, sb); }
    }

    {
        constexpr size_t attn_rows = 2;
        constexpr size_t attn_kv_len = 4;
        constexpr size_t attn_q_heads = 2;
        constexpr size_t attn_kv_heads = 1;
        constexpr size_t attn_hd = 4;
        constexpr size_t attn_q_offset = 1;
        float qv[attn_rows * attn_q_heads * attn_hd];
        float kv[attn_kv_len * attn_kv_heads * attn_hd];
        float vv[attn_kv_len * attn_kv_heads * attn_hd];
        uint16_t kv_f16[attn_kv_len * attn_kv_heads * attn_hd];
        uint16_t vv_f16[attn_kv_len * attn_kv_heads * attn_hd];
        float kv_f16_ref[attn_kv_len * attn_kv_heads * attn_hd];
        float vv_f16_ref[attn_kv_len * attn_kv_heads * attn_hd];
        float attn_ref[attn_rows * attn_q_heads * attn_hd];
        float attn_f16_ref[attn_rows * attn_q_heads * attn_hd];
        float attn_got[attn_rows * attn_q_heads * attn_hd];
        float attn_f16_got[attn_rows * attn_q_heads * attn_hd];
        for (size_t i = 0; i < attn_rows * attn_q_heads * attn_hd; i++) {
            qv[i] = sinf(0.19f * (float) i) * 0.7f;
        }
        for (size_t i = 0; i < attn_kv_len * attn_kv_heads * attn_hd; i++) {
            kv[i] = cosf(0.11f * (float) i) * 0.4f;
            vv[i] = -0.3f + 0.07f * (float) ((i * 3u) % 11u);
        }
        encode_f16_array(attn_kv_len * attn_kv_heads * attn_hd, kv,
                         kv_f16);
        encode_f16_array(attn_kv_len * attn_kv_heads * attn_hd, vv,
                         vv_f16);
        decode_f16_array(attn_kv_len * attn_kv_heads * attn_hd, kv_f16,
                         kv_f16_ref);
        decode_f16_array(attn_kv_len * attn_kv_heads * attn_hd, vv_f16,
                         vv_f16_ref);
        ref_attention_mqa(qv, kv, vv, attn_rows, attn_kv_len,
                          attn_q_heads, attn_kv_heads, attn_hd,
                          attn_q_offset, 0, attn_ref);
        ref_attention_mqa(qv, kv_f16_ref, vv_f16_ref, attn_rows,
                          attn_kv_len, attn_q_heads, attn_kv_heads,
                          attn_hd, attn_q_offset, 2, attn_f16_ref);

        struct geist_buffer *qb = nullptr;
        struct geist_buffer *kb = nullptr;
        struct geist_buffer *vb = nullptr;
        struct geist_buffer *ob = nullptr;
        struct geist_buffer *kb_f16 = nullptr;
        struct geist_buffer *vb_f16 = nullptr;
        struct geist_buffer *ob_f16 = nullptr;
        enum geist_status as = v->buffer_create(
            be, sizeof(qv), GEIST_BUFFER_ACTIVATION,
            GEIST_MEMORY_DEVICE, &qb);
        if (as == GEIST_OK) {
            as = v->buffer_create(be, sizeof(kv), GEIST_BUFFER_ACTIVATION,
                                  GEIST_MEMORY_DEVICE, &kb);
        }
        if (as == GEIST_OK) {
            as = v->buffer_create(be, sizeof(vv), GEIST_BUFFER_ACTIVATION,
                                  GEIST_MEMORY_DEVICE, &vb);
        }
        if (as == GEIST_OK) {
            as = v->buffer_create(be, sizeof(attn_got),
                                  GEIST_BUFFER_ACTIVATION,
                                  GEIST_MEMORY_DEVICE, &ob);
        }
        if (as == GEIST_OK) {
            as = v->buffer_create(be, sizeof(kv_f16), GEIST_BUFFER_KV_CACHE,
                                  GEIST_MEMORY_DEVICE, &kb_f16);
        }
        if (as == GEIST_OK) {
            as = v->buffer_create(be, sizeof(vv_f16), GEIST_BUFFER_KV_CACHE,
                                  GEIST_MEMORY_DEVICE, &vb_f16);
        }
        if (as == GEIST_OK) {
            as = v->buffer_create(be, sizeof(attn_f16_got),
                                  GEIST_BUFFER_ACTIVATION,
                                  GEIST_MEMORY_DEVICE, &ob_f16);
        }
        if (as == GEIST_OK) {
            as = v->buffer_upload(qb, sizeof(qv), (const uint8_t *) qv);
        }
        if (as == GEIST_OK) {
            as = v->buffer_upload(kb, sizeof(kv), (const uint8_t *) kv);
        }
        if (as == GEIST_OK) {
            as = v->buffer_upload(vb, sizeof(vv), (const uint8_t *) vv);
        }
        if (as == GEIST_OK) {
            as = v->buffer_upload(kb_f16, sizeof(kv_f16),
                                  (const uint8_t *) kv_f16);
        }
        if (as == GEIST_OK) {
            as = v->buffer_upload(vb_f16, sizeof(vv_f16),
                                  (const uint8_t *) vv_f16);
        }
        struct geist_tensor tq =
            tensor_f32_3d(qb, attn_rows, attn_q_heads, attn_hd);
        struct geist_tensor tk =
            tensor_f32_3d(kb, attn_kv_len, attn_kv_heads, attn_hd);
        struct geist_tensor tv =
            tensor_f32_3d(vb, attn_kv_len, attn_kv_heads, attn_hd);
        struct geist_tensor to =
            tensor_f32_3d(ob, attn_rows, attn_q_heads, attn_hd);
        struct geist_tensor tk_f16 =
            tensor_f16_3d(kb_f16, attn_kv_len, attn_kv_heads, attn_hd);
        struct geist_tensor tv_f16 =
            tensor_f16_3d(vb_f16, attn_kv_len, attn_kv_heads, attn_hd);
        struct geist_tensor to_f16 =
            tensor_f32_3d(ob_f16, attn_rows, attn_q_heads, attn_hd);
        if (as == GEIST_OK) {
            as = v->attention(be, &tq, &tk, &tv, attn_q_offset, 0, &to);
        }
        if (as == GEIST_OK) {
            as = v->buffer_download(sizeof(attn_got),
                                    (uint8_t *) attn_got, ob);
        }
        fails += check_status(as, GEIST_OK, "Metal attention runs");
        if (as == GEIST_OK) {
            const ptrdiff_t bad = geist_fp32_close_array(
                attn_got, attn_ref, attn_rows * attn_q_heads * attn_hd,
                2e-5f, 2e-6f);
            if (bad >= 0) {
                fprintf(stderr,
                        "FAIL: attention[%td]: got %.7f want %.7f\n",
                        bad, (double) attn_got[bad],
                        (double) attn_ref[bad]);
                fails++;
            }
        }
        if (as == GEIST_OK) {
            as = v->attention(be, &tq, &tk_f16, &tv_f16, attn_q_offset, 2,
                              &to_f16);
        }
        if (as == GEIST_OK) {
            as = v->buffer_download(sizeof(attn_f16_got),
                                    (uint8_t *) attn_f16_got, ob_f16);
        }
        fails += check_status(as, GEIST_OK,
                              "Metal F16 KV batched causal attention runs");
        if (as == GEIST_OK) {
            const ptrdiff_t bad = geist_fp32_close_array(
                attn_f16_got, attn_f16_ref,
                attn_rows * attn_q_heads * attn_hd, 2e-4f, 2e-5f);
            if (bad >= 0) {
                fprintf(stderr,
                        "FAIL: F16 KV batched attention[%td]: got %.7f "
                        "want %.7f\n",
                        bad, (double) attn_f16_got[bad],
                        (double) attn_f16_ref[bad]);
                fails++;
            }
        }
        if (qb != nullptr) { v->buffer_destroy(be, qb); }
        if (kb != nullptr) { v->buffer_destroy(be, kb); }
        if (vb != nullptr) { v->buffer_destroy(be, vb); }
        if (ob != nullptr) { v->buffer_destroy(be, ob); }
        if (kb_f16 != nullptr) { v->buffer_destroy(be, kb_f16); }
        if (vb_f16 != nullptr) { v->buffer_destroy(be, vb_f16); }
        if (ob_f16 != nullptr) { v->buffer_destroy(be, ob_f16); }
    }

    {
        constexpr size_t rows = 2;
        constexpr size_t cols = 7;
        float x[rows * cols];
        float z[rows * cols];
        float gelu_ref[rows * cols];
        float mul_ref[rows * cols];
        float fused_ref[rows * cols];
        float got[rows * cols];
        for (size_t i = 0; i < rows * cols; i++) {
            x[i] = -2.25f + 0.37f * (float) i;
            z[i] = 0.35f + 0.11f * (float) ((i * 5u) % 9u);
            gelu_ref[i] = ref_gelu_tanh(x[i]);
            mul_ref[i] = x[i] * z[i];
            fused_ref[i] = gelu_ref[i] * z[i];
        }

        struct geist_buffer *xb = nullptr;
        struct geist_buffer *zb = nullptr;
        struct geist_buffer *yb = nullptr;
        enum geist_status ps = v->buffer_create(
            be, sizeof(x), GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_DEVICE, &xb);
        if (ps == GEIST_OK) {
            ps = v->buffer_create(be, sizeof(z), GEIST_BUFFER_ACTIVATION,
                                  GEIST_MEMORY_DEVICE, &zb);
        }
        if (ps == GEIST_OK) {
            ps = v->buffer_create(be, sizeof(got), GEIST_BUFFER_ACTIVATION,
                                  GEIST_MEMORY_DEVICE, &yb);
        }
        if (ps == GEIST_OK) {
            ps = v->buffer_upload(xb, sizeof(x), (const uint8_t *) x);
        }
        if (ps == GEIST_OK) {
            ps = v->buffer_upload(zb, sizeof(z), (const uint8_t *) z);
        }
        struct geist_tensor tx = tensor_f32_2d(xb, rows, cols);
        struct geist_tensor tz = tensor_f32_2d(zb, rows, cols);
        struct geist_tensor ty = tensor_f32_2d(yb, rows, cols);

        if (ps == GEIST_OK) {
            ps = v->gelu_tanh(be, &tx, &ty);
        }
        if (ps == GEIST_OK) {
            ps = v->buffer_download(sizeof(got), (uint8_t *) got, yb);
        }
        fails += check_status(ps, GEIST_OK, "Metal gelu_tanh runs");
        if (ps == GEIST_OK) {
            const ptrdiff_t bad = geist_fp32_close_array(
                got, gelu_ref, rows * cols, 2e-5f, 2e-6f);
            if (bad >= 0) {
                fprintf(stderr,
                        "FAIL: gelu_tanh[%td]: got %.7f want %.7f\n",
                        bad, (double) got[bad], (double) gelu_ref[bad]);
                fails++;
            }
        }

        if (ps == GEIST_OK) {
            ps = v->mul(be, &tx, &tz, &ty);
        }
        if (ps == GEIST_OK) {
            ps = v->buffer_download(sizeof(got), (uint8_t *) got, yb);
        }
        fails += check_status(ps, GEIST_OK, "Metal mul runs");
        if (ps == GEIST_OK) {
            const ptrdiff_t bad = geist_fp32_close_array(
                got, mul_ref, rows * cols, 1e-6f, 1e-6f);
            if (bad >= 0) {
                fprintf(stderr,
                        "FAIL: mul[%td]: got %.7f want %.7f\n",
                        bad, (double) got[bad], (double) mul_ref[bad]);
                fails++;
            }
        }

        if (ps == GEIST_OK) {
            ps = v->gelu_tanh_mul(be, &tx, &tz, &ty);
        }
        if (ps == GEIST_OK) {
            ps = v->buffer_download(sizeof(got), (uint8_t *) got, yb);
        }
        fails += check_status(ps, GEIST_OK, "Metal gelu_tanh_mul runs");
        if (ps == GEIST_OK) {
            const ptrdiff_t bad = geist_fp32_close_array(
                got, fused_ref, rows * cols, 2e-5f, 2e-6f);
            if (bad >= 0) {
                fprintf(stderr,
                        "FAIL: gelu_tanh_mul[%td]: got %.7f want %.7f\n",
                        bad, (double) got[bad], (double) fused_ref[bad]);
                fails++;
            }
        }

        if (xb != nullptr) { v->buffer_destroy(be, xb); }
        if (zb != nullptr) { v->buffer_destroy(be, zb); }
        if (yb != nullptr) { v->buffer_destroy(be, yb); }
    }

    {
        constexpr size_t f32_rows = 3;
        constexpr size_t f32_in = 5;
        constexpr size_t f32_out = 4;
        float f32_x[f32_rows * f32_in];
        float f32_w[f32_out * f32_in];
        float f32_ref[f32_rows * f32_out];
        float f32_got[f32_rows * f32_out];
        for (size_t i = 0; i < f32_rows * f32_in; i++) {
            f32_x[i] = sinf((float) i * 0.21f) + 0.1f * (float) (i % 3u);
        }
        for (size_t i = 0; i < f32_out * f32_in; i++) {
            f32_w[i] = cosf((float) i * 0.17f) - 0.05f * (float) (i % 5u);
        }
        for (size_t r = 0; r < f32_rows; r++) {
            for (size_t o = 0; o < f32_out; o++) {
                double acc = 0.0;
                for (size_t k = 0; k < f32_in; k++) {
                    acc += (double) f32_x[r * f32_in + k] *
                           (double) f32_w[o * f32_in + k];
                }
                f32_ref[r * f32_out + o] = (float) acc;
            }
        }
        memset(f32_got, 0, sizeof(f32_got));
        struct geist_buffer *f32_xb = nullptr;
        struct geist_buffer *f32_wb = nullptr;
        struct geist_buffer *f32_yb = nullptr;
        enum geist_status fs = v->buffer_create(
            be, sizeof(f32_x), GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_DEVICE,
            &f32_xb);
        if (fs == GEIST_OK) {
            fs = v->buffer_create(be, sizeof(f32_w), GEIST_BUFFER_WEIGHT,
                                  GEIST_MEMORY_DEVICE, &f32_wb);
        }
        if (fs == GEIST_OK) {
            fs = v->buffer_create(be, sizeof(f32_got),
                                  GEIST_BUFFER_ACTIVATION,
                                  GEIST_MEMORY_DEVICE, &f32_yb);
        }
        if (fs == GEIST_OK) {
            fs = v->buffer_upload(f32_xb, sizeof(f32_x),
                                  (const uint8_t *) f32_x);
        }
        if (fs == GEIST_OK) {
            fs = v->buffer_upload(f32_wb, sizeof(f32_w),
                                  (const uint8_t *) f32_w);
        }
        struct geist_tensor f32_tx =
            tensor_f32_2d(f32_xb, f32_rows, f32_in);
        struct geist_tensor f32_tw =
            tensor_f32_2d(f32_wb, f32_out, f32_in);
        struct geist_tensor f32_ty =
            tensor_f32_2d(f32_yb, f32_rows, f32_out);
        if (fs == GEIST_OK) {
            fs = v->matmul_f32_dense(be, &f32_tx, &f32_tw, &f32_ty);
        }
        if (fs == GEIST_OK) {
            fs = v->buffer_download(sizeof(f32_got),
                                    (uint8_t *) f32_got, f32_yb);
        }
        fails += check_status(fs, GEIST_OK, "F32 dense matmul runs");
        if (fs == GEIST_OK) {
            ptrdiff_t bad = geist_fp32_close_array(
                f32_got, f32_ref, f32_rows * f32_out, 2e-5f, 2e-6f);
            if (bad >= 0) {
                fprintf(stderr,
                        "FAIL: F32 matmul[%td]: got %.7f want %.7f\n",
                        bad, (double) f32_got[bad],
                        (double) f32_ref[bad]);
                fails++;
            }
        }
        if (f32_xb != nullptr) { v->buffer_destroy(be, f32_xb); }
        if (f32_wb != nullptr) { v->buffer_destroy(be, f32_wb); }
        if (f32_yb != nullptr) { v->buffer_destroy(be, f32_yb); }
    }

    constexpr size_t n_in = 512;
    constexpr size_t n_out = 9;
    constexpr size_t q4_bytes = n_out * (n_in / 256u) * 144u;
    float x[n_in];
    float y_ref[n_out];
    float y_got[n_out];
    uint8_t w[q4_bytes];
    for (size_t i = 0; i < n_in; i++) {
        x[i] = sinf((float) i * 0.091f) * 0.5f +
               cosf((float) i * 0.017f) * 0.125f;
    }
    pack_q4k_matrix(n_in, n_out, w);
    ref_q4k_matvec(x, n_in, n_out, y_ref);
    memset(y_got, 0, sizeof(y_got));

    struct geist_buffer *xb = nullptr;
    struct geist_buffer *wb = nullptr;
    struct geist_buffer *yb = nullptr;
    struct geist_buffer *emb_yb = nullptr;
    enum geist_status s = v->buffer_create(
        be, sizeof(x), GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_DEVICE, &xb);
    if (s == GEIST_OK) {
        s = v->buffer_create(be, sizeof(w), GEIST_BUFFER_WEIGHT,
                             GEIST_MEMORY_DEVICE, &wb);
    }
    if (s == GEIST_OK) {
        s = v->buffer_create(be, sizeof(y_got), GEIST_BUFFER_ACTIVATION,
                             GEIST_MEMORY_DEVICE, &yb);
    }
    fails += check_status(s, GEIST_OK, "Q4_K buffers create OK");
    if (s == GEIST_OK) {
        s = v->buffer_upload(xb, sizeof(x), (const uint8_t *) x);
    }
    if (s == GEIST_OK) {
        s = v->buffer_upload(wb, sizeof(w), w);
    }
    fails += check_status(s, GEIST_OK, "Q4_K buffers upload OK");

    struct geist_tensor tx = tensor_f32_1d(xb, n_in);
    struct geist_tensor tw = tensor_q4k_2d(wb, n_out, n_in);
    struct geist_tensor ty = tensor_f32_1d(yb, n_out);
    if (s == GEIST_OK) {
        s = v->matvec_q4k(be, &tx, &tw, &ty);
    }
    fails += check_status(s, GEIST_OK, "Q4_K raw matvec runs");
    if (s == GEIST_OK) {
        s = v->buffer_download(sizeof(y_got), (uint8_t *) y_got, yb);
    }
    fails += check_status(s, GEIST_OK, "Q4_K raw output download OK");
    if (s == GEIST_OK) {
        ptrdiff_t bad =
            geist_fp32_close_array(y_got, y_ref, n_out, 2e-4f, 4e-5f);
        if (bad >= 0) {
            fprintf(stderr,
                    "FAIL: Q4_K raw matvec[%td]: got %.7f want %.7f\n",
                    bad, (double) y_got[bad], (double) y_ref[bad]);
            fails++;
        }
    }
    memset(y_got, 0, sizeof(y_got));
    if (s == GEIST_OK) {
        s = v->prepare_weight_layout(be, &tw);
    }
    fails += check_status(s, GEIST_OK,
                          "Q4_K prepare packed weight layout OK");
    if (s == GEIST_OK) {
        s = v->matvec_q4k(be, &tx, &tw, &ty);
    }
    fails += check_status(s, GEIST_OK, "Q4_K matvec runs");
    if (s == GEIST_OK) {
        s = v->buffer_download(sizeof(y_got), (uint8_t *) y_got, yb);
    }
    fails += check_status(s, GEIST_OK, "Q4_K output download OK");
    if (s == GEIST_OK) {
        ptrdiff_t bad =
            geist_fp32_close_array(y_got, y_ref, n_out, 2e-4f, 4e-5f);
        if (bad >= 0) {
            fprintf(stderr,
                    "FAIL: Q4_K matvec[%td]: got %.7f want %.7f\n",
                    bad, (double) y_got[bad], (double) y_ref[bad]);
            fails++;
        }
    }

    constexpr geist_token_t emb_token = 4;
    constexpr float emb_scale = 1.75f;
    float emb_ref[n_in];
    float emb_got[n_in];
    for (size_t block = 0; block < n_in / 256u; block++) {
        for (size_t sub = 0; sub < 8u; sub++) {
            for (size_t idx = 0; idx < 32u; idx++) {
                const size_t k = block * 256u + sub * 32u + idx;
                emb_ref[k] = (float) q4k_value((size_t) emb_token, block,
                                               sub, idx) *
                             emb_scale;
            }
        }
    }
    memset(emb_got, 0, sizeof(emb_got));
    s = v->buffer_create(be, sizeof(emb_got), GEIST_BUFFER_ACTIVATION,
                         GEIST_MEMORY_DEVICE, &emb_yb);
    fails += check_status(s, GEIST_OK, "Q4_K embedding output buffer create OK");
    struct geist_tensor temb_y = tensor_f32_1d(emb_yb, n_in);
    if (s == GEIST_OK) {
        s = v->embedding_lookup_scaled(be, &tw, emb_token, emb_scale,
                                       &temb_y);
    }
    fails += check_status(s, GEIST_OK, "Q4_K scaled embedding lookup runs");
    if (s == GEIST_OK) {
        s = v->buffer_download(sizeof(emb_got), (uint8_t *) emb_got,
                               emb_yb);
    }
    fails += check_status(s, GEIST_OK, "Q4_K embedding output download OK");
    if (s == GEIST_OK) {
        ptrdiff_t bad =
            geist_fp32_close_array(emb_got, emb_ref, n_in, 0.0f, 0.0f);
        if (bad >= 0) {
            fprintf(stderr,
                    "FAIL: Q4_K embedding[%td]: got %.7f want %.7f\n",
                    bad, (double) emb_got[bad], (double) emb_ref[bad]);
            fails++;
        }
    }

    {
        constexpr size_t large_n_in = 256;
        constexpr size_t large_n_out = 8200;
        constexpr size_t large_w_bytes =
            large_n_out * (large_n_in / 256u) * 144u;
        float *large_x = heap_alloc_array_aligned(float, large_n_in);
        float *large_ref = heap_alloc_array_aligned(float, large_n_out);
        float *large_got = heap_calloc_array_aligned(float, large_n_out);
        uint8_t *large_w = heap_alloc_array_aligned(uint8_t, large_w_bytes);
        struct geist_buffer *large_xb = nullptr;
        struct geist_buffer *large_wb = nullptr;
        struct geist_buffer *large_yb = nullptr;
        if (large_x == nullptr || large_ref == nullptr ||
            large_got == nullptr || large_w == nullptr) {
            fprintf(stderr, "FAIL: large Q4_K host allocation failed\n");
            fails++;
        } else {
            for (size_t i = 0; i < large_n_in; i++) {
                large_x[i] = sinf((float) i * 0.071f) * 0.25f +
                             cosf((float) i * 0.037f) * 0.5f;
            }
            pack_q4k_matrix(large_n_in, large_n_out, large_w);
            ref_q4k_matvec(large_x, large_n_in, large_n_out, large_ref);

            s = v->buffer_create(be, large_n_in * sizeof(float),
                                 GEIST_BUFFER_ACTIVATION,
                                 GEIST_MEMORY_DEVICE, &large_xb);
            if (s == GEIST_OK) {
                s = v->buffer_create(be, large_w_bytes,
                                     GEIST_BUFFER_WEIGHT,
                                     GEIST_MEMORY_DEVICE, &large_wb);
            }
            if (s == GEIST_OK) {
                s = v->buffer_create(be, large_n_out * sizeof(float),
                                     GEIST_BUFFER_ACTIVATION,
                                     GEIST_MEMORY_DEVICE, &large_yb);
            }
            if (s == GEIST_OK) {
                s = v->buffer_upload(large_xb, large_n_in * sizeof(float),
                                     (const uint8_t *) large_x);
            }
            if (s == GEIST_OK) {
                s = v->buffer_upload(large_wb, large_w_bytes, large_w);
            }
            fails += check_status(s, GEIST_OK,
                                  "large Q4_K buffers setup OK");
            struct geist_tensor large_tx =
                tensor_f32_1d(large_xb, large_n_in);
            struct geist_tensor large_tw =
                tensor_q4k_2d(large_wb, large_n_out, large_n_in);
            struct geist_tensor large_ty =
                tensor_f32_1d(large_yb, large_n_out);
            if (s == GEIST_OK) {
                s = v->prepare_weight_layout_from_host(
                    be, &large_tw, large_w_bytes, large_w);
            }
            fails += check_status(
                s, GEIST_OK,
                "large Q4_K host prepare packed weight layout above legacy cap OK");
            if (s == GEIST_OK) {
                s = v->matvec_q4k(be, &large_tx, &large_tw, &large_ty);
            }
            fails += check_status(s, GEIST_OK,
                                  "large Q4_K prepared matvec runs");
            if (s == GEIST_OK) {
                s = v->buffer_download(large_n_out * sizeof(float),
                                       (uint8_t *) large_got, large_yb);
            }
            fails += check_status(s, GEIST_OK,
                                  "large Q4_K prepared output download OK");
            if (s == GEIST_OK) {
                ptrdiff_t bad = geist_fp32_close_array(
                    large_got, large_ref, large_n_out, 2e-4f, 4e-5f);
                if (bad >= 0) {
                    fprintf(stderr,
                            "FAIL: large Q4_K matvec[%td]: got %.7f want %.7f\n",
                            bad, (double) large_got[bad],
                            (double) large_ref[bad]);
                    fails++;
                }
            }
        }
        if (large_xb != nullptr) { v->buffer_destroy(be, large_xb); }
        if (large_wb != nullptr) { v->buffer_destroy(be, large_wb); }
        if (large_yb != nullptr) { v->buffer_destroy(be, large_yb); }
        safe_free((void **) &large_x);
        safe_free((void **) &large_ref);
        safe_free((void **) &large_got);
        safe_free((void **) &large_w);
        s = GEIST_OK;
    }

    int seq_token = 0;
    int nested_token = 0;
    constexpr size_t matmul_rows = 16;
    float x_matmul[matmul_rows * n_in];
    float y_matmul_ref[matmul_rows * n_out];
    float y_matmul_got[matmul_rows * n_out];
    for (size_t r = 0; r < matmul_rows; r++) {
        for (size_t i = 0; i < n_in; i++) {
            x_matmul[r * n_in + i] =
                sinf((float) (r * 17u + i) * 0.043f) * 0.375f +
                cosf((float) (r + i * 3u) * 0.019f) * 0.25f;
        }
    }
    ref_q4k_matmul(x_matmul, matmul_rows, n_in, n_out, y_matmul_ref);
    memset(y_matmul_got, 0, sizeof(y_matmul_got));

    struct geist_buffer *xmb = nullptr;
    struct geist_buffer *ymb = nullptr;
    s = v->buffer_create(be, sizeof(x_matmul), GEIST_BUFFER_ACTIVATION,
                         GEIST_MEMORY_DEVICE, &xmb);
    if (s == GEIST_OK) {
        s = v->buffer_create(be, sizeof(y_matmul_got),
                             GEIST_BUFFER_ACTIVATION,
                             GEIST_MEMORY_DEVICE, &ymb);
    }
    fails += check_status(s, GEIST_OK, "Q4_K matmul buffers create OK");
    if (s == GEIST_OK) {
        s = v->buffer_upload(xmb, sizeof(x_matmul),
                             (const uint8_t *) x_matmul);
    }
    fails += check_status(s, GEIST_OK, "Q4_K matmul input upload OK");
    struct geist_tensor txm = tensor_f32_2d(xmb, matmul_rows, n_in);
    struct geist_tensor tym = tensor_f32_2d(ymb, matmul_rows, n_out);
    if (s == GEIST_OK) {
        s = v->matmul_q4k(be, &txm, &tw, &tym);
    }
    fails += check_status(s, GEIST_OK, "Q4_K matmul runs");
    if (s == GEIST_OK) {
        s = v->buffer_download(sizeof(y_matmul_got),
                               (uint8_t *) y_matmul_got, ymb);
    }
    fails += check_status(s, GEIST_OK, "Q4_K matmul output download OK");
    if (s == GEIST_OK) {
        ptrdiff_t bad = geist_fp32_close_array(
            y_matmul_got, y_matmul_ref, matmul_rows * n_out, 2e-4f, 5e-5f);
        if (bad >= 0) {
            fprintf(stderr,
                    "FAIL: Q4_K matmul[%td]: got %.7f want %.7f\n",
                    bad, (double) y_matmul_got[bad],
                    (double) y_matmul_ref[bad]);
            fails++;
        }
    }

    {
        constexpr size_t sg_rows = 32;
        constexpr size_t sg_out = 64;
        constexpr size_t sg_q4_bytes = sg_out * (n_in / 256u) * 144u;
        float x_sg[sg_rows * n_in];
        float y_sg_ref[sg_rows * sg_out];
        float y_sg_got[sg_rows * sg_out];
        uint8_t w_sg[sg_q4_bytes];
        for (size_t r = 0; r < sg_rows; r++) {
            for (size_t i = 0; i < n_in; i++) {
                x_sg[r * n_in + i] =
                    sinf((float) (r * 23u + i) * 0.031f) * 0.3125f +
                    cosf((float) (r * 5u + i * 7u) * 0.011f) * 0.1875f;
            }
        }
        pack_q4k_matrix(n_in, sg_out, w_sg);
        ref_q4k_matmul(x_sg, sg_rows, n_in, sg_out, y_sg_ref);
        memset(y_sg_got, 0, sizeof(y_sg_got));

        struct geist_buffer *xsgb = nullptr;
        struct geist_buffer *wsgb = nullptr;
        struct geist_buffer *ysgb = nullptr;
        enum geist_status sg_s = v->buffer_create(
            be, sizeof(x_sg), GEIST_BUFFER_ACTIVATION,
            GEIST_MEMORY_DEVICE, &xsgb);
        if (sg_s == GEIST_OK) {
            sg_s = v->buffer_create(be, sizeof(w_sg),
                                    GEIST_BUFFER_WEIGHT,
                                    GEIST_MEMORY_DEVICE, &wsgb);
        }
        if (sg_s == GEIST_OK) {
            sg_s = v->buffer_create(be, sizeof(y_sg_got),
                                    GEIST_BUFFER_ACTIVATION,
                                    GEIST_MEMORY_DEVICE, &ysgb);
        }
        fails += check_status(sg_s, GEIST_OK,
                              "Q4_K 32x64 matmul buffers create OK");
        if (sg_s == GEIST_OK) {
            sg_s = v->buffer_upload(xsgb, sizeof(x_sg),
                                    (const uint8_t *) x_sg);
        }
        if (sg_s == GEIST_OK) {
            sg_s = v->buffer_upload(wsgb, sizeof(w_sg),
                                    (const uint8_t *) w_sg);
        }
        fails += check_status(sg_s, GEIST_OK,
                              "Q4_K 32x64 matmul inputs upload OK");
        struct geist_tensor txsg = tensor_f32_2d(xsgb, sg_rows, n_in);
        struct geist_tensor twsg = tensor_q4k_2d(wsgb, sg_out, n_in);
        struct geist_tensor tysg = tensor_f32_2d(ysgb, sg_rows, sg_out);
        if (sg_s == GEIST_OK) {
            sg_s = v->matmul_q4k(be, &txsg, &twsg, &tysg);
        }
        fails += check_status(sg_s, GEIST_OK,
                              "Q4_K 32x64 matmul runs");
        if (sg_s == GEIST_OK) {
            sg_s = v->buffer_download(sizeof(y_sg_got),
                                      (uint8_t *) y_sg_got, ysgb);
        }
        fails += check_status(sg_s, GEIST_OK,
                              "Q4_K 32x64 matmul output download OK");
        if (sg_s == GEIST_OK) {
            ptrdiff_t bad = geist_fp32_close_array(
                y_sg_got, y_sg_ref, sg_rows * sg_out, 2e-4f, 5e-5f);
            if (bad >= 0) {
                fprintf(stderr,
                        "FAIL: Q4_K 32x64 matmul[%td]: got %.7f want %.7f\n",
                        bad, (double) y_sg_got[bad],
                        (double) y_sg_ref[bad]);
                fails++;
            }
        }
        if (xsgb != nullptr) { v->buffer_destroy(be, xsgb); }
        if (wsgb != nullptr) { v->buffer_destroy(be, wsgb); }
        if (ysgb != nullptr) { v->buffer_destroy(be, ysgb); }
    }

    {
        constexpr size_t q6_out = 11;
        constexpr size_t q6_bytes = q6_out * (n_in / 256u) * 210u;
        float q6_ref[matmul_rows * q6_out];
        float q6_got[matmul_rows * q6_out];
        float q6_vec_ref[q6_out];
        float q6_vec_got[q6_out];
        uint8_t q6_w[q6_bytes];
        pack_q6k_matrix(n_in, q6_out, q6_w);
        ref_q6k_matmul(x, 1, n_in, q6_out, q6_vec_ref);
        ref_q6k_matmul(x_matmul, matmul_rows, n_in, q6_out, q6_ref);
        memset(q6_got, 0, sizeof(q6_got));
        memset(q6_vec_got, 0, sizeof(q6_vec_got));

        struct geist_buffer *q6_wb = nullptr;
        struct geist_buffer *q6_yb = nullptr;
        struct geist_buffer *q6_ymb = nullptr;
        s = v->buffer_create(be, sizeof(q6_w), GEIST_BUFFER_WEIGHT,
                             GEIST_MEMORY_DEVICE, &q6_wb);
        if (s == GEIST_OK) {
            s = v->buffer_create(be, sizeof(q6_vec_got),
                                 GEIST_BUFFER_ACTIVATION,
                                 GEIST_MEMORY_DEVICE, &q6_yb);
        }
        if (s == GEIST_OK) {
            s = v->buffer_create(be, sizeof(q6_got),
                                 GEIST_BUFFER_ACTIVATION,
                                 GEIST_MEMORY_DEVICE, &q6_ymb);
        }
        if (s == GEIST_OK) {
            s = v->buffer_upload(q6_wb, sizeof(q6_w), q6_w);
        }
        fails += check_status(s, GEIST_OK, "Q6_K buffers setup OK");
        struct geist_tensor q6_tw = tensor_q6k_2d(q6_wb, q6_out, n_in);
        struct geist_tensor q6_ty = tensor_f32_1d(q6_yb, q6_out);
        struct geist_tensor q6_tym =
            tensor_f32_2d(q6_ymb, matmul_rows, q6_out);
        if (s == GEIST_OK) {
            s = v->prepare_weight_layout(be, &q6_tw);
        }
        fails += check_status(s, GEIST_OK,
                              "Q6_K prepare packed weight layout OK");
        if (s == GEIST_OK) {
            s = v->matvec_q6k(be, &tx, &q6_tw, &q6_ty);
        }
        fails += check_status(s, GEIST_OK, "Q6_K matvec runs");
        if (s == GEIST_OK) {
            s = v->buffer_download(sizeof(q6_vec_got),
                                   (uint8_t *) q6_vec_got, q6_yb);
        }
        fails += check_status(s, GEIST_OK, "Q6_K matvec download OK");
        if (s == GEIST_OK) {
            ptrdiff_t bad = geist_fp32_close_array(
                q6_vec_got, q6_vec_ref, q6_out, 2e-4f, 5e-5f);
            if (bad >= 0) {
                fprintf(stderr,
                        "FAIL: Q6_K matvec[%td]: got %.7f want %.7f\n",
                        bad, (double) q6_vec_got[bad],
                        (double) q6_vec_ref[bad]);
                fails++;
            }
        }
        if (s == GEIST_OK) {
            s = v->matmul_q6k(be, &txm, &q6_tw, &q6_tym);
        }
        fails += check_status(s, GEIST_OK, "Q6_K matmul runs");
        if (s == GEIST_OK) {
            s = v->buffer_download(sizeof(q6_got),
                                   (uint8_t *) q6_got, q6_ymb);
        }
        fails += check_status(s, GEIST_OK, "Q6_K matmul download OK");
        if (s == GEIST_OK) {
            ptrdiff_t bad = geist_fp32_close_array(
                q6_got, q6_ref, matmul_rows * q6_out, 2e-4f, 5e-5f);
            if (bad >= 0) {
                fprintf(stderr,
                        "FAIL: Q6_K matmul[%td]: got %.7f want %.7f\n",
                        bad, (double) q6_got[bad], (double) q6_ref[bad]);
                fails++;
            }
        }
        if (q6_wb != nullptr) { v->buffer_destroy(be, q6_wb); }
        if (q6_yb != nullptr) { v->buffer_destroy(be, q6_yb); }
        if (q6_ymb != nullptr) { v->buffer_destroy(be, q6_ymb); }
        s = GEIST_OK;
    }

    {
        constexpr size_t q6_cache_in = 256;
        constexpr size_t q6_cache_out = 1024;
        constexpr size_t q6_cache_bytes =
            q6_cache_out * (q6_cache_in / 256u) * 210u;
        float *q6_cache_x = heap_alloc_array_aligned(float, q6_cache_in);
        float *q6_cache_ref = heap_alloc_array_aligned(float, q6_cache_out);
        float *q6_cache_got = heap_calloc_array_aligned(float, q6_cache_out);
        uint8_t *q6_cache_w =
            heap_alloc_array_aligned(uint8_t, q6_cache_bytes);
        struct geist_buffer *q6_cache_xb = nullptr;
        struct geist_buffer *q6_cache_wb = nullptr;
        struct geist_buffer *q6_cache_yb = nullptr;
        if (q6_cache_x == nullptr || q6_cache_ref == nullptr ||
            q6_cache_got == nullptr || q6_cache_w == nullptr) {
            fprintf(stderr, "FAIL: Q6_K cache host allocation failed\n");
            fails++;
        } else {
            for (size_t i = 0; i < q6_cache_in; i++) {
                q6_cache_x[i] = sinf((float) i * 0.051f) * 0.125f +
                                cosf((float) i * 0.017f) * 0.25f;
            }
            pack_q6k_matrix(q6_cache_in, q6_cache_out, q6_cache_w);
            ref_q6k_matmul(q6_cache_x, 1, q6_cache_in, q6_cache_out,
                           q6_cache_ref);
            s = v->buffer_create(be, q6_cache_in * sizeof(float),
                                 GEIST_BUFFER_ACTIVATION,
                                 GEIST_MEMORY_DEVICE, &q6_cache_xb);
            if (s == GEIST_OK) {
                s = v->buffer_create(be, q6_cache_bytes,
                                     GEIST_BUFFER_WEIGHT,
                                     GEIST_MEMORY_DEVICE, &q6_cache_wb);
            }
            if (s == GEIST_OK) {
                s = v->buffer_create(be, q6_cache_out * sizeof(float),
                                     GEIST_BUFFER_ACTIVATION,
                                     GEIST_MEMORY_DEVICE, &q6_cache_yb);
            }
            if (s == GEIST_OK) {
                s = v->buffer_upload(q6_cache_xb,
                                     q6_cache_in * sizeof(float),
                                     (const uint8_t *) q6_cache_x);
            }
            if (s == GEIST_OK) {
                s = v->buffer_upload(q6_cache_wb, q6_cache_bytes,
                                     q6_cache_w);
            }
            fails += check_status(s, GEIST_OK,
                                  "Q6_K cache buffers setup OK");
            struct geist_tensor q6_cache_tx =
                tensor_f32_1d(q6_cache_xb, q6_cache_in);
            struct geist_tensor q6_cache_tw =
                tensor_q6k_2d(q6_cache_wb, q6_cache_out, q6_cache_in);
            struct geist_tensor q6_cache_ty =
                tensor_f32_1d(q6_cache_yb, q6_cache_out);
            if (s == GEIST_OK) {
                s = v->prepare_weight_layout_from_host(
                    be, &q6_cache_tw, q6_cache_bytes, q6_cache_w);
            }
            fails += check_status(s, GEIST_OK,
                                  "Q6_K NT4 host cache prepare OK");
            if (s == GEIST_OK) {
                s = v->matvec_q6k(be, &q6_cache_tx, &q6_cache_tw,
                                  &q6_cache_ty);
            }
            fails += check_status(s, GEIST_OK,
                                  "Q6_K NT4 cached matvec runs");
            if (s == GEIST_OK) {
                s = v->buffer_download(q6_cache_out * sizeof(float),
                                       (uint8_t *) q6_cache_got,
                                       q6_cache_yb);
            }
            fails += check_status(s, GEIST_OK,
                                  "Q6_K NT4 cached output download OK");
            if (s == GEIST_OK) {
                ptrdiff_t bad = geist_fp32_close_array(
                    q6_cache_got, q6_cache_ref, q6_cache_out,
                    2e-4f, 5e-5f);
                if (bad >= 0) {
                    fprintf(stderr,
                            "FAIL: Q6_K NT4 cached matvec[%td]: got %.7f want %.7f\n",
                            bad, (double) q6_cache_got[bad],
                            (double) q6_cache_ref[bad]);
                    fails++;
                }
            }
        }
        if (q6_cache_xb != nullptr) { v->buffer_destroy(be, q6_cache_xb); }
        if (q6_cache_wb != nullptr) { v->buffer_destroy(be, q6_cache_wb); }
        if (q6_cache_yb != nullptr) { v->buffer_destroy(be, q6_cache_yb); }
        safe_free((void **) &q6_cache_x);
        safe_free((void **) &q6_cache_ref);
        safe_free((void **) &q6_cache_got);
        safe_free((void **) &q6_cache_w);
        s = GEIST_OK;
    }

    {
        constexpr size_t ple_rows = 3;
        constexpr size_t ple_d = 16;
        constexpr size_t ple_hpl = 10;
        float ple_hidden[ple_rows * ple_d];
        float ple_input[ple_rows * ple_hpl];
        float ple_gate_w[ple_hpl * ple_d];
        float ple_proj_w[ple_d * ple_hpl];
        float ple_norm[ple_d];
        float ple_gate[ple_rows * ple_hpl];
        float ple_proj[ple_rows * ple_d];
        float ple_post[ple_rows * ple_d];
        float ple_ref[ple_rows * ple_d];
        float ple_got[ple_rows * ple_d];

        for (size_t i = 0; i < ple_rows * ple_d; i++) {
            ple_hidden[i] = sinf((float) i * 0.17f) * 0.125f +
                            cosf((float) i * 0.07f) * 0.0625f;
        }
        for (size_t i = 0; i < ple_rows * ple_hpl; i++) {
            ple_input[i] = 0.75f + sinf((float) i * 0.11f) * 0.03125f;
        }
        for (size_t i = 0; i < ple_hpl * ple_d; i++) {
            ple_gate_w[i] = sinf((float) i * 0.05f) * 0.018f;
        }
        for (size_t i = 0; i < ple_d * ple_hpl; i++) {
            ple_proj_w[i] = cosf((float) i * 0.041f) * 0.015f;
        }
        for (size_t i = 0; i < ple_d; i++) {
            ple_norm[i] = 0.9f + (float) (i % 5u) * 0.025f;
        }
        ref_ple_f32_block(ple_hidden, ple_input, ple_gate_w, ple_proj_w,
                          ple_norm, ple_rows, ple_d, ple_hpl, 1e-6f,
                          ple_gate, ple_proj, ple_post, ple_ref);

        struct geist_buffer *hidden_b = nullptr;
        struct geist_buffer *input_b = nullptr;
        struct geist_buffer *gate_w_b = nullptr;
        struct geist_buffer *proj_w_b = nullptr;
        struct geist_buffer *norm_b = nullptr;
        struct geist_buffer *gate_b = nullptr;
        struct geist_buffer *proj_b = nullptr;
        struct geist_buffer *out_b = nullptr;

        s = v->buffer_create(be, sizeof(ple_hidden), GEIST_BUFFER_ACTIVATION,
                             GEIST_MEMORY_DEVICE, &hidden_b);
        if (s == GEIST_OK) {
            s = v->buffer_create(be, sizeof(ple_input), GEIST_BUFFER_ACTIVATION,
                                 GEIST_MEMORY_DEVICE, &input_b);
        }
        if (s == GEIST_OK) {
            s = v->buffer_create(be, sizeof(ple_gate_w), GEIST_BUFFER_WEIGHT,
                                 GEIST_MEMORY_DEVICE, &gate_w_b);
        }
        if (s == GEIST_OK) {
            s = v->buffer_create(be, sizeof(ple_proj_w), GEIST_BUFFER_WEIGHT,
                                 GEIST_MEMORY_DEVICE, &proj_w_b);
        }
        if (s == GEIST_OK) {
            s = v->buffer_create(be, sizeof(ple_norm), GEIST_BUFFER_WEIGHT,
                                 GEIST_MEMORY_DEVICE, &norm_b);
        }
        if (s == GEIST_OK) {
            s = v->buffer_create(be, sizeof(ple_gate), GEIST_BUFFER_SCRATCH,
                                 GEIST_MEMORY_DEVICE, &gate_b);
        }
        if (s == GEIST_OK) {
            s = v->buffer_create(be, sizeof(ple_proj), GEIST_BUFFER_SCRATCH,
                                 GEIST_MEMORY_DEVICE, &proj_b);
        }
        if (s == GEIST_OK) {
            s = v->buffer_create(be, sizeof(ple_got), GEIST_BUFFER_ACTIVATION,
                                 GEIST_MEMORY_DEVICE, &out_b);
        }
        fails += check_status(s, GEIST_OK, "PLE F32 block buffers create OK");

        if (s == GEIST_OK) {
            s = v->buffer_upload(hidden_b, sizeof(ple_hidden),
                                 (const uint8_t *) ple_hidden);
        }
        if (s == GEIST_OK) {
            s = v->buffer_upload(input_b, sizeof(ple_input),
                                 (const uint8_t *) ple_input);
        }
        if (s == GEIST_OK) {
            s = v->buffer_upload(gate_w_b, sizeof(ple_gate_w),
                                 (const uint8_t *) ple_gate_w);
        }
        if (s == GEIST_OK) {
            s = v->buffer_upload(proj_w_b, sizeof(ple_proj_w),
                                 (const uint8_t *) ple_proj_w);
        }
        if (s == GEIST_OK) {
            s = v->buffer_upload(norm_b, sizeof(ple_norm),
                                 (const uint8_t *) ple_norm);
        }
        fails += check_status(s, GEIST_OK, "PLE F32 block buffers upload OK");

        struct geist_tensor t_hidden =
            tensor_f32_2d(hidden_b, ple_rows, ple_d);
        struct geist_tensor t_input =
            tensor_f32_2d(input_b, ple_rows, ple_hpl);
        struct geist_tensor t_gate_w =
            tensor_f32_2d(gate_w_b, ple_hpl, ple_d);
        struct geist_tensor t_proj_w =
            tensor_f32_2d(proj_w_b, ple_d, ple_hpl);
        struct geist_tensor t_norm = tensor_f32_1d(norm_b, ple_d);
        struct geist_tensor t_gate =
            tensor_f32_2d(gate_b, ple_rows, ple_hpl);
        struct geist_tensor t_proj =
            tensor_f32_2d(proj_b, ple_rows, ple_d);
        struct geist_tensor t_out =
            tensor_f32_2d(out_b, ple_rows, ple_d);
        const struct geist_backend_ple_block ple_block = {
            .struct_size = sizeof(ple_block),
            .seq = ple_rows,
            .d_model = ple_d,
            .hidden_per_layer = ple_hpl,
            .eps = 1e-6f,
            .hidden = &t_hidden,
            .per_layer_input = &t_input,
            .per_layer_gate_weight = &t_gate_w,
            .per_layer_proj_weight = &t_proj_w,
            .post_per_layer_norm_weight = &t_norm,
            .gate_scratch = &t_gate,
            .proj_scratch = &t_proj,
            .out = &t_out,
        };
        if (s == GEIST_OK) {
            s = v->ple_block(be, &ple_block);
        }
        fails += check_status(s, GEIST_OK, "PLE F32 block runs");
        if (s == GEIST_OK) {
            s = v->buffer_download(sizeof(ple_got), (uint8_t *) ple_got,
                                   out_b);
        }
        fails += check_status(s, GEIST_OK, "PLE F32 block download OK");
        if (s == GEIST_OK) {
            ptrdiff_t bad = geist_fp32_close_array(
                ple_got, ple_ref, ple_rows * ple_d, 1e-4f, 2e-5f);
            if (bad >= 0) {
                fprintf(stderr,
                        "FAIL: PLE F32 block[%td]: got %.7f want %.7f\n",
                        bad, (double) ple_got[bad],
                        (double) ple_ref[bad]);
                fails++;
            }
        }

        if (hidden_b != nullptr) { v->buffer_destroy(be, hidden_b); }
        if (input_b != nullptr) { v->buffer_destroy(be, input_b); }
        if (gate_w_b != nullptr) { v->buffer_destroy(be, gate_w_b); }
        if (proj_w_b != nullptr) { v->buffer_destroy(be, proj_w_b); }
        if (norm_b != nullptr) { v->buffer_destroy(be, norm_b); }
        if (gate_b != nullptr) { v->buffer_destroy(be, gate_b); }
        if (proj_b != nullptr) { v->buffer_destroy(be, proj_b); }
        if (out_b != nullptr) { v->buffer_destroy(be, out_b); }
        s = GEIST_OK;
    }

    constexpr size_t ffn_rows = 9;
    constexpr size_t ffn_d_model = 512;
    constexpr size_t ffn_inter = 512;
    constexpr size_t ffn_q4_model_bytes =
        ffn_inter * (ffn_d_model / 256u) * 144u;
    constexpr size_t ffn_q4_down_bytes =
        ffn_d_model * (ffn_inter / 256u) * 144u;
    constexpr size_t ffn_q6_down_bytes =
        ffn_d_model * (ffn_inter / 256u) * 210u;
    float ffn_residual[ffn_rows * ffn_d_model];
    float ffn_norm[ffn_d_model];
    float ffn_post_norm[ffn_d_model];
    float ffn_pre[ffn_rows * ffn_d_model];
    float ffn_gate[ffn_rows * ffn_inter];
    float ffn_up[ffn_rows * ffn_inter];
    float ffn_mid[ffn_rows * ffn_inter];
    float ffn_out[ffn_rows * ffn_d_model];
    float ffn_post[ffn_rows * ffn_d_model];
    float ffn_ref[ffn_rows * ffn_d_model];
    float ffn_ref_q6[ffn_rows * ffn_d_model];
    float ffn_got[ffn_rows * ffn_d_model];
    uint8_t ffn_gate_w[ffn_q4_model_bytes];
    uint8_t ffn_up_w[ffn_q4_model_bytes];
    uint8_t ffn_down_w[ffn_q4_down_bytes];
    uint8_t ffn_down_q6_w[ffn_q6_down_bytes];

    for (size_t i = 0; i < ffn_rows * ffn_d_model; i++) {
        ffn_residual[i] =
            sinf((float) i * 0.031f) * 0.0625f +
            cosf((float) i * 0.011f) * 0.03125f;
    }
    for (size_t i = 0; i < ffn_d_model; i++) {
        ffn_norm[i] = 0.0075f + (float) (i % 7u) * 0.00025f;
        ffn_post_norm[i] = 0.25f + (float) (i % 5u) * 0.015625f;
    }
    pack_q4k_matrix(ffn_d_model, ffn_inter, ffn_gate_w);
    pack_q4k_matrix(ffn_d_model, ffn_inter, ffn_up_w);
    pack_q4k_matrix(ffn_inter, ffn_d_model, ffn_down_w);
    pack_q6k_matrix(ffn_inter, ffn_d_model, ffn_down_q6_w);
    ref_ffn_geglu_q4k(ffn_residual, ffn_norm, ffn_gate_w, ffn_up_w,
                      ffn_down_w, ffn_post_norm, ffn_rows, ffn_d_model,
                      ffn_inter, 1e-6f, ffn_pre, ffn_gate, ffn_up, ffn_mid,
                      ffn_out, ffn_post, ffn_ref);
    ref_ffn_geglu_q4k_q6down(ffn_residual, ffn_norm, ffn_post_norm,
                             ffn_rows, ffn_d_model, ffn_inter, 1e-6f,
                             ffn_pre, ffn_gate, ffn_up, ffn_mid, ffn_out,
                             ffn_post, ffn_ref_q6);
    memset(ffn_got, 0, sizeof(ffn_got));

    struct geist_buffer *ffn_residual_b = nullptr;
    struct geist_buffer *ffn_norm_b = nullptr;
    struct geist_buffer *ffn_post_norm_b = nullptr;
    struct geist_buffer *ffn_gate_w_b = nullptr;
    struct geist_buffer *ffn_up_w_b = nullptr;
    struct geist_buffer *ffn_down_w_b = nullptr;
    struct geist_buffer *ffn_down_q6_w_b = nullptr;
    struct geist_buffer *ffn_pre_b = nullptr;
    struct geist_buffer *ffn_gate_b = nullptr;
    struct geist_buffer *ffn_up_b = nullptr;
    struct geist_buffer *ffn_out_b = nullptr;
    struct geist_buffer *ffn_post_b = nullptr;
    struct geist_buffer *ffn_dst_b = nullptr;

    s = v->buffer_create(be, sizeof(ffn_residual), GEIST_BUFFER_ACTIVATION,
                         GEIST_MEMORY_DEVICE, &ffn_residual_b);
    if (s == GEIST_OK) {
        s = v->buffer_create(be, sizeof(ffn_norm), GEIST_BUFFER_WEIGHT,
                             GEIST_MEMORY_DEVICE, &ffn_norm_b);
    }
    if (s == GEIST_OK) {
        s = v->buffer_create(be, sizeof(ffn_post_norm), GEIST_BUFFER_WEIGHT,
                             GEIST_MEMORY_DEVICE, &ffn_post_norm_b);
    }
    if (s == GEIST_OK) {
        s = v->buffer_create(be, sizeof(ffn_gate_w), GEIST_BUFFER_WEIGHT,
                             GEIST_MEMORY_DEVICE, &ffn_gate_w_b);
    }
    if (s == GEIST_OK) {
        s = v->buffer_create(be, sizeof(ffn_up_w), GEIST_BUFFER_WEIGHT,
                             GEIST_MEMORY_DEVICE, &ffn_up_w_b);
    }
    if (s == GEIST_OK) {
        s = v->buffer_create(be, sizeof(ffn_down_w), GEIST_BUFFER_WEIGHT,
                             GEIST_MEMORY_DEVICE, &ffn_down_w_b);
    }
    if (s == GEIST_OK) {
        s = v->buffer_create(be, sizeof(ffn_down_q6_w), GEIST_BUFFER_WEIGHT,
                             GEIST_MEMORY_DEVICE, &ffn_down_q6_w_b);
    }
    if (s == GEIST_OK) {
        s = v->buffer_create(be, sizeof(ffn_pre), GEIST_BUFFER_SCRATCH,
                             GEIST_MEMORY_DEVICE, &ffn_pre_b);
    }
    if (s == GEIST_OK) {
        s = v->buffer_create(be, sizeof(ffn_gate), GEIST_BUFFER_SCRATCH,
                             GEIST_MEMORY_DEVICE, &ffn_gate_b);
    }
    if (s == GEIST_OK) {
        s = v->buffer_create(be, sizeof(ffn_up), GEIST_BUFFER_SCRATCH,
                             GEIST_MEMORY_DEVICE, &ffn_up_b);
    }
    if (s == GEIST_OK) {
        s = v->buffer_create(be, sizeof(ffn_out), GEIST_BUFFER_SCRATCH,
                             GEIST_MEMORY_DEVICE, &ffn_out_b);
    }
    if (s == GEIST_OK) {
        s = v->buffer_create(be, sizeof(ffn_post), GEIST_BUFFER_SCRATCH,
                             GEIST_MEMORY_DEVICE, &ffn_post_b);
    }
    if (s == GEIST_OK) {
        s = v->buffer_create(be, sizeof(ffn_got), GEIST_BUFFER_ACTIVATION,
                             GEIST_MEMORY_DEVICE, &ffn_dst_b);
    }
    fails += check_status(s, GEIST_OK, "FFN block buffers create OK");

    if (s == GEIST_OK) {
        s = v->buffer_upload(ffn_residual_b, sizeof(ffn_residual),
                             (const uint8_t *) ffn_residual);
    }
    if (s == GEIST_OK) {
        s = v->buffer_upload(ffn_norm_b, sizeof(ffn_norm),
                             (const uint8_t *) ffn_norm);
    }
    if (s == GEIST_OK) {
        s = v->buffer_upload(ffn_post_norm_b, sizeof(ffn_post_norm),
                             (const uint8_t *) ffn_post_norm);
    }
    if (s == GEIST_OK) {
        s = v->buffer_upload(ffn_gate_w_b, sizeof(ffn_gate_w), ffn_gate_w);
    }
    if (s == GEIST_OK) {
        s = v->buffer_upload(ffn_up_w_b, sizeof(ffn_up_w), ffn_up_w);
    }
    if (s == GEIST_OK) {
        s = v->buffer_upload(ffn_down_w_b, sizeof(ffn_down_w), ffn_down_w);
    }
    if (s == GEIST_OK) {
        s = v->buffer_upload(ffn_down_q6_w_b, sizeof(ffn_down_q6_w),
                             ffn_down_q6_w);
    }
    fails += check_status(s, GEIST_OK, "FFN block buffers upload OK");

    struct geist_tensor t_ffn_residual =
        tensor_f32_2d(ffn_residual_b, ffn_rows, ffn_d_model);
    struct geist_tensor t_ffn_norm = tensor_f32_1d(ffn_norm_b, ffn_d_model);
    struct geist_tensor t_ffn_post_norm =
        tensor_f32_1d(ffn_post_norm_b, ffn_d_model);
    struct geist_tensor t_ffn_gate_w =
        tensor_q4k_2d(ffn_gate_w_b, ffn_inter, ffn_d_model);
    struct geist_tensor t_ffn_up_w =
        tensor_q4k_2d(ffn_up_w_b, ffn_inter, ffn_d_model);
    struct geist_tensor t_ffn_down_w =
        tensor_q4k_2d(ffn_down_w_b, ffn_d_model, ffn_inter);
    struct geist_tensor t_ffn_down_q6_w =
        tensor_q6k_2d(ffn_down_q6_w_b, ffn_d_model, ffn_inter);
    struct geist_tensor t_ffn_pre =
        tensor_f32_2d(ffn_pre_b, ffn_rows, ffn_d_model);
    struct geist_tensor t_ffn_gate =
        tensor_f32_2d(ffn_gate_b, ffn_rows, ffn_inter);
    struct geist_tensor t_ffn_up =
        tensor_f32_2d(ffn_up_b, ffn_rows, ffn_inter);
    struct geist_tensor t_ffn_out =
        tensor_f32_2d(ffn_out_b, ffn_rows, ffn_d_model);
    struct geist_tensor t_ffn_post =
        tensor_f32_2d(ffn_post_b, ffn_rows, ffn_d_model);
    struct geist_tensor t_ffn_dst =
        tensor_f32_2d(ffn_dst_b, ffn_rows, ffn_d_model);
    if (s == GEIST_OK) {
        s = v->prepare_weight_layout(be, &t_ffn_gate_w);
    }
    if (s == GEIST_OK) {
        s = v->prepare_weight_layout(be, &t_ffn_up_w);
    }
    if (s == GEIST_OK) {
        s = v->prepare_weight_layout(be, &t_ffn_down_w);
    }
    fails += check_status(s, GEIST_OK,
                          "FFN Q4_K prepare packed weight layouts OK");
    const struct geist_backend_ffn_geglu_block ffn_block = {
        .struct_size = sizeof(ffn_block),
        .seq = ffn_rows,
        .d_model = ffn_d_model,
        .inter = ffn_inter,
        .eps = 1e-6f,
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
        .out = &t_ffn_dst,
    };
    if (s == GEIST_OK) {
        s = v->ffn_geglu_block(be, &ffn_block);
    }
    fails += check_status(s, GEIST_OK, "FFN GEGLU block runs");
    if (s == GEIST_OK) {
        s = v->buffer_download(sizeof(ffn_got), (uint8_t *) ffn_got,
                               ffn_dst_b);
    }
    fails += check_status(s, GEIST_OK, "FFN GEGLU block download OK");
    if (s == GEIST_OK) {
        ptrdiff_t bad = geist_fp32_close_array(
            ffn_got, ffn_ref, ffn_rows * ffn_d_model, 1e-3f, 2e-4f);
        if (bad >= 0) {
            fprintf(stderr,
                    "FAIL: FFN GEGLU block[%td]: got %.7f want %.7f\n",
                    bad, (double) ffn_got[bad], (double) ffn_ref[bad]);
            fails++;
        }
    }

    struct geist_tensor t_ffn_residual_one =
        tensor_f32_2d(ffn_residual_b, 1, ffn_d_model);
    struct geist_tensor t_ffn_pre_one =
        tensor_f32_2d(ffn_pre_b, 1, ffn_d_model);
    struct geist_tensor t_ffn_gate_one =
        tensor_f32_2d(ffn_gate_b, 1, ffn_inter);
    struct geist_tensor t_ffn_up_one =
        tensor_f32_2d(ffn_up_b, 1, ffn_inter);
    struct geist_tensor t_ffn_out_one =
        tensor_f32_2d(ffn_out_b, 1, ffn_d_model);
    struct geist_tensor t_ffn_post_one =
        tensor_f32_2d(ffn_post_b, 1, ffn_d_model);
    struct geist_tensor t_ffn_dst_one =
        tensor_f32_2d(ffn_dst_b, 1, ffn_d_model);
    const struct geist_backend_ffn_geglu_block ffn_one_row_block = {
        .struct_size = sizeof(ffn_one_row_block),
        .seq = 1,
        .d_model = ffn_d_model,
        .inter = ffn_inter,
        .eps = 1e-6f,
        .residual = &t_ffn_residual_one,
        .ffn_norm_weight = &t_ffn_norm,
        .gate_weight = &t_ffn_gate_w,
        .up_weight = &t_ffn_up_w,
        .down_weight = &t_ffn_down_w,
        .post_ffw_norm_weight = &t_ffn_post_norm,
        .pre_ff_scratch = &t_ffn_pre_one,
        .gate_scratch = &t_ffn_gate_one,
        .up_scratch = &t_ffn_up_one,
        .ffn_out_scratch = &t_ffn_out_one,
        .post_ff_scratch = &t_ffn_post_one,
        .out = &t_ffn_dst_one,
    };
    if (s == GEIST_OK) {
        memset(ffn_got, 0, sizeof(ffn_got));
        s = v->ffn_geglu_block(be, &ffn_one_row_block);
    }
    fails += check_status(s, GEIST_OK,
                          "FFN GEGLU one-row packed block runs");
    if (s == GEIST_OK) {
        s = v->buffer_download(sizeof(ffn_got), (uint8_t *) ffn_got,
                               ffn_dst_b);
    }
    fails += check_status(s, GEIST_OK,
                          "FFN GEGLU one-row packed block download OK");
    if (s == GEIST_OK) {
        ptrdiff_t bad = geist_fp32_close_array(
            ffn_got, ffn_ref, ffn_d_model, 1e-3f, 2e-4f);
        if (bad >= 0) {
            fprintf(stderr,
                    "FAIL: FFN GEGLU packed row[%td]: got %.7f want %.7f\n",
                    bad, (double) ffn_got[bad], (double) ffn_ref[bad]);
            fails++;
        }
    }

    const struct geist_backend_ffn_geglu_block ffn_q6_down_block = {
        .struct_size = sizeof(ffn_q6_down_block),
        .seq = ffn_rows,
        .d_model = ffn_d_model,
        .inter = ffn_inter,
        .eps = 1e-6f,
        .residual = &t_ffn_residual,
        .ffn_norm_weight = &t_ffn_norm,
        .gate_weight = &t_ffn_gate_w,
        .up_weight = &t_ffn_up_w,
        .down_weight = &t_ffn_down_q6_w,
        .post_ffw_norm_weight = &t_ffn_post_norm,
        .pre_ff_scratch = &t_ffn_pre,
        .gate_scratch = &t_ffn_gate,
        .up_scratch = &t_ffn_up,
        .ffn_out_scratch = &t_ffn_out,
        .post_ff_scratch = &t_ffn_post,
        .out = &t_ffn_dst,
    };
    if (s == GEIST_OK) {
        s = v->prepare_weight_layout(be, &t_ffn_down_q6_w);
    }
    fails += check_status(s, GEIST_OK,
                          "FFN Q6_K prepare packed down layout OK");
    if (s == GEIST_OK) {
        memset(ffn_got, 0, sizeof(ffn_got));
        s = v->ffn_geglu_block(be, &ffn_q6_down_block);
    }
    fails += check_status(s, GEIST_OK, "FFN GEGLU Q6 down block runs");
    if (s == GEIST_OK) {
        s = v->buffer_download(sizeof(ffn_got), (uint8_t *) ffn_got,
                               ffn_dst_b);
    }
    fails += check_status(s, GEIST_OK,
                          "FFN GEGLU Q6 down block download OK");
    if (s == GEIST_OK) {
        ptrdiff_t bad = geist_fp32_close_array(
            ffn_got, ffn_ref_q6, ffn_rows * ffn_d_model, 1e-3f, 2e-4f);
        if (bad >= 0) {
            fprintf(stderr,
                    "FAIL: FFN GEGLU Q6 down block[%td]: got %.7f want %.7f\n",
                    bad, (double) ffn_got[bad], (double) ffn_ref_q6[bad]);
            fails++;
        }
    }

    const struct geist_backend_ffn_geglu_block ffn_q6_down_one_row_block = {
        .struct_size = sizeof(ffn_q6_down_one_row_block),
        .seq = 1,
        .d_model = ffn_d_model,
        .inter = ffn_inter,
        .eps = 1e-6f,
        .residual = &t_ffn_residual_one,
        .ffn_norm_weight = &t_ffn_norm,
        .gate_weight = &t_ffn_gate_w,
        .up_weight = &t_ffn_up_w,
        .down_weight = &t_ffn_down_q6_w,
        .post_ffw_norm_weight = &t_ffn_post_norm,
        .pre_ff_scratch = &t_ffn_pre_one,
        .gate_scratch = &t_ffn_gate_one,
        .up_scratch = &t_ffn_up_one,
        .ffn_out_scratch = &t_ffn_out_one,
        .post_ff_scratch = &t_ffn_post_one,
        .out = &t_ffn_dst_one,
    };
    if (s == GEIST_OK) {
        memset(ffn_got, 0, sizeof(ffn_got));
        s = v->ffn_geglu_block(be, &ffn_q6_down_one_row_block);
    }
    fails += check_status(s, GEIST_OK,
                          "FFN GEGLU Q6 down one-row block runs");
    if (s == GEIST_OK) {
        s = v->buffer_download(sizeof(ffn_got), (uint8_t *) ffn_got,
                               ffn_dst_b);
    }
    fails += check_status(s, GEIST_OK,
                          "FFN GEGLU Q6 down one-row block download OK");
    if (s == GEIST_OK) {
        ptrdiff_t bad = geist_fp32_close_array(
            ffn_got, ffn_ref_q6, ffn_d_model, 1e-3f, 2e-4f);
        if (bad >= 0) {
            fprintf(stderr,
                    "FAIL: FFN GEGLU Q6 down one-row block[%td]: got %.7f "
                    "want %.7f\n",
                    bad, (double) ffn_got[bad],
                    (double) ffn_ref_q6[bad]);
            fails++;
        }
    }

    {
        constexpr size_t attn_d = 512;
        constexpr size_t attn_q_heads = 2;
        constexpr size_t attn_kv_heads = 1;
        constexpr size_t attn_hd = 256;
        constexpr size_t attn_q_out = attn_q_heads * attn_hd;
        constexpr size_t attn_kv_out = attn_kv_heads * attn_hd;
        constexpr size_t attn_kv_len = 2;
        constexpr size_t attn_q_bytes =
            attn_q_out * (attn_d / 256u) * 144u;
        constexpr size_t attn_kv_bytes =
            attn_kv_out * (attn_d / 256u) * 144u;
        constexpr size_t attn_o_bytes =
            attn_d * (attn_q_out / 256u) * 144u;
        float attn_residual[attn_d];
        float attn_norm[attn_d];
        float attn_q_norm[attn_hd];
        float attn_k_norm[attn_hd];
        float attn_v_norm[attn_hd];
        float attn_post_norm[attn_d];
        float attn_cos[attn_hd];
        float attn_sin[attn_hd];
        float attn_k_cache[attn_kv_len * attn_kv_out];
        float attn_v_cache[attn_kv_len * attn_kv_out];
        float attn_k_cache_ref[attn_kv_len * attn_kv_out];
        float attn_v_cache_ref[attn_kv_len * attn_kv_out];
        uint16_t attn_k_cache_f16[attn_kv_len * attn_kv_out];
        uint16_t attn_v_cache_f16[attn_kv_len * attn_kv_out];
        uint16_t attn_k_cache_f16_bits[attn_kv_len * attn_kv_out];
        uint16_t attn_v_cache_f16_bits[attn_kv_len * attn_kv_out];
        float attn_k_cache_f16_got[attn_kv_len * attn_kv_out];
        float attn_v_cache_f16_got[attn_kv_len * attn_kv_out];
        float attn_normed[attn_d];
        float attn_q[attn_q_out];
        float attn_k[attn_kv_out];
        float attn_v[attn_kv_out];
        float attn_mid[attn_q_out];
        float attn_o[attn_d];
        float attn_post[attn_d];
        float attn_ref[attn_d];
        float attn_got[attn_d];
        float attn_f16_got[attn_d];
        float attn_appended_k[attn_kv_out];
        float attn_appended_v[attn_kv_out];
        uint16_t attn_appended_k_f16[attn_kv_out];
        uint16_t attn_appended_v_f16[attn_kv_out];
        float attn_appended_k_f16_ref[attn_kv_out];
        float attn_appended_v_f16_ref[attn_kv_out];
        uint8_t attn_q_w[attn_q_bytes];
        uint8_t attn_k_w[attn_kv_bytes];
        uint8_t attn_v_w[attn_kv_bytes];
        uint8_t attn_o_w[attn_o_bytes];

        for (size_t i = 0; i < attn_d; i++) {
            attn_residual[i] =
                sinf((float) i * 0.021f) * 0.03125f +
                cosf((float) i * 0.017f) * 0.015625f;
            attn_norm[i] = 0.00625f + (float) (i % 9u) * 0.000125f;
            attn_post_norm[i] = 0.50f + (float) (i % 7u) * 0.0078125f;
        }
        for (size_t i = 0; i < attn_hd; i++) {
            attn_q_norm[i] = 0.50f + (float) (i % 5u) * 0.0078125f;
            attn_k_norm[i] = 0.75f + (float) (i % 3u) * 0.015625f;
            attn_v_norm[i] = 1.0f;
            attn_cos[i] = 1.0f;
            attn_sin[i] = 0.0f;
        }
        for (size_t i = 0; i < attn_kv_len * attn_kv_out; i++) {
            attn_k_cache[i] = (float) ((int) (i % 17u) - 8) * 0.00390625f;
            attn_v_cache[i] = (float) ((int) (i % 13u) - 6) * 0.0048828125f;
            attn_k_cache_ref[i] = attn_k_cache[i];
            attn_v_cache_ref[i] = attn_v_cache[i];
        }
        encode_f16_array(attn_kv_len * attn_kv_out, attn_k_cache,
                         attn_k_cache_f16);
        encode_f16_array(attn_kv_len * attn_kv_out, attn_v_cache,
                         attn_v_cache_f16);
        pack_q4k_matrix(attn_d, attn_q_out, attn_q_w);
        pack_q4k_matrix(attn_d, attn_kv_out, attn_k_w);
        pack_q4k_matrix(attn_d, attn_kv_out, attn_v_w);
        pack_q4k_matrix(attn_q_out, attn_d, attn_o_w);
        ref_attention_q4k(attn_residual, attn_norm, attn_q_w, attn_k_w,
                          attn_v_w, attn_q_norm, attn_k_norm, attn_v_norm,
                          attn_o_w, attn_post_norm, attn_k_cache_ref,
                          attn_v_cache_ref, 1, attn_kv_len, attn_d,
                          attn_q_heads, attn_kv_heads, attn_hd, 1e-6f,
                          attn_normed, attn_q, attn_k, attn_v, attn_mid,
                          attn_o, attn_post, attn_ref, attn_appended_k,
                          attn_appended_v);
        encode_f16_array(attn_kv_out, attn_appended_k, attn_appended_k_f16);
        encode_f16_array(attn_kv_out, attn_appended_v, attn_appended_v_f16);
        decode_f16_array(attn_kv_out, attn_appended_k_f16,
                         attn_appended_k_f16_ref);
        decode_f16_array(attn_kv_out, attn_appended_v_f16,
                         attn_appended_v_f16_ref);
        memset(attn_got, 0, sizeof(attn_got));
        memset(attn_f16_got, 0, sizeof(attn_f16_got));

        struct geist_buffer *res_b = nullptr;
        struct geist_buffer *norm_b = nullptr;
        struct geist_buffer *q_norm_b = nullptr;
        struct geist_buffer *k_norm_b = nullptr;
        struct geist_buffer *v_norm_b = nullptr;
        struct geist_buffer *post_norm_b = nullptr;
        struct geist_buffer *cos_b = nullptr;
        struct geist_buffer *sin_b = nullptr;
        struct geist_buffer *q_w_b = nullptr;
        struct geist_buffer *k_w_b = nullptr;
        struct geist_buffer *v_w_b = nullptr;
        struct geist_buffer *o_w_b = nullptr;
        struct geist_buffer *k_cache_b = nullptr;
        struct geist_buffer *v_cache_b = nullptr;
        struct geist_buffer *k_cache_f16_b = nullptr;
        struct geist_buffer *v_cache_f16_b = nullptr;
        struct geist_buffer *normed_b = nullptr;
        struct geist_buffer *q_b = nullptr;
        struct geist_buffer *k_b = nullptr;
        struct geist_buffer *v_b = nullptr;
        struct geist_buffer *mid_b = nullptr;
        struct geist_buffer *o_b = nullptr;
        struct geist_buffer *post_b = nullptr;
        struct geist_buffer *out_b = nullptr;
        struct geist_buffer *out_f16_b = nullptr;

#define CREATE_ATTN_BUF(buf, bytes, role) \
        do { \
            if (s == GEIST_OK) { \
                s = v->buffer_create(be, (bytes), (role), \
                                     GEIST_MEMORY_DEVICE, &(buf)); \
            } \
        } while (0)
#define UPLOAD_ATTN_BUF(buf, bytes, ptr) \
        do { \
            if (s == GEIST_OK) { \
                s = v->buffer_upload((buf), (bytes), (const uint8_t *) (ptr)); \
            } \
        } while (0)

        CREATE_ATTN_BUF(res_b, sizeof(attn_residual), GEIST_BUFFER_ACTIVATION);
        CREATE_ATTN_BUF(norm_b, sizeof(attn_norm), GEIST_BUFFER_WEIGHT);
        CREATE_ATTN_BUF(q_norm_b, sizeof(attn_q_norm), GEIST_BUFFER_WEIGHT);
        CREATE_ATTN_BUF(k_norm_b, sizeof(attn_k_norm), GEIST_BUFFER_WEIGHT);
        CREATE_ATTN_BUF(v_norm_b, sizeof(attn_v_norm), GEIST_BUFFER_WEIGHT);
        CREATE_ATTN_BUF(post_norm_b, sizeof(attn_post_norm), GEIST_BUFFER_WEIGHT);
        CREATE_ATTN_BUF(cos_b, sizeof(attn_cos), GEIST_BUFFER_WEIGHT);
        CREATE_ATTN_BUF(sin_b, sizeof(attn_sin), GEIST_BUFFER_WEIGHT);
        CREATE_ATTN_BUF(q_w_b, sizeof(attn_q_w), GEIST_BUFFER_WEIGHT);
        CREATE_ATTN_BUF(k_w_b, sizeof(attn_k_w), GEIST_BUFFER_WEIGHT);
        CREATE_ATTN_BUF(v_w_b, sizeof(attn_v_w), GEIST_BUFFER_WEIGHT);
        CREATE_ATTN_BUF(o_w_b, sizeof(attn_o_w), GEIST_BUFFER_WEIGHT);
        CREATE_ATTN_BUF(k_cache_b, sizeof(attn_k_cache), GEIST_BUFFER_KV_CACHE);
        CREATE_ATTN_BUF(v_cache_b, sizeof(attn_v_cache), GEIST_BUFFER_KV_CACHE);
        CREATE_ATTN_BUF(k_cache_f16_b, sizeof(attn_k_cache_f16),
                        GEIST_BUFFER_KV_CACHE);
        CREATE_ATTN_BUF(v_cache_f16_b, sizeof(attn_v_cache_f16),
                        GEIST_BUFFER_KV_CACHE);
        CREATE_ATTN_BUF(normed_b, sizeof(attn_normed), GEIST_BUFFER_SCRATCH);
        CREATE_ATTN_BUF(q_b, sizeof(attn_q), GEIST_BUFFER_SCRATCH);
        CREATE_ATTN_BUF(k_b, sizeof(attn_k), GEIST_BUFFER_SCRATCH);
        CREATE_ATTN_BUF(v_b, sizeof(attn_v), GEIST_BUFFER_SCRATCH);
        CREATE_ATTN_BUF(mid_b, sizeof(attn_mid), GEIST_BUFFER_SCRATCH);
        CREATE_ATTN_BUF(o_b, sizeof(attn_o), GEIST_BUFFER_SCRATCH);
        CREATE_ATTN_BUF(post_b, sizeof(attn_post), GEIST_BUFFER_SCRATCH);
        CREATE_ATTN_BUF(out_b, sizeof(attn_got), GEIST_BUFFER_ACTIVATION);
        CREATE_ATTN_BUF(out_f16_b, sizeof(attn_f16_got),
                        GEIST_BUFFER_ACTIVATION);
        fails += check_status(s, GEIST_OK, "attention block buffers create OK");

        UPLOAD_ATTN_BUF(res_b, sizeof(attn_residual), attn_residual);
        UPLOAD_ATTN_BUF(norm_b, sizeof(attn_norm), attn_norm);
        UPLOAD_ATTN_BUF(q_norm_b, sizeof(attn_q_norm), attn_q_norm);
        UPLOAD_ATTN_BUF(k_norm_b, sizeof(attn_k_norm), attn_k_norm);
        UPLOAD_ATTN_BUF(v_norm_b, sizeof(attn_v_norm), attn_v_norm);
        UPLOAD_ATTN_BUF(post_norm_b, sizeof(attn_post_norm), attn_post_norm);
        UPLOAD_ATTN_BUF(cos_b, sizeof(attn_cos), attn_cos);
        UPLOAD_ATTN_BUF(sin_b, sizeof(attn_sin), attn_sin);
        UPLOAD_ATTN_BUF(q_w_b, sizeof(attn_q_w), attn_q_w);
        UPLOAD_ATTN_BUF(k_w_b, sizeof(attn_k_w), attn_k_w);
        UPLOAD_ATTN_BUF(v_w_b, sizeof(attn_v_w), attn_v_w);
        UPLOAD_ATTN_BUF(o_w_b, sizeof(attn_o_w), attn_o_w);
        UPLOAD_ATTN_BUF(k_cache_b, sizeof(attn_k_cache), attn_k_cache);
        UPLOAD_ATTN_BUF(v_cache_b, sizeof(attn_v_cache), attn_v_cache);
        UPLOAD_ATTN_BUF(k_cache_f16_b, sizeof(attn_k_cache_f16),
                        attn_k_cache_f16);
        UPLOAD_ATTN_BUF(v_cache_f16_b, sizeof(attn_v_cache_f16),
                        attn_v_cache_f16);
        fails += check_status(s, GEIST_OK, "attention block buffers upload OK");

        struct geist_tensor t_res = tensor_f32_2d(res_b, 1, attn_d);
        struct geist_tensor t_norm = tensor_f32_1d(norm_b, attn_d);
        struct geist_tensor t_q_norm = tensor_f32_1d(q_norm_b, attn_hd);
        struct geist_tensor t_k_norm = tensor_f32_1d(k_norm_b, attn_hd);
        struct geist_tensor t_v_norm = tensor_f32_1d(v_norm_b, attn_hd);
        struct geist_tensor t_post_norm = tensor_f32_1d(post_norm_b, attn_d);
        struct geist_tensor t_cos = tensor_f32_2d(cos_b, 1, attn_hd);
        struct geist_tensor t_sin = tensor_f32_2d(sin_b, 1, attn_hd);
        struct geist_tensor t_q_w = tensor_q4k_2d(q_w_b, attn_q_out, attn_d);
        struct geist_tensor t_k_w = tensor_q4k_2d(k_w_b, attn_kv_out, attn_d);
        struct geist_tensor t_v_w = tensor_q4k_2d(v_w_b, attn_kv_out, attn_d);
        struct geist_tensor t_o_w = tensor_q4k_2d(o_w_b, attn_d, attn_q_out);
        if (s == GEIST_OK) {
            s = v->prepare_weight_layout(be, &t_q_w);
        }
        if (s == GEIST_OK) {
            s = v->prepare_weight_layout(be, &t_k_w);
        }
        if (s == GEIST_OK) {
            s = v->prepare_weight_layout(be, &t_v_w);
        }
        if (s == GEIST_OK) {
            s = v->prepare_weight_layout(be, &t_o_w);
        }
        fails += check_status(s, GEIST_OK,
                              "attention block weight layouts prepare OK");
        struct geist_tensor t_k_cache =
            tensor_f32_3d(k_cache_b, attn_kv_len, attn_kv_heads, attn_hd);
        struct geist_tensor t_v_cache =
            tensor_f32_3d(v_cache_b, attn_kv_len, attn_kv_heads, attn_hd);
        struct geist_tensor t_normed = tensor_f32_2d(normed_b, 1, attn_d);
        struct geist_tensor t_q = tensor_f32_2d(q_b, 1, attn_q_out);
        struct geist_tensor t_k = tensor_f32_2d(k_b, 1, attn_kv_out);
        struct geist_tensor t_v = tensor_f32_2d(v_b, 1, attn_kv_out);
        struct geist_tensor t_mid = tensor_f32_2d(mid_b, 1, attn_q_out);
        struct geist_tensor t_o = tensor_f32_2d(o_b, 1, attn_d);
        struct geist_tensor t_post = tensor_f32_2d(post_b, 1, attn_d);
        struct geist_tensor t_out = tensor_f32_2d(out_b, 1, attn_d);
        const struct geist_backend_attention_block attn_block = {
            .struct_size = sizeof(attn_block),
            .q_position = 1,
            .kv_len = attn_kv_len,
            .d_model = attn_d,
            .q_heads = attn_q_heads,
            .kv_heads = attn_kv_heads,
            .head_dim = attn_hd,
            .sliding_window = 0,
            .eps = 1e-6f,
            .residual = &t_res,
            .attn_norm_weight = &t_norm,
            .q_proj_weight = &t_q_w,
            .k_proj_weight = &t_k_w,
            .v_proj_weight = &t_v_w,
            .q_norm_weight = &t_q_norm,
            .k_norm_weight = &t_k_norm,
            .v_norm_weight = &t_v_norm,
            .cos = &t_cos,
            .sin = &t_sin,
            .k_cache = &t_k_cache,
            .v_cache = &t_v_cache,
            .o_proj_weight = &t_o_w,
            .post_attn_norm_weight = &t_post_norm,
            .normed_scratch = &t_normed,
            .q_scratch = &t_q,
            .k_scratch = &t_k,
            .v_scratch = &t_v,
            .attn_scratch = &t_mid,
            .o_scratch = &t_o,
            .post_attn_scratch = &t_post,
            .out = &t_out,
        };
        if (s == GEIST_OK) {
            s = v->attention_block(be, &attn_block);
        }
        if (s != GEIST_OK) {
            fprintf(stderr, "metal attention error: %s\n",
                    geist_backend_errmsg(be));
        }
        fails += check_status(s, GEIST_OK, "attention block runs");
        if (s == GEIST_OK) {
            s = v->buffer_download(sizeof(attn_got), (uint8_t *) attn_got,
                                   out_b);
        }
        fails += check_status(s, GEIST_OK, "attention block output download OK");
        if (s == GEIST_OK) {
            s = v->buffer_download(sizeof(attn_k_cache),
                                   (uint8_t *) attn_k_cache, k_cache_b);
        }
        if (s == GEIST_OK) {
            s = v->buffer_download(sizeof(attn_v_cache),
                                   (uint8_t *) attn_v_cache, v_cache_b);
        }
        fails += check_status(s, GEIST_OK, "attention block cache download OK");
        if (s == GEIST_OK) {
            ptrdiff_t bad = geist_fp32_close_array(
                attn_got, attn_ref, attn_d, 2e-3f, 5e-4f);
            if (bad >= 0) {
                fprintf(stderr,
                        "FAIL: attention block[%td]: got %.7f want %.7f\n",
                        bad, (double) attn_got[bad],
                        (double) attn_ref[bad]);
                fails++;
            }
            bad = geist_fp32_close_array(
                attn_k_cache + attn_kv_out, attn_appended_k, attn_kv_out,
                2e-3f, 5e-4f);
            if (bad >= 0) {
                fprintf(stderr,
                        "FAIL: attention k append[%td]: got %.7f want %.7f\n",
                        bad, (double) attn_k_cache[attn_kv_out + bad],
                        (double) attn_appended_k[bad]);
                fails++;
            }
            bad = geist_fp32_close_array(
                attn_v_cache + attn_kv_out, attn_appended_v, attn_kv_out,
                2e-3f, 5e-4f);
            if (bad >= 0) {
                fprintf(stderr,
                        "FAIL: attention v append[%td]: got %.7f want %.7f\n",
                        bad, (double) attn_v_cache[attn_kv_out + bad],
                        (double) attn_appended_v[bad]);
                fails++;
            }
        }

        struct geist_tensor t_k_cache_f16 =
            tensor_f16_3d(k_cache_f16_b, attn_kv_len, attn_kv_heads, attn_hd);
        struct geist_tensor t_v_cache_f16 =
            tensor_f16_3d(v_cache_f16_b, attn_kv_len, attn_kv_heads, attn_hd);
        struct geist_tensor t_out_f16 = tensor_f32_2d(out_f16_b, 1, attn_d);
        struct geist_backend_attention_block attn_f16_block = attn_block;
        attn_f16_block.k_cache = &t_k_cache_f16;
        attn_f16_block.v_cache = &t_v_cache_f16;
        attn_f16_block.out = &t_out_f16;
        if (s == GEIST_OK) {
            s = v->attention_block(be, &attn_f16_block);
        }
        if (s != GEIST_OK) {
            fprintf(stderr, "metal F16 KV attention error: %s\n",
                    geist_backend_errmsg(be));
        }
        fails += check_status(s, GEIST_OK, "attention block F16 KV runs");
        if (s == GEIST_OK) {
            s = v->buffer_download(sizeof(attn_f16_got),
                                   (uint8_t *) attn_f16_got, out_f16_b);
        }
        fails += check_status(s, GEIST_OK,
                              "attention block F16 KV output download OK");
        if (s == GEIST_OK) {
            s = v->buffer_download(sizeof(attn_k_cache_f16_bits),
                                   (uint8_t *) attn_k_cache_f16_bits,
                                   k_cache_f16_b);
        }
        if (s == GEIST_OK) {
            s = v->buffer_download(sizeof(attn_v_cache_f16_bits),
                                   (uint8_t *) attn_v_cache_f16_bits,
                                   v_cache_f16_b);
        }
        fails += check_status(s, GEIST_OK,
                              "attention block F16 KV cache download OK");
        if (s == GEIST_OK) {
            decode_f16_array(attn_kv_len * attn_kv_out,
                             attn_k_cache_f16_bits, attn_k_cache_f16_got);
            decode_f16_array(attn_kv_len * attn_kv_out,
                             attn_v_cache_f16_bits, attn_v_cache_f16_got);
            ptrdiff_t bad = geist_fp32_close_array(
                attn_f16_got, attn_ref, attn_d, 6e-3f, 1e-3f);
            if (bad >= 0) {
                fprintf(stderr,
                        "FAIL: attention F16 KV block[%td]: got %.7f want %.7f\n",
                        bad, (double) attn_f16_got[bad],
                        (double) attn_ref[bad]);
                fails++;
            }
            bad = geist_fp32_close_array(
                attn_k_cache_f16_got + attn_kv_out,
                attn_appended_k_f16_ref, attn_kv_out, 1e-5f, 1e-5f);
            if (bad >= 0) {
                fprintf(stderr,
                        "FAIL: attention F16 k append[%td]: got %.7f want %.7f\n",
                        bad,
                        (double) attn_k_cache_f16_got[attn_kv_out + bad],
                        (double) attn_appended_k_f16_ref[bad]);
                fails++;
            }
            bad = geist_fp32_close_array(
                attn_v_cache_f16_got + attn_kv_out,
                attn_appended_v_f16_ref, attn_kv_out, 1e-5f, 1e-5f);
            if (bad >= 0) {
                fprintf(stderr,
                        "FAIL: attention F16 v append[%td]: got %.7f want %.7f\n",
                        bad,
                        (double) attn_v_cache_f16_got[attn_kv_out + bad],
                        (double) attn_appended_v_f16_ref[bad]);
                fails++;
            }
        }

#undef CREATE_ATTN_BUF
#undef UPLOAD_ATTN_BUF
        if (res_b != nullptr) { v->buffer_destroy(be, res_b); }
        if (norm_b != nullptr) { v->buffer_destroy(be, norm_b); }
        if (q_norm_b != nullptr) { v->buffer_destroy(be, q_norm_b); }
        if (k_norm_b != nullptr) { v->buffer_destroy(be, k_norm_b); }
        if (v_norm_b != nullptr) { v->buffer_destroy(be, v_norm_b); }
        if (post_norm_b != nullptr) { v->buffer_destroy(be, post_norm_b); }
        if (cos_b != nullptr) { v->buffer_destroy(be, cos_b); }
        if (sin_b != nullptr) { v->buffer_destroy(be, sin_b); }
        if (q_w_b != nullptr) { v->buffer_destroy(be, q_w_b); }
        if (k_w_b != nullptr) { v->buffer_destroy(be, k_w_b); }
        if (v_w_b != nullptr) { v->buffer_destroy(be, v_w_b); }
        if (o_w_b != nullptr) { v->buffer_destroy(be, o_w_b); }
        if (k_cache_b != nullptr) { v->buffer_destroy(be, k_cache_b); }
        if (v_cache_b != nullptr) { v->buffer_destroy(be, v_cache_b); }
        if (k_cache_f16_b != nullptr) { v->buffer_destroy(be, k_cache_f16_b); }
        if (v_cache_f16_b != nullptr) { v->buffer_destroy(be, v_cache_f16_b); }
        if (normed_b != nullptr) { v->buffer_destroy(be, normed_b); }
        if (q_b != nullptr) { v->buffer_destroy(be, q_b); }
        if (k_b != nullptr) { v->buffer_destroy(be, k_b); }
        if (v_b != nullptr) { v->buffer_destroy(be, v_b); }
        if (mid_b != nullptr) { v->buffer_destroy(be, mid_b); }
        if (o_b != nullptr) { v->buffer_destroy(be, o_b); }
        if (post_b != nullptr) { v->buffer_destroy(be, post_b); }
        if (out_b != nullptr) { v->buffer_destroy(be, out_b); }
        if (out_f16_b != nullptr) { v->buffer_destroy(be, out_f16_b); }
        s = GEIST_OK;
    }

    if (s == GEIST_OK) {
        memset(y_matmul_got, 0, sizeof(y_matmul_got));
        s = v->buffer_upload(ymb, sizeof(y_matmul_got),
                             (const uint8_t *) y_matmul_got);
    }
    if (s == GEIST_OK) {
        s = v->command_sequence_begin(
            be, GEIST_COMMAND_SEQUENCE_VERIFY_GREEDY, &seq_token);
    }
    if (s == GEIST_OK) {
        s = v->matmul_q4k(be, &txm, &tw, &tym);
    }
    if (s == GEIST_OK) {
        s = v->command_sequence_end(be, seq_token, true);
    }
    fails += check_status(s, GEIST_OK,
                          "Q4_K matmul command sequence submit OK");
    if (s == GEIST_OK) {
        memset(y_matmul_got, 0, sizeof(y_matmul_got));
        s = v->buffer_download(sizeof(y_matmul_got),
                               (uint8_t *) y_matmul_got, ymb);
    }
    fails += check_status(s, GEIST_OK,
                          "Q4_K matmul sequence output download OK");
    if (s == GEIST_OK) {
        ptrdiff_t bad = geist_fp32_close_array(
            y_matmul_got, y_matmul_ref, matmul_rows * n_out, 2e-4f, 5e-5f);
        if (bad >= 0) {
            fprintf(stderr,
                    "FAIL: Q4_K sequence matmul[%td]: got %.7f want %.7f\n",
                    bad, (double) y_matmul_got[bad],
                    (double) y_matmul_ref[bad]);
            fails++;
        }
    }

    float y_zero[n_out];
    float y_abort[n_out];
    for (size_t i = 0; i < n_out; i++) {
        y_zero[i] = 0.0f;
        y_abort[i] = -123.0f - (float) i;
    }

    s = v->command_sequence_begin(
        be, GEIST_COMMAND_SEQUENCE_DECODE_LAYER_LOOP, &seq_token);
    fails += check_status(s, GEIST_OK,
                          "metal decode layer sequence begin OK");
    if (s == GEIST_OK) {
        s = v->command_sequence_end(be, seq_token, false);
    }
    fails += check_status(s, GEIST_OK,
                          "metal decode layer sequence cleanup OK");

    s = v->command_sequence_begin(
        be, GEIST_COMMAND_SEQUENCE_DECODE_GREEDY_STEP, &seq_token);
    fails += check_status(s, GEIST_OK,
                          "metal decode greedy sequence begin OK");
    if (s == GEIST_OK) {
        s = v->command_sequence_end(be, seq_token, false);
    }
    fails += check_status(s, GEIST_OK,
                          "metal decode greedy sequence cleanup OK");

    s = v->command_sequence_begin(
        be, GEIST_COMMAND_SEQUENCE_VERIFY_GREEDY, &seq_token);
    fails += check_status(s, GEIST_OK, "command sequence begin OK");
    if (s == GEIST_OK) {
        enum geist_status nested = v->command_sequence_begin(
            be, GEIST_COMMAND_SEQUENCE_VERIFY_GREEDY, &nested_token);
        fails += check_status(nested, GEIST_E_INVALID_ARG,
                              "command sequence rejects nesting");
        s = v->command_sequence_end(be, seq_token, false);
        fails += check_status(s, GEIST_OK,
                              "command sequence cleanup after nesting OK");
    }

    seq_token = 0;
    s = v->command_sequence_begin(
        be, GEIST_COMMAND_SEQUENCE_VERIFY_GREEDY, &seq_token);
    fails += check_status(s, GEIST_OK,
                          "command sequence begin for token test OK");
    if (s == GEIST_OK) {
        enum geist_status bad_end =
            v->command_sequence_end(be, seq_token + 1, false);
        fails += check_status(bad_end, GEIST_E_INVALID_ARG,
                              "command sequence rejects invalid token");
        s = v->command_sequence_end(be, seq_token, false);
        fails += check_status(s, GEIST_OK,
                              "command sequence cleanup after bad token OK");
    }

    s = v->buffer_upload(yb, sizeof(y_zero), (const uint8_t *) y_zero);
    if (s == GEIST_OK) {
        s = v->command_sequence_begin(
            be, GEIST_COMMAND_SEQUENCE_VERIFY_GREEDY, &seq_token);
    }
    if (s == GEIST_OK) {
        s = v->matvec_q4k(be, &tx, &tw, &ty);
    }
    if (s == GEIST_OK) {
        s = v->command_sequence_end(be, seq_token, true);
    }
    fails += check_status(s, GEIST_OK,
                          "Q4_K matvec command sequence submit OK");
    if (s == GEIST_OK) {
        memset(y_got, 0, sizeof(y_got));
        s = v->buffer_download(sizeof(y_got), (uint8_t *) y_got, yb);
    }
    fails += check_status(s, GEIST_OK,
                          "Q4_K sequence output download OK");
    if (s == GEIST_OK) {
        ptrdiff_t bad =
            geist_fp32_close_array(y_got, y_ref, n_out, 2e-4f, 4e-5f);
        if (bad >= 0) {
            fprintf(stderr,
                    "FAIL: Q4_K sequence matvec[%td]: got %.7f want %.7f\n",
                    bad, (double) y_got[bad], (double) y_ref[bad]);
            fails++;
        }
    }

    s = v->buffer_upload(yb, sizeof(y_abort), (const uint8_t *) y_abort);
    if (s == GEIST_OK) {
        s = v->command_sequence_begin(
            be, GEIST_COMMAND_SEQUENCE_VERIFY_GREEDY, &seq_token);
    }
    if (s == GEIST_OK) {
        s = v->matvec_q4k(be, &tx, &tw, &ty);
    }
    if (s == GEIST_OK) {
        s = v->command_sequence_end(be, seq_token, false);
    }
    fails += check_status(s, GEIST_OK,
                          "Q4_K matvec command sequence abort OK");
    if (s == GEIST_OK) {
        memset(y_got, 0, sizeof(y_got));
        s = v->buffer_download(sizeof(y_got), (uint8_t *) y_got, yb);
    }
    fails += check_status(s, GEIST_OK,
                          "Q4_K aborted output download OK");
    for (size_t i = 0; s == GEIST_OK && i < n_out; i++) {
        if (y_got[i] != y_abort[i]) {
            fprintf(stderr,
                    "FAIL: Q4_K aborted sequence wrote y[%zu]: got %.7f want %.7f\n",
                    i, (double) y_got[i], (double) y_abort[i]);
            fails++;
            break;
        }
    }

    struct geist_tensor bad_x = tensor_f32_2d(xb, 2, n_in / 2);
    s = v->matvec_q4k(be, &bad_x, &tw, &ty);
    fails += check_status(s, GEIST_E_UNSUPPORTED,
                          "Q4_K matvec rejects batched x");

    s = v->matmul_q4k(be, &tx, &tw, &tym);
    fails += check_status(s, GEIST_E_UNSUPPORTED,
                          "Q4_K matmul rejects vector x");

    struct geist_tensor bad_w = tensor_q4k_2d(wb, n_out, n_in - 1);
    s = v->matvec_q4k(be, &tx, &bad_w, &ty);
    fails += check_status(s, GEIST_E_UNSUPPORTED,
                          "Q4_K matvec rejects unaligned weight width");

    s = v->matmul_q4k(be, &txm, &bad_w, &tym);
    fails += check_status(s, GEIST_E_UNSUPPORTED,
                          "Q4_K matmul rejects unaligned weight width");

    if (xb != nullptr) { v->buffer_destroy(be, xb); }
    if (wb != nullptr) { v->buffer_destroy(be, wb); }
    if (yb != nullptr) { v->buffer_destroy(be, yb); }
    if (emb_yb != nullptr) { v->buffer_destroy(be, emb_yb); }
    if (xmb != nullptr) { v->buffer_destroy(be, xmb); }
    if (ymb != nullptr) { v->buffer_destroy(be, ymb); }
    if (ffn_residual_b != nullptr) { v->buffer_destroy(be, ffn_residual_b); }
    if (ffn_norm_b != nullptr) { v->buffer_destroy(be, ffn_norm_b); }
    if (ffn_post_norm_b != nullptr) {
        v->buffer_destroy(be, ffn_post_norm_b);
    }
    if (ffn_gate_w_b != nullptr) { v->buffer_destroy(be, ffn_gate_w_b); }
    if (ffn_up_w_b != nullptr) { v->buffer_destroy(be, ffn_up_w_b); }
    if (ffn_down_w_b != nullptr) { v->buffer_destroy(be, ffn_down_w_b); }
    if (ffn_down_q6_w_b != nullptr) {
        v->buffer_destroy(be, ffn_down_q6_w_b);
    }
    if (ffn_pre_b != nullptr) { v->buffer_destroy(be, ffn_pre_b); }
    if (ffn_gate_b != nullptr) { v->buffer_destroy(be, ffn_gate_b); }
    if (ffn_up_b != nullptr) { v->buffer_destroy(be, ffn_up_b); }
    if (ffn_out_b != nullptr) { v->buffer_destroy(be, ffn_out_b); }
    if (ffn_post_b != nullptr) { v->buffer_destroy(be, ffn_post_b); }
    if (ffn_dst_b != nullptr) { v->buffer_destroy(be, ffn_dst_b); }
    geist_backend_destroy(be);
    if (pack_cache_path != nullptr) {
        const struct cache_file_counts pack_files =
            count_and_remove_cache_files(pack_cache_path);
        fails += check(pack_files.q4k > 0,
                       "metal Q4_K pack cache writes at least one file");
        fails += check(pack_files.q6k > 0,
                       "metal Q6_K pack cache writes at least one file");
        (void) rmdir(pack_cache_path);
    }

    if (fails == 0) {
        printf("PASS: backend metal Q4_K matvec\n");
        return GEIST_TEST_PASS;
    }
    fprintf(stderr, "FAILED: %d check(s)\n", fails);
    return GEIST_TEST_FAIL;
}
