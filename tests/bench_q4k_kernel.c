/*
 * bench_q4k_kernel — focused microbench for the Q4_K W4A8 hot loop.
 *
 * Loads one or more Q4_K weight tensors from a real GGUF, pre-quantizes
 * an input vector ONCE, then loops `linear_q4k_decode_w4a8_pre` enough
 * times to get stable timing (≥ 200ms total per shape).
 *
 * Reports:
 *   ms/call          — average per-call wall time
 *   GB/s effective   — Q4_K weight bytes read per call / time
 *   cyc/byte (est)   — at a fixed clock; only meaningful when wrapped in `perf stat`
 *
 * Usage:
 *   bench_q4k_kernel <gguf>                 — bench all five canonical shapes
 *   bench_q4k_kernel <gguf> <tensor_name>   — bench just one
 *
 * On Pi 5, wrap in:
 *   perf stat -e cycles,instructions,L1-dcache-loads,L1-dcache-load-misses,branch-misses \
 *     -- ./bench_q4k_kernel gemma4-e2b-Q4_K_M.gguf blk.0.attn_q.weight
 *
 * Threading: single-threaded by default (set OMP_NUM_THREADS=1) so the
 * microbench measures inner-kernel efficiency, not OMP scheduling.
 */
#include "gguf_reader.h"
#include "gguf_quant.h"
#include "gemma4_kernels.h"
#include "test_utils.h"

#include <geist_backend.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static struct geist_tensor bench_tensor_2d(struct geist_buffer *buf,
                                           enum geist_dtype dtype,
                                           enum geist_layout layout,
                                           size_t rows,
                                           size_t cols) {
    return (struct geist_tensor){
        .buffer = buf,
        .dtype = dtype,
        .layout = layout,
        .ndim = 2,
        .shape = {(int64_t) rows, (int64_t) cols, 0, 0},
        .stride = {(int64_t) cols, 1, 0, 0},
    };
}

static struct geist_tensor bench_tensor_3d(struct geist_buffer *buf,
                                           enum geist_dtype dtype,
                                           enum geist_layout layout,
                                           size_t d0,
                                           size_t d1,
                                           size_t d2) {
    return (struct geist_tensor){
        .buffer = buf,
        .dtype = dtype,
        .layout = layout,
        .ndim = 3,
        .shape = {(int64_t) d0, (int64_t) d1, (int64_t) d2, 0},
        .stride = {(int64_t) (d1 * d2), (int64_t) d2, 1, 0},
    };
}

static void bench_pack_q4k_matrix(size_t n_in, size_t n_out, uint8_t *dst) {
    const size_t blocks_per_row = n_in / Q4_K_BLOCK_ELEMS;
    for (size_t row = 0; row < n_out; row++) {
        for (size_t block = 0; block < blocks_per_row; block++) {
            uint8_t *b = dst + (row * blocks_per_row + block) * Q4_K_BLOCK_BYTES;
            memset(b, 0, Q4_K_BLOCK_BYTES);
            b[0] = 0x00u;
            b[1] = 0x3cu;
            b[2] = 0x00u;
            b[3] = 0x00u;
            b[4] = 1u;
            b[5] = 1u;
            b[6] = 1u;
            b[7] = 1u;
            b[12] = 1u;
            b[13] = 1u;
            b[14] = 1u;
            b[15] = 1u;
            for (size_t pair = 0; pair < 4u; pair++) {
                for (size_t idx = 0; idx < 32u; idx++) {
                    const uint8_t lo =
                        (uint8_t) ((row * 3u + block * 5u +
                                    (pair * 2u) * 7u + idx) & 15u);
                    const uint8_t hi =
                        (uint8_t) ((row * 3u + block * 5u +
                                    (pair * 2u + 1u) * 7u + idx) & 15u);
                    b[16u + pair * 32u + idx] =
                        (uint8_t) (lo | (uint8_t) (hi << 4u));
                }
            }
        }
    }
}

static void bench_vulkan_qk(const struct gguf_tensor_t *t,
                            const char *name,
                            size_t rows,
                            size_t weight_bytes,
                            enum geist_dtype dtype,
                            const char *label) {
    if (getenv("GEIST_Q4K_BENCH_VULKAN") == NULL) {
        return;
    }
    if (t == NULL || rows == 0) {
        return;
    }

    const size_t n_in = t->dims[0];
    const size_t n_out = t->dims[1];
    struct geist_backend *be = NULL;
    enum geist_status s = geist_backend_create("vulkan", NULL, NULL, &be);
    if (s != GEIST_OK) {
        fprintf(stderr, "  %-32s [Vulkan %s m=%zu] SKIP (%s)\n",
                name, label, rows, geist_last_create_error());
        return;
    }
    const struct geist_backend_vtbl *v = be->desc->vtbl;
    const bool is_q4 = dtype == GEIST_DTYPE_Q4_K;
    if (v == NULL ||
        (is_q4 && v->matmul_q4k == NULL) ||
        (!is_q4 && v->matmul_q6k == NULL)) {
        fprintf(stderr, "  %-32s [Vulkan %s m=%zu] SKIP (matmul missing)\n",
                name, label, rows);
        geist_backend_destroy(be);
        return;
    }

    float *x_host = (float *) aligned_alloc(64, rows * n_in * sizeof(float));
    float *y_host = (float *) aligned_alloc(64, rows * n_out * sizeof(float));
    struct geist_buffer *x = NULL;
    struct geist_buffer *w = NULL;
    struct geist_buffer *y = NULL;
    if (x_host == NULL || y_host == NULL) {
        fprintf(stderr, "  %-32s [Vulkan %s m=%zu] SKIP (host alloc fail)\n",
                name, label, rows);
        goto done;
    }
    for (size_t r = 0; r < rows; r++) {
        for (size_t i = 0; i < n_in; i++) {
            x_host[r * n_in + i] =
                ((float) ((r * 13u + i) % 4096u) * 0.0019f) - 3.9f;
        }
    }

    s = v->buffer_create(be, rows * n_in * sizeof(float),
                         GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_DEVICE, &x);
    if (s == GEIST_OK) {
        s = v->buffer_create(be, weight_bytes, GEIST_BUFFER_WEIGHT,
                             GEIST_MEMORY_DEVICE, &w);
    }
    if (s == GEIST_OK) {
        s = v->buffer_create(be, rows * n_out * sizeof(float),
                             GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_DEVICE, &y);
    }
    if (s != GEIST_OK) {
        fprintf(stderr, "  %-32s [Vulkan %s m=%zu] SKIP (buffer_create: %s)\n",
                name, label, rows, geist_backend_errmsg(be));
        goto done;
    }
    s = v->buffer_upload(x, rows * n_in * sizeof(float),
                         (const uint8_t *) x_host);
    if (s == GEIST_OK) {
        s = v->buffer_upload(w, weight_bytes, (const uint8_t *) t->data);
    }
    if (s != GEIST_OK) {
        fprintf(stderr, "  %-32s [Vulkan %s m=%zu] SKIP (upload: %s)\n",
                name, label, rows, geist_backend_errmsg(be));
        goto done;
    }

    struct geist_tensor tx =
        bench_tensor_2d(x, GEIST_DTYPE_F32, GEIST_LAYOUT_DENSE, rows, n_in);
    struct geist_tensor tw =
        bench_tensor_2d(w, dtype, GEIST_LAYOUT_BLOCK_QUANTIZED, n_out, n_in);
    struct geist_tensor ty =
        bench_tensor_2d(y, GEIST_DTYPE_F32, GEIST_LAYOUT_DENSE, rows, n_out);

    s = is_q4 ? v->matmul_q4k(be, &tx, &tw, &ty) :
                v->matmul_q6k(be, &tx, &tw, &ty);
    if (s != GEIST_OK) {
        fprintf(stderr, "  %-32s [Vulkan %s m=%zu] SKIP (matmul: %s)\n",
                name, label, rows, geist_backend_errmsg(be));
        goto done;
    }
    const double twarm = now_ms();
    s = is_q4 ? v->matmul_q4k(be, &tx, &tw, &ty) :
                v->matmul_q6k(be, &tx, &tw, &ty);
    const double single_ms = now_ms() - twarm;
    if (s != GEIST_OK) {
        fprintf(stderr, "  %-32s [Vulkan %s m=%zu] SKIP (matmul warm: %s)\n",
                name, label, rows, geist_backend_errmsg(be));
        goto done;
    }

    int n_iter = (int) (200.0 / (single_ms + 0.001)) + 3;
    if (n_iter > 1000) {
        n_iter = 1000;
    }
    if (n_iter < 3) {
        n_iter = 3;
    }
    const double t0 = now_ms();
    for (int it = 0; it < n_iter; it++) {
        s = is_q4 ? v->matmul_q4k(be, &tx, &tw, &ty) :
                    v->matmul_q6k(be, &tx, &tw, &ty);
        if (s != GEIST_OK) {
            break;
        }
    }
    const double dt_ms = (now_ms() - t0) / n_iter;
    if (s != GEIST_OK) {
        fprintf(stderr, "  %-32s [Vulkan %s m=%zu] FAIL (matmul: %s)\n",
                name, label, rows, geist_backend_errmsg(be));
        goto done;
    }
    s = v->buffer_download(rows * n_out * sizeof(float),
                           (uint8_t *) y_host, y);
    if (s != GEIST_OK) {
        fprintf(stderr, "  %-32s [Vulkan %s m=%zu] FAIL (download: %s)\n",
                name, label, rows, geist_backend_errmsg(be));
        goto done;
    }
    const double logical_gbps =
        ((double) weight_bytes * (double) rows) / (dt_ms * 1e6);
    printf("  %-32s [Vulkan %s m=%zu] logical %7.1f MB  %7.2f ms  %5.2f GB/s  (%d it)\n",
           name,
           label,
           rows,
           ((double) weight_bytes * (double) rows) / (1024.0 * 1024.0),
           dt_ms,
           logical_gbps,
           n_iter);
    fflush(stdout);
    if (y_host[0] == 0.0f && y_host[rows * n_out - 1] == 0.0f) {
        fprintf(stderr, "(zero Vulkan output)\n");
    }

done:
    if (x != NULL) { v->buffer_destroy(be, x); }
    if (w != NULL) { v->buffer_destroy(be, w); }
    if (y != NULL) { v->buffer_destroy(be, y); }
    free(x_host);
    free(y_host);
    geist_backend_destroy(be);
}

static bool bench_attn_sibling_name(const char *name,
                                    const char *suffix,
                                    size_t dst_n,
                                    char dst[static dst_n]) {
    const char *needle = strstr(name, "attn_q.weight");
    if (needle == NULL) {
        return false;
    }
    const size_t prefix_n = (size_t) (needle - name);
    const int n = snprintf(dst, dst_n, "%.*s%s", (int) prefix_n, name, suffix);
    return n > 0 && (size_t) n < dst_n;
}

static bool bench_ffn_sibling_name(const char *name,
                                   const char *suffix,
                                   size_t dst_n,
                                   char dst[static dst_n]) {
    const char *needle = strstr(name, "ffn_gate.weight");
    if (needle == NULL) {
        return false;
    }
    const size_t prefix_n = (size_t) (needle - name);
    const int n = snprintf(dst, dst_n, "%.*s%s", (int) prefix_n, name, suffix);
    return n > 0 && (size_t) n < dst_n;
}

static size_t bench_attn_head_dim(size_t q_out, size_t kv_out) {
    static const size_t candidates[] = {256, 128, 64, 32};
    for (size_t i = 0; i < sizeof(candidates) / sizeof(candidates[0]); i++) {
        const size_t hd = candidates[i];
        if (q_out % hd == 0 && kv_out % hd == 0) {
            return hd;
        }
    }
    return 0;
}

static void bench_metal_attention_q4k(const struct gguf_ctx *ctx,
                                      const struct gguf_tensor_t *q_t,
                                      const char *name) {
    if (getenv("GEIST_Q4K_BENCH_METAL_ATTN") == NULL) {
        return;
    }
    if (ctx == NULL || q_t == NULL || q_t->dtype != GGUF_TYPE_Q4_K ||
        strstr(name, "attn_q.weight") == NULL) {
        return;
    }

    char k_name[256];
    char v_name[256];
    char o_name[256];
    if (!bench_attn_sibling_name(name, "attn_k.weight",
                                 sizeof(k_name), k_name) ||
        !bench_attn_sibling_name(name, "attn_v.weight",
                                 sizeof(v_name), v_name) ||
        !bench_attn_sibling_name(name, "attn_output.weight",
                                 sizeof(o_name), o_name)) {
        return;
    }

    const struct gguf_tensor_t *k_t = gguf_get_tensor(ctx, k_name);
    const struct gguf_tensor_t *v_t = gguf_get_tensor(ctx, v_name);
    const struct gguf_tensor_t *o_t = gguf_get_tensor(ctx, o_name);
    if (k_t == NULL || v_t == NULL || o_t == NULL ||
        k_t->dtype != GGUF_TYPE_Q4_K ||
        (v_t->dtype != GGUF_TYPE_Q4_K && v_t->dtype != GGUF_TYPE_Q6_K) ||
        o_t->dtype != GGUF_TYPE_Q4_K) {
        fprintf(stderr, "  %-32s [Metal ATTN Q4_K] SKIP (sibling weights missing)\n",
                name);
        return;
    }

    const size_t d_model = q_t->dims[0];
    const size_t q_out = q_t->dims[1];
    const size_t kv_out = k_t->dims[1];
    const size_t o_rows = o_t->dims[1];
    const size_t o_cols = o_t->dims[0];
    if (k_t->dims[0] != d_model || v_t->dims[0] != d_model ||
        v_t->dims[1] != kv_out || o_rows != d_model || o_cols != q_out) {
        fprintf(stderr, "  %-32s [Metal ATTN Q4_K] SKIP (shape mismatch)\n",
                name);
        return;
    }

    const size_t head_dim = bench_attn_head_dim(q_out, kv_out);
    if (head_dim == 0) {
        fprintf(stderr, "  %-32s [Metal ATTN Q4_K] SKIP (head dim unknown)\n",
                name);
        return;
    }
    const size_t q_heads = q_out / head_dim;
    const size_t kv_heads = kv_out / head_dim;
    const size_t kv_len = getenv("GEIST_BENCH_KV") != NULL
                              ? (size_t) atoi(getenv("GEIST_BENCH_KV"))
                              : 128u;
    if (kv_len == 0 || kv_len > 1024u) {
        fprintf(stderr, "  %-32s [Metal ATTN Q4_K] SKIP (GEIST_BENCH_KV must be 1..1024)\n",
                name);
        return;
    }

    struct geist_backend *be = NULL;
    enum geist_status s = geist_backend_create("metal", NULL, NULL, &be);
    if (s != GEIST_OK) {
        fprintf(stderr, "  %-32s [Metal ATTN Q4_K] SKIP (%s)\n",
                name, geist_last_create_error());
        return;
    }
    const struct geist_backend_vtbl *v = be->desc->vtbl;
    if (v == NULL || v->attention_block == NULL) {
        fprintf(stderr, "  %-32s [Metal ATTN Q4_K] SKIP (attention_block missing)\n",
                name);
        geist_backend_destroy(be);
        return;
    }

    float *residual_host = (float *) aligned_alloc(64, d_model * sizeof(float));
    float *norm_host = (float *) aligned_alloc(64, d_model * sizeof(float));
    float *post_norm_host = (float *) aligned_alloc(64, d_model * sizeof(float));
    float *q_norm_host = (float *) aligned_alloc(64, head_dim * sizeof(float));
    float *k_norm_host = (float *) aligned_alloc(64, head_dim * sizeof(float));
    float *v_norm_host = (float *) aligned_alloc(64, head_dim * sizeof(float));
    float *cos_host = (float *) aligned_alloc(64, head_dim * sizeof(float));
    float *sin_host = (float *) aligned_alloc(64, head_dim * sizeof(float));
    float *k_cache_host =
        (float *) aligned_alloc(64, kv_len * kv_out * sizeof(float));
    float *v_cache_host =
        (float *) aligned_alloc(64, kv_len * kv_out * sizeof(float));
    float *out_host = (float *) aligned_alloc(64, d_model * sizeof(float));
    if (residual_host == NULL || norm_host == NULL || post_norm_host == NULL ||
        q_norm_host == NULL || k_norm_host == NULL || v_norm_host == NULL ||
        cos_host == NULL || sin_host == NULL || k_cache_host == NULL ||
        v_cache_host == NULL || out_host == NULL) {
        fprintf(stderr, "  %-32s [Metal ATTN Q4_K] SKIP (host alloc fail)\n",
                name);
        goto done;
    }
    for (size_t i = 0; i < d_model; i++) {
        residual_host[i] =
            ((float) ((i * 19u) % 4096u) * 0.00006103515625f) - 0.125f;
        norm_host[i] = 0.75f + (float) (i % 13u) * 0.001f;
        post_norm_host[i] = 0.5f + (float) (i % 7u) * 0.0015f;
    }
    for (size_t i = 0; i < head_dim; i++) {
        q_norm_host[i] = 0.9f + (float) (i % 5u) * 0.002f;
        k_norm_host[i] = 0.8f + (float) (i % 3u) * 0.002f;
        v_norm_host[i] = 1.0f;
        cos_host[i] = 1.0f;
        sin_host[i] = 0.0f;
    }
    for (size_t i = 0; i < kv_len * kv_out; i++) {
        k_cache_host[i] =
            ((float) ((i * 23u) % 1024u) * 0.0001220703125f) - 0.0625f;
        v_cache_host[i] =
            ((float) ((i * 29u) % 1024u) * 0.0001220703125f) - 0.0625f;
    }

    struct geist_buffer *residual_b = NULL;
    struct geist_buffer *norm_b = NULL;
    struct geist_buffer *q_norm_b = NULL;
    struct geist_buffer *k_norm_b = NULL;
    struct geist_buffer *v_norm_b = NULL;
    struct geist_buffer *post_norm_b = NULL;
    struct geist_buffer *cos_b = NULL;
    struct geist_buffer *sin_b = NULL;
    struct geist_buffer *q_w_b = NULL;
    struct geist_buffer *k_w_b = NULL;
    struct geist_buffer *v_w_b = NULL;
    struct geist_buffer *o_w_b = NULL;
    struct geist_buffer *k_cache_b = NULL;
    struct geist_buffer *v_cache_b = NULL;
    struct geist_buffer *normed_b = NULL;
    struct geist_buffer *q_b = NULL;
    struct geist_buffer *k_b = NULL;
    struct geist_buffer *vv_b = NULL;
    struct geist_buffer *attn_b = NULL;
    struct geist_buffer *o_b = NULL;
    struct geist_buffer *post_b = NULL;
    struct geist_buffer *out_b = NULL;

#define CREATE_BUF(ptr_, bytes_, kind_)                                             \
    do {                                                                            \
        if (s == GEIST_OK) {                                                        \
            s = v->buffer_create(be, (bytes_), (kind_), GEIST_MEMORY_DEVICE, &(ptr_)); \
        }                                                                           \
    } while (0)
#define UPLOAD_BUF(ptr_, bytes_, data_)                                             \
    do {                                                                            \
        if (s == GEIST_OK) {                                                        \
            s = v->buffer_upload((ptr_), (bytes_), (const uint8_t *) (data_));       \
        }                                                                           \
    } while (0)

    CREATE_BUF(residual_b, d_model * sizeof(float), GEIST_BUFFER_ACTIVATION);
    CREATE_BUF(norm_b, d_model * sizeof(float), GEIST_BUFFER_WEIGHT);
    CREATE_BUF(q_norm_b, head_dim * sizeof(float), GEIST_BUFFER_WEIGHT);
    CREATE_BUF(k_norm_b, head_dim * sizeof(float), GEIST_BUFFER_WEIGHT);
    CREATE_BUF(v_norm_b, head_dim * sizeof(float), GEIST_BUFFER_WEIGHT);
    CREATE_BUF(post_norm_b, d_model * sizeof(float), GEIST_BUFFER_WEIGHT);
    CREATE_BUF(cos_b, head_dim * sizeof(float), GEIST_BUFFER_WEIGHT);
    CREATE_BUF(sin_b, head_dim * sizeof(float), GEIST_BUFFER_WEIGHT);
    CREATE_BUF(q_w_b, q_t->nbytes, GEIST_BUFFER_WEIGHT);
    CREATE_BUF(k_w_b, k_t->nbytes, GEIST_BUFFER_WEIGHT);
    CREATE_BUF(v_w_b, v_t->nbytes, GEIST_BUFFER_WEIGHT);
    CREATE_BUF(o_w_b, o_t->nbytes, GEIST_BUFFER_WEIGHT);
    CREATE_BUF(k_cache_b, kv_len * kv_out * sizeof(float), GEIST_BUFFER_KV_CACHE);
    CREATE_BUF(v_cache_b, kv_len * kv_out * sizeof(float), GEIST_BUFFER_KV_CACHE);
    CREATE_BUF(normed_b, d_model * sizeof(float), GEIST_BUFFER_SCRATCH);
    CREATE_BUF(q_b, q_out * sizeof(float), GEIST_BUFFER_SCRATCH);
    CREATE_BUF(k_b, kv_out * sizeof(float), GEIST_BUFFER_SCRATCH);
    CREATE_BUF(vv_b, kv_out * sizeof(float), GEIST_BUFFER_SCRATCH);
    CREATE_BUF(attn_b, q_out * sizeof(float), GEIST_BUFFER_SCRATCH);
    CREATE_BUF(o_b, d_model * sizeof(float), GEIST_BUFFER_SCRATCH);
    CREATE_BUF(post_b, d_model * sizeof(float), GEIST_BUFFER_SCRATCH);
    CREATE_BUF(out_b, d_model * sizeof(float), GEIST_BUFFER_ACTIVATION);

    UPLOAD_BUF(residual_b, d_model * sizeof(float), residual_host);
    UPLOAD_BUF(norm_b, d_model * sizeof(float), norm_host);
    UPLOAD_BUF(q_norm_b, head_dim * sizeof(float), q_norm_host);
    UPLOAD_BUF(k_norm_b, head_dim * sizeof(float), k_norm_host);
    UPLOAD_BUF(v_norm_b, head_dim * sizeof(float), v_norm_host);
    UPLOAD_BUF(post_norm_b, d_model * sizeof(float), post_norm_host);
    UPLOAD_BUF(cos_b, head_dim * sizeof(float), cos_host);
    UPLOAD_BUF(sin_b, head_dim * sizeof(float), sin_host);
    UPLOAD_BUF(q_w_b, q_t->nbytes, q_t->data);
    UPLOAD_BUF(k_w_b, k_t->nbytes, k_t->data);
    UPLOAD_BUF(v_w_b, v_t->nbytes, v_t->data);
    UPLOAD_BUF(o_w_b, o_t->nbytes, o_t->data);
    UPLOAD_BUF(k_cache_b, kv_len * kv_out * sizeof(float), k_cache_host);
    UPLOAD_BUF(v_cache_b, kv_len * kv_out * sizeof(float), v_cache_host);

    if (s != GEIST_OK) {
        fprintf(stderr, "  %-32s [Metal ATTN Q4_K kv=%zu] SKIP (setup: %s)\n",
                name, kv_len, geist_backend_errmsg(be));
        goto attn_done;
    }

    struct geist_tensor t_residual =
        bench_tensor_2d(residual_b, GEIST_DTYPE_F32, GEIST_LAYOUT_DENSE,
                        1, d_model);
    struct geist_tensor t_norm =
        bench_tensor_2d(norm_b, GEIST_DTYPE_F32, GEIST_LAYOUT_DENSE,
                        1, d_model);
    struct geist_tensor t_q_norm =
        bench_tensor_2d(q_norm_b, GEIST_DTYPE_F32, GEIST_LAYOUT_DENSE,
                        1, head_dim);
    struct geist_tensor t_k_norm =
        bench_tensor_2d(k_norm_b, GEIST_DTYPE_F32, GEIST_LAYOUT_DENSE,
                        1, head_dim);
    struct geist_tensor t_v_norm =
        bench_tensor_2d(v_norm_b, GEIST_DTYPE_F32, GEIST_LAYOUT_DENSE,
                        1, head_dim);
    struct geist_tensor t_post_norm =
        bench_tensor_2d(post_norm_b, GEIST_DTYPE_F32, GEIST_LAYOUT_DENSE,
                        1, d_model);
    struct geist_tensor t_cos =
        bench_tensor_2d(cos_b, GEIST_DTYPE_F32, GEIST_LAYOUT_DENSE,
                        1, head_dim);
    struct geist_tensor t_sin =
        bench_tensor_2d(sin_b, GEIST_DTYPE_F32, GEIST_LAYOUT_DENSE,
                        1, head_dim);
    struct geist_tensor t_q_w =
        bench_tensor_2d(q_w_b, GEIST_DTYPE_Q4_K,
                        GEIST_LAYOUT_BLOCK_QUANTIZED, q_out, d_model);
    struct geist_tensor t_k_w =
        bench_tensor_2d(k_w_b, GEIST_DTYPE_Q4_K,
                        GEIST_LAYOUT_BLOCK_QUANTIZED, kv_out, d_model);
    struct geist_tensor t_v_w =
        bench_tensor_2d(v_w_b,
                        v_t->dtype == GGUF_TYPE_Q6_K ? GEIST_DTYPE_Q6_K
                                                     : GEIST_DTYPE_Q4_K,
                        GEIST_LAYOUT_BLOCK_QUANTIZED, kv_out, d_model);
    struct geist_tensor t_o_w =
        bench_tensor_2d(o_w_b, GEIST_DTYPE_Q4_K,
                        GEIST_LAYOUT_BLOCK_QUANTIZED, d_model, q_out);
    struct geist_tensor t_k_cache =
        bench_tensor_3d(k_cache_b, GEIST_DTYPE_F32, GEIST_LAYOUT_DENSE,
                        kv_len, kv_heads, head_dim);
    struct geist_tensor t_v_cache =
        bench_tensor_3d(v_cache_b, GEIST_DTYPE_F32, GEIST_LAYOUT_DENSE,
                        kv_len, kv_heads, head_dim);
    struct geist_tensor t_normed =
        bench_tensor_2d(normed_b, GEIST_DTYPE_F32, GEIST_LAYOUT_DENSE,
                        1, d_model);
    struct geist_tensor t_q =
        bench_tensor_2d(q_b, GEIST_DTYPE_F32, GEIST_LAYOUT_DENSE, 1, q_out);
    struct geist_tensor t_k =
        bench_tensor_2d(k_b, GEIST_DTYPE_F32, GEIST_LAYOUT_DENSE, 1, kv_out);
    struct geist_tensor t_v =
        bench_tensor_2d(vv_b, GEIST_DTYPE_F32, GEIST_LAYOUT_DENSE, 1, kv_out);
    struct geist_tensor t_attn =
        bench_tensor_2d(attn_b, GEIST_DTYPE_F32, GEIST_LAYOUT_DENSE, 1, q_out);
    struct geist_tensor t_o =
        bench_tensor_2d(o_b, GEIST_DTYPE_F32, GEIST_LAYOUT_DENSE, 1, d_model);
    struct geist_tensor t_post =
        bench_tensor_2d(post_b, GEIST_DTYPE_F32, GEIST_LAYOUT_DENSE, 1, d_model);
    struct geist_tensor t_out =
        bench_tensor_2d(out_b, GEIST_DTYPE_F32, GEIST_LAYOUT_DENSE, 1, d_model);
    const struct geist_backend_attention_block block = {
        .struct_size = sizeof(block),
        .q_position = kv_len - 1u,
        .kv_len = kv_len,
        .d_model = d_model,
        .q_heads = q_heads,
        .kv_heads = kv_heads,
        .head_dim = head_dim,
        .sliding_window = 0,
        .eps = 1e-6f,
        .residual = &t_residual,
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
        .attn_scratch = &t_attn,
        .o_scratch = &t_o,
        .post_attn_scratch = &t_post,
        .out = &t_out,
    };

    s = v->attention_block(be, &block);
    if (s == GEIST_OK) {
        const double twarm = now_ms();
        s = v->attention_block(be, &block);
        const double single_ms = now_ms() - twarm;
        int n_iter = (int) (200.0 / (single_ms + 0.001)) + 3;
        if (n_iter > 1000) {
            n_iter = 1000;
        }
        if (n_iter < 3) {
            n_iter = 3;
        }
        const double t0 = now_ms();
        for (int it = 0; it < n_iter; it++) {
            s = v->attention_block(be, &block);
            if (s != GEIST_OK) {
                break;
            }
        }
        const double dt_ms = (now_ms() - t0) / n_iter;
        if (s == GEIST_OK) {
            s = v->buffer_download(d_model * sizeof(float),
                                   (uint8_t *) out_host, out_b);
        }
        if (s == GEIST_OK) {
            const double logical_bytes =
                (double) q_t->nbytes + (double) k_t->nbytes +
                (double) v_t->nbytes + (double) o_t->nbytes +
                (double) kv_len * (double) kv_out * 2.0 * sizeof(float);
            printf("  %-32s [Metal ATTN Q4_K kv=%zu] logical %7.1f MB  %7.2f ms  %5.2f GB/s  (%d it)\n",
                   name,
                   kv_len,
                   logical_bytes / (1024.0 * 1024.0),
                   dt_ms,
                   logical_bytes / (dt_ms * 1e6),
                   n_iter);
            fflush(stdout);
            if (out_host[0] == 0.0f && out_host[d_model - 1] == 0.0f) {
                fprintf(stderr, "(zero Metal attention output)\n");
            }
        }
        if (s == GEIST_OK &&
            getenv("GEIST_Q4K_BENCH_METAL_SEQUENCE") != NULL &&
            v->command_sequence_begin != NULL &&
            v->command_sequence_end != NULL) {
            int seq_iter = n_iter;
            if (seq_iter > 128) {
                seq_iter = 128;
            }
            int token = 0;
            const double ts0 = now_ms();
            s = v->command_sequence_begin(
                be, GEIST_COMMAND_SEQUENCE_DECODE_LAYER_LOOP, &token);
            if (s == GEIST_OK) {
                for (int it = 0; it < seq_iter; it++) {
                    s = v->attention_block(be, &block);
                    if (s != GEIST_OK) {
                        break;
                    }
                }
                enum geist_status es =
                    v->command_sequence_end(be, token, s == GEIST_OK);
                if (s == GEIST_OK) {
                    s = es;
                }
            }
            const double seq_dt_ms = (now_ms() - ts0) / (double) seq_iter;
            if (s == GEIST_OK) {
                const double logical_bytes =
                    (double) q_t->nbytes + (double) k_t->nbytes +
                    (double) v_t->nbytes + (double) o_t->nbytes +
                    (double) kv_len * (double) kv_out * 2.0 * sizeof(float);
                printf("  %-32s [Metal ATTN Q4_K kv=%zu seq] logical %7.1f MB  %7.2f ms  %5.2f GB/s  (%d blocks)\n",
                       name,
                       kv_len,
                       logical_bytes / (1024.0 * 1024.0),
                       seq_dt_ms,
                       logical_bytes / (seq_dt_ms * 1e6),
                       seq_iter);
                fflush(stdout);
            } else {
                fprintf(stderr,
                        "  %-32s [Metal ATTN Q4_K kv=%zu seq] FAIL (%s)\n",
                        name, kv_len, geist_backend_errmsg(be));
            }
        }
    }
    if (s != GEIST_OK) {
        fprintf(stderr, "  %-32s [Metal ATTN Q4_K kv=%zu] FAIL (%s)\n",
                name, kv_len, geist_backend_errmsg(be));
    }

attn_done:
#undef CREATE_BUF
#undef UPLOAD_BUF
    if (residual_b != NULL) { v->buffer_destroy(be, residual_b); }
    if (norm_b != NULL) { v->buffer_destroy(be, norm_b); }
    if (q_norm_b != NULL) { v->buffer_destroy(be, q_norm_b); }
    if (k_norm_b != NULL) { v->buffer_destroy(be, k_norm_b); }
    if (v_norm_b != NULL) { v->buffer_destroy(be, v_norm_b); }
    if (post_norm_b != NULL) { v->buffer_destroy(be, post_norm_b); }
    if (cos_b != NULL) { v->buffer_destroy(be, cos_b); }
    if (sin_b != NULL) { v->buffer_destroy(be, sin_b); }
    if (q_w_b != NULL) { v->buffer_destroy(be, q_w_b); }
    if (k_w_b != NULL) { v->buffer_destroy(be, k_w_b); }
    if (v_w_b != NULL) { v->buffer_destroy(be, v_w_b); }
    if (o_w_b != NULL) { v->buffer_destroy(be, o_w_b); }
    if (k_cache_b != NULL) { v->buffer_destroy(be, k_cache_b); }
    if (v_cache_b != NULL) { v->buffer_destroy(be, v_cache_b); }
    if (normed_b != NULL) { v->buffer_destroy(be, normed_b); }
    if (q_b != NULL) { v->buffer_destroy(be, q_b); }
    if (k_b != NULL) { v->buffer_destroy(be, k_b); }
    if (vv_b != NULL) { v->buffer_destroy(be, vv_b); }
    if (attn_b != NULL) { v->buffer_destroy(be, attn_b); }
    if (o_b != NULL) { v->buffer_destroy(be, o_b); }
    if (post_b != NULL) { v->buffer_destroy(be, post_b); }
    if (out_b != NULL) { v->buffer_destroy(be, out_b); }

done:
    free(residual_host);
    free(norm_host);
    free(post_norm_host);
    free(q_norm_host);
    free(k_norm_host);
    free(v_norm_host);
    free(cos_host);
    free(sin_host);
    free(k_cache_host);
    free(v_cache_host);
    free(out_host);
    geist_backend_destroy(be);
}

static void bench_metal_layer_gemma_q4q6(const struct gguf_ctx *ctx,
                                         const struct gguf_tensor_t *q_t,
                                         const char *name) {
    if (getenv("GEIST_Q4K_BENCH_METAL_LAYER") == NULL) {
        return;
    }
    if (ctx == NULL || q_t == NULL || q_t->dtype != GGUF_TYPE_Q4_K ||
        strstr(name, "attn_q.weight") == NULL) {
        return;
    }

    char k_name[256];
    char v_name[256];
    char o_name[256];
    char gate_name[256];
    char up_name[256];
    char down_name[256];
    if (!bench_attn_sibling_name(name, "attn_k.weight", sizeof(k_name), k_name) ||
        !bench_attn_sibling_name(name, "attn_v.weight", sizeof(v_name), v_name) ||
        !bench_attn_sibling_name(name, "attn_output.weight", sizeof(o_name), o_name) ||
        !bench_attn_sibling_name(name, "ffn_gate.weight", sizeof(gate_name), gate_name) ||
        !bench_attn_sibling_name(name, "ffn_up.weight", sizeof(up_name), up_name) ||
        !bench_attn_sibling_name(name, "ffn_down.weight", sizeof(down_name), down_name)) {
        return;
    }

    const struct gguf_tensor_t *k_t = gguf_get_tensor(ctx, k_name);
    const struct gguf_tensor_t *v_t = gguf_get_tensor(ctx, v_name);
    const struct gguf_tensor_t *o_t = gguf_get_tensor(ctx, o_name);
    const struct gguf_tensor_t *gate_t = gguf_get_tensor(ctx, gate_name);
    const struct gguf_tensor_t *up_t = gguf_get_tensor(ctx, up_name);
    const struct gguf_tensor_t *down_t = gguf_get_tensor(ctx, down_name);
    if (k_t == NULL || v_t == NULL || o_t == NULL ||
        gate_t == NULL || up_t == NULL || down_t == NULL ||
        k_t->dtype != GGUF_TYPE_Q4_K ||
        (v_t->dtype != GGUF_TYPE_Q4_K && v_t->dtype != GGUF_TYPE_Q6_K) ||
        o_t->dtype != GGUF_TYPE_Q4_K ||
        gate_t->dtype != GGUF_TYPE_Q4_K || up_t->dtype != GGUF_TYPE_Q4_K ||
        (down_t->dtype != GGUF_TYPE_Q4_K && down_t->dtype != GGUF_TYPE_Q6_K)) {
        fprintf(stderr, "  %-32s [Metal LAYER Gemma] SKIP (sibling weights missing)\n",
                name);
        return;
    }

    const size_t d_model = q_t->dims[0];
    const size_t q_out = q_t->dims[1];
    const size_t kv_out = k_t->dims[1];
    const size_t inter = gate_t->dims[1];
    if (k_t->dims[0] != d_model || v_t->dims[0] != d_model ||
        v_t->dims[1] != kv_out || o_t->dims[1] != d_model ||
        o_t->dims[0] != q_out || gate_t->dims[0] != d_model ||
        up_t->dims[0] != d_model || up_t->dims[1] != inter ||
        down_t->dims[0] != inter || down_t->dims[1] != d_model) {
        fprintf(stderr, "  %-32s [Metal LAYER Gemma] SKIP (shape mismatch)\n",
                name);
        return;
    }

    const size_t head_dim = bench_attn_head_dim(q_out, kv_out);
    if (head_dim == 0) {
        return;
    }
    const size_t q_heads = q_out / head_dim;
    const size_t kv_heads = kv_out / head_dim;
    const size_t kv_len = getenv("GEIST_BENCH_KV") != NULL
                              ? (size_t) atoi(getenv("GEIST_BENCH_KV"))
                              : 128u;
    if (kv_len == 0 || kv_len > 1024u) {
        return;
    }

    struct geist_backend *be = NULL;
    enum geist_status s = geist_backend_create("metal", NULL, NULL, &be);
    if (s != GEIST_OK) {
        fprintf(stderr, "  %-32s [Metal LAYER Gemma] SKIP (%s)\n",
                name, geist_last_create_error());
        return;
    }
    const struct geist_backend_vtbl *v = be->desc->vtbl;
    if (v == NULL || v->attention_block == NULL || v->ffn_geglu_block == NULL) {
        fprintf(stderr, "  %-32s [Metal LAYER Gemma] SKIP (block ops missing)\n",
                name);
        geist_backend_destroy(be);
        return;
    }

    float *residual_host = (float *) aligned_alloc(64, d_model * sizeof(float));
    float *attn_norm_host = (float *) aligned_alloc(64, d_model * sizeof(float));
    float *attn_post_norm_host = (float *) aligned_alloc(64, d_model * sizeof(float));
    float *ffn_norm_host = (float *) aligned_alloc(64, d_model * sizeof(float));
    float *ffn_post_norm_host = (float *) aligned_alloc(64, d_model * sizeof(float));
    float *q_norm_host = (float *) aligned_alloc(64, head_dim * sizeof(float));
    float *k_norm_host = (float *) aligned_alloc(64, head_dim * sizeof(float));
    float *v_norm_host = (float *) aligned_alloc(64, head_dim * sizeof(float));
    float *cos_host = (float *) aligned_alloc(64, head_dim * sizeof(float));
    float *sin_host = (float *) aligned_alloc(64, head_dim * sizeof(float));
    float *k_cache_host = (float *) aligned_alloc(64, kv_len * kv_out * sizeof(float));
    float *v_cache_host = (float *) aligned_alloc(64, kv_len * kv_out * sizeof(float));
    float *out_host = (float *) aligned_alloc(64, d_model * sizeof(float));
    if (residual_host == NULL || attn_norm_host == NULL ||
        attn_post_norm_host == NULL || ffn_norm_host == NULL ||
        ffn_post_norm_host == NULL || q_norm_host == NULL ||
        k_norm_host == NULL || v_norm_host == NULL || cos_host == NULL ||
        sin_host == NULL || k_cache_host == NULL || v_cache_host == NULL ||
        out_host == NULL) {
        fprintf(stderr, "  %-32s [Metal LAYER Gemma] SKIP (host alloc fail)\n",
                name);
        goto done;
    }

    for (size_t i = 0; i < d_model; i++) {
        residual_host[i] = ((float) ((i * 19u) % 4096u) * 0.00006103515625f) - 0.125f;
        attn_norm_host[i] = 0.75f + (float) (i % 13u) * 0.001f;
        attn_post_norm_host[i] = 0.5f + (float) (i % 7u) * 0.0015f;
        ffn_norm_host[i] = 0.0075f + (float) (i % 7u) * 0.00025f;
        ffn_post_norm_host[i] = 0.25f + (float) (i % 5u) * 0.015625f;
    }
    for (size_t i = 0; i < head_dim; i++) {
        q_norm_host[i] = 0.9f + (float) (i % 5u) * 0.002f;
        k_norm_host[i] = 0.8f + (float) (i % 3u) * 0.002f;
        v_norm_host[i] = 1.0f;
        cos_host[i] = 1.0f;
        sin_host[i] = 0.0f;
    }
    for (size_t i = 0; i < kv_len * kv_out; i++) {
        k_cache_host[i] = ((float) ((i * 23u) % 1024u) * 0.0001220703125f) - 0.0625f;
        v_cache_host[i] = ((float) ((i * 29u) % 1024u) * 0.0001220703125f) - 0.0625f;
    }

    struct geist_buffer *res_b = NULL, *attn_norm_b = NULL, *attn_post_norm_b = NULL;
    struct geist_buffer *ffn_norm_b = NULL, *ffn_post_norm_b = NULL;
    struct geist_buffer *q_norm_b = NULL, *k_norm_b = NULL, *v_norm_b = NULL;
    struct geist_buffer *cos_b = NULL, *sin_b = NULL;
    struct geist_buffer *q_w_b = NULL, *k_w_b = NULL, *v_w_b = NULL, *o_w_b = NULL;
    struct geist_buffer *gate_w_b = NULL, *up_w_b = NULL, *down_w_b = NULL;
    struct geist_buffer *k_cache_b = NULL, *v_cache_b = NULL;
    struct geist_buffer *attn_normed_b = NULL, *q_b = NULL, *k_b = NULL, *vv_b = NULL;
    struct geist_buffer *attn_b = NULL, *o_b = NULL, *attn_post_b = NULL, *attn_out_b = NULL;
    struct geist_buffer *ffn_pre_b = NULL, *gate_s_b = NULL, *up_s_b = NULL;
    struct geist_buffer *ffn_out_b = NULL, *ffn_post_b = NULL, *out_b = NULL;

#define CREATE_LAYER_BUF(ptr_, bytes_, kind_)                                      \
    do {                                                                           \
        if (s == GEIST_OK) {                                                       \
            s = v->buffer_create(be, (bytes_), (kind_), GEIST_MEMORY_DEVICE, &(ptr_)); \
        }                                                                          \
    } while (0)
#define UPLOAD_LAYER_BUF(ptr_, bytes_, data_)                                      \
    do {                                                                           \
        if (s == GEIST_OK) {                                                       \
            s = v->buffer_upload((ptr_), (bytes_), (const uint8_t *) (data_));      \
        }                                                                          \
    } while (0)

    CREATE_LAYER_BUF(res_b, d_model * sizeof(float), GEIST_BUFFER_ACTIVATION);
    CREATE_LAYER_BUF(attn_norm_b, d_model * sizeof(float), GEIST_BUFFER_WEIGHT);
    CREATE_LAYER_BUF(attn_post_norm_b, d_model * sizeof(float), GEIST_BUFFER_WEIGHT);
    CREATE_LAYER_BUF(ffn_norm_b, d_model * sizeof(float), GEIST_BUFFER_WEIGHT);
    CREATE_LAYER_BUF(ffn_post_norm_b, d_model * sizeof(float), GEIST_BUFFER_WEIGHT);
    CREATE_LAYER_BUF(q_norm_b, head_dim * sizeof(float), GEIST_BUFFER_WEIGHT);
    CREATE_LAYER_BUF(k_norm_b, head_dim * sizeof(float), GEIST_BUFFER_WEIGHT);
    CREATE_LAYER_BUF(v_norm_b, head_dim * sizeof(float), GEIST_BUFFER_WEIGHT);
    CREATE_LAYER_BUF(cos_b, head_dim * sizeof(float), GEIST_BUFFER_WEIGHT);
    CREATE_LAYER_BUF(sin_b, head_dim * sizeof(float), GEIST_BUFFER_WEIGHT);
    CREATE_LAYER_BUF(q_w_b, q_t->nbytes, GEIST_BUFFER_WEIGHT);
    CREATE_LAYER_BUF(k_w_b, k_t->nbytes, GEIST_BUFFER_WEIGHT);
    CREATE_LAYER_BUF(v_w_b, v_t->nbytes, GEIST_BUFFER_WEIGHT);
    CREATE_LAYER_BUF(o_w_b, o_t->nbytes, GEIST_BUFFER_WEIGHT);
    CREATE_LAYER_BUF(gate_w_b, gate_t->nbytes, GEIST_BUFFER_WEIGHT);
    CREATE_LAYER_BUF(up_w_b, up_t->nbytes, GEIST_BUFFER_WEIGHT);
    CREATE_LAYER_BUF(down_w_b, down_t->nbytes, GEIST_BUFFER_WEIGHT);
    CREATE_LAYER_BUF(k_cache_b, kv_len * kv_out * sizeof(float), GEIST_BUFFER_KV_CACHE);
    CREATE_LAYER_BUF(v_cache_b, kv_len * kv_out * sizeof(float), GEIST_BUFFER_KV_CACHE);
    CREATE_LAYER_BUF(attn_normed_b, d_model * sizeof(float), GEIST_BUFFER_SCRATCH);
    CREATE_LAYER_BUF(q_b, q_out * sizeof(float), GEIST_BUFFER_SCRATCH);
    CREATE_LAYER_BUF(k_b, kv_out * sizeof(float), GEIST_BUFFER_SCRATCH);
    CREATE_LAYER_BUF(vv_b, kv_out * sizeof(float), GEIST_BUFFER_SCRATCH);
    CREATE_LAYER_BUF(attn_b, q_out * sizeof(float), GEIST_BUFFER_SCRATCH);
    CREATE_LAYER_BUF(o_b, d_model * sizeof(float), GEIST_BUFFER_SCRATCH);
    CREATE_LAYER_BUF(attn_post_b, d_model * sizeof(float), GEIST_BUFFER_SCRATCH);
    CREATE_LAYER_BUF(attn_out_b, d_model * sizeof(float), GEIST_BUFFER_ACTIVATION);
    CREATE_LAYER_BUF(ffn_pre_b, d_model * sizeof(float), GEIST_BUFFER_SCRATCH);
    CREATE_LAYER_BUF(gate_s_b, inter * sizeof(float), GEIST_BUFFER_SCRATCH);
    CREATE_LAYER_BUF(up_s_b, inter * sizeof(float), GEIST_BUFFER_SCRATCH);
    CREATE_LAYER_BUF(ffn_out_b, d_model * sizeof(float), GEIST_BUFFER_SCRATCH);
    CREATE_LAYER_BUF(ffn_post_b, d_model * sizeof(float), GEIST_BUFFER_SCRATCH);
    CREATE_LAYER_BUF(out_b, d_model * sizeof(float), GEIST_BUFFER_ACTIVATION);

    UPLOAD_LAYER_BUF(res_b, d_model * sizeof(float), residual_host);
    UPLOAD_LAYER_BUF(attn_norm_b, d_model * sizeof(float), attn_norm_host);
    UPLOAD_LAYER_BUF(attn_post_norm_b, d_model * sizeof(float), attn_post_norm_host);
    UPLOAD_LAYER_BUF(ffn_norm_b, d_model * sizeof(float), ffn_norm_host);
    UPLOAD_LAYER_BUF(ffn_post_norm_b, d_model * sizeof(float), ffn_post_norm_host);
    UPLOAD_LAYER_BUF(q_norm_b, head_dim * sizeof(float), q_norm_host);
    UPLOAD_LAYER_BUF(k_norm_b, head_dim * sizeof(float), k_norm_host);
    UPLOAD_LAYER_BUF(v_norm_b, head_dim * sizeof(float), v_norm_host);
    UPLOAD_LAYER_BUF(cos_b, head_dim * sizeof(float), cos_host);
    UPLOAD_LAYER_BUF(sin_b, head_dim * sizeof(float), sin_host);
    UPLOAD_LAYER_BUF(q_w_b, q_t->nbytes, q_t->data);
    UPLOAD_LAYER_BUF(k_w_b, k_t->nbytes, k_t->data);
    UPLOAD_LAYER_BUF(v_w_b, v_t->nbytes, v_t->data);
    UPLOAD_LAYER_BUF(o_w_b, o_t->nbytes, o_t->data);
    UPLOAD_LAYER_BUF(gate_w_b, gate_t->nbytes, gate_t->data);
    UPLOAD_LAYER_BUF(up_w_b, up_t->nbytes, up_t->data);
    UPLOAD_LAYER_BUF(down_w_b, down_t->nbytes, down_t->data);
    UPLOAD_LAYER_BUF(k_cache_b, kv_len * kv_out * sizeof(float), k_cache_host);
    UPLOAD_LAYER_BUF(v_cache_b, kv_len * kv_out * sizeof(float), v_cache_host);
    if (s != GEIST_OK) {
        fprintf(stderr, "  %-32s [Metal LAYER Gemma] SKIP (setup: %s)\n",
                name, geist_backend_errmsg(be));
        goto layer_done;
    }

    struct geist_tensor t_res = bench_tensor_2d(res_b, GEIST_DTYPE_F32, GEIST_LAYOUT_DENSE, 1, d_model);
    struct geist_tensor t_attn_norm = bench_tensor_2d(attn_norm_b, GEIST_DTYPE_F32, GEIST_LAYOUT_DENSE, 1, d_model);
    struct geist_tensor t_attn_post_norm = bench_tensor_2d(attn_post_norm_b, GEIST_DTYPE_F32, GEIST_LAYOUT_DENSE, 1, d_model);
    struct geist_tensor t_ffn_norm = bench_tensor_2d(ffn_norm_b, GEIST_DTYPE_F32, GEIST_LAYOUT_DENSE, 1, d_model);
    struct geist_tensor t_ffn_post_norm = bench_tensor_2d(ffn_post_norm_b, GEIST_DTYPE_F32, GEIST_LAYOUT_DENSE, 1, d_model);
    struct geist_tensor t_q_norm = bench_tensor_2d(q_norm_b, GEIST_DTYPE_F32, GEIST_LAYOUT_DENSE, 1, head_dim);
    struct geist_tensor t_k_norm = bench_tensor_2d(k_norm_b, GEIST_DTYPE_F32, GEIST_LAYOUT_DENSE, 1, head_dim);
    struct geist_tensor t_v_norm = bench_tensor_2d(v_norm_b, GEIST_DTYPE_F32, GEIST_LAYOUT_DENSE, 1, head_dim);
    struct geist_tensor t_cos = bench_tensor_2d(cos_b, GEIST_DTYPE_F32, GEIST_LAYOUT_DENSE, 1, head_dim);
    struct geist_tensor t_sin = bench_tensor_2d(sin_b, GEIST_DTYPE_F32, GEIST_LAYOUT_DENSE, 1, head_dim);
    struct geist_tensor t_q_w = bench_tensor_2d(q_w_b, GEIST_DTYPE_Q4_K, GEIST_LAYOUT_BLOCK_QUANTIZED, q_out, d_model);
    struct geist_tensor t_k_w = bench_tensor_2d(k_w_b, GEIST_DTYPE_Q4_K, GEIST_LAYOUT_BLOCK_QUANTIZED, kv_out, d_model);
    struct geist_tensor t_v_w = bench_tensor_2d(v_w_b, v_t->dtype == GGUF_TYPE_Q6_K ? GEIST_DTYPE_Q6_K : GEIST_DTYPE_Q4_K, GEIST_LAYOUT_BLOCK_QUANTIZED, kv_out, d_model);
    struct geist_tensor t_o_w = bench_tensor_2d(o_w_b, GEIST_DTYPE_Q4_K, GEIST_LAYOUT_BLOCK_QUANTIZED, d_model, q_out);
    struct geist_tensor t_gate_w = bench_tensor_2d(gate_w_b, GEIST_DTYPE_Q4_K, GEIST_LAYOUT_BLOCK_QUANTIZED, inter, d_model);
    struct geist_tensor t_up_w = bench_tensor_2d(up_w_b, GEIST_DTYPE_Q4_K, GEIST_LAYOUT_BLOCK_QUANTIZED, inter, d_model);
    struct geist_tensor t_down_w = bench_tensor_2d(down_w_b, down_t->dtype == GGUF_TYPE_Q6_K ? GEIST_DTYPE_Q6_K : GEIST_DTYPE_Q4_K, GEIST_LAYOUT_BLOCK_QUANTIZED, d_model, inter);
    struct geist_tensor t_k_cache = bench_tensor_3d(k_cache_b, GEIST_DTYPE_F32, GEIST_LAYOUT_DENSE, kv_len, kv_heads, head_dim);
    struct geist_tensor t_v_cache = bench_tensor_3d(v_cache_b, GEIST_DTYPE_F32, GEIST_LAYOUT_DENSE, kv_len, kv_heads, head_dim);
    struct geist_tensor t_attn_normed = bench_tensor_2d(attn_normed_b, GEIST_DTYPE_F32, GEIST_LAYOUT_DENSE, 1, d_model);
    struct geist_tensor t_q = bench_tensor_2d(q_b, GEIST_DTYPE_F32, GEIST_LAYOUT_DENSE, 1, q_out);
    struct geist_tensor t_k = bench_tensor_2d(k_b, GEIST_DTYPE_F32, GEIST_LAYOUT_DENSE, 1, kv_out);
    struct geist_tensor t_v = bench_tensor_2d(vv_b, GEIST_DTYPE_F32, GEIST_LAYOUT_DENSE, 1, kv_out);
    struct geist_tensor t_attn = bench_tensor_2d(attn_b, GEIST_DTYPE_F32, GEIST_LAYOUT_DENSE, 1, q_out);
    struct geist_tensor t_o = bench_tensor_2d(o_b, GEIST_DTYPE_F32, GEIST_LAYOUT_DENSE, 1, d_model);
    struct geist_tensor t_attn_post = bench_tensor_2d(attn_post_b, GEIST_DTYPE_F32, GEIST_LAYOUT_DENSE, 1, d_model);
    struct geist_tensor t_attn_out = bench_tensor_2d(attn_out_b, GEIST_DTYPE_F32, GEIST_LAYOUT_DENSE, 1, d_model);
    struct geist_tensor t_ffn_pre = bench_tensor_2d(ffn_pre_b, GEIST_DTYPE_F32, GEIST_LAYOUT_DENSE, 1, d_model);
    struct geist_tensor t_gate_s = bench_tensor_2d(gate_s_b, GEIST_DTYPE_F32, GEIST_LAYOUT_DENSE, 1, inter);
    struct geist_tensor t_up_s = bench_tensor_2d(up_s_b, GEIST_DTYPE_F32, GEIST_LAYOUT_DENSE, 1, inter);
    struct geist_tensor t_ffn_out = bench_tensor_2d(ffn_out_b, GEIST_DTYPE_F32, GEIST_LAYOUT_DENSE, 1, d_model);
    struct geist_tensor t_ffn_post = bench_tensor_2d(ffn_post_b, GEIST_DTYPE_F32, GEIST_LAYOUT_DENSE, 1, d_model);
    struct geist_tensor t_out = bench_tensor_2d(out_b, GEIST_DTYPE_F32, GEIST_LAYOUT_DENSE, 1, d_model);

    const struct geist_backend_attention_block attn_block = {
        .struct_size = sizeof(attn_block), .q_position = kv_len - 1u,
        .kv_len = kv_len, .d_model = d_model, .q_heads = q_heads,
        .kv_heads = kv_heads, .head_dim = head_dim, .sliding_window = 0,
        .eps = 1e-6f, .residual = &t_res, .attn_norm_weight = &t_attn_norm,
        .q_proj_weight = &t_q_w, .k_proj_weight = &t_k_w,
        .v_proj_weight = &t_v_w, .q_norm_weight = &t_q_norm,
        .k_norm_weight = &t_k_norm, .v_norm_weight = &t_v_norm,
        .cos = &t_cos, .sin = &t_sin, .k_cache = &t_k_cache,
        .v_cache = &t_v_cache, .o_proj_weight = &t_o_w,
        .post_attn_norm_weight = &t_attn_post_norm,
        .normed_scratch = &t_attn_normed, .q_scratch = &t_q,
        .k_scratch = &t_k, .v_scratch = &t_v,
        .attn_scratch = &t_attn, .o_scratch = &t_o,
        .post_attn_scratch = &t_attn_post, .out = &t_attn_out,
    };
    const struct geist_backend_ffn_geglu_block ffn_block = {
        .struct_size = sizeof(ffn_block), .seq = 1, .d_model = d_model,
        .inter = inter, .eps = 1e-6f, .residual = &t_attn_out,
        .ffn_norm_weight = &t_ffn_norm, .gate_weight = &t_gate_w,
        .up_weight = &t_up_w, .down_weight = &t_down_w,
        .post_ffw_norm_weight = &t_ffn_post_norm,
        .pre_ff_scratch = &t_ffn_pre, .gate_scratch = &t_gate_s,
        .up_scratch = &t_up_s, .ffn_out_scratch = &t_ffn_out,
        .post_ff_scratch = &t_ffn_post, .out = &t_out,
    };

    s = v->attention_block(be, &attn_block);
    if (s == GEIST_OK) {
        s = v->ffn_geglu_block(be, &ffn_block);
    }
    if (s == GEIST_OK) {
        const double twarm = now_ms();
        s = v->attention_block(be, &attn_block);
        if (s == GEIST_OK) {
            s = v->ffn_geglu_block(be, &ffn_block);
        }
        const double single_ms = now_ms() - twarm;
        int n_iter = (int) (200.0 / (single_ms + 0.001)) + 3;
        if (n_iter > 1000) { n_iter = 1000; }
        if (n_iter < 3) { n_iter = 3; }
        const double t0 = now_ms();
        for (int it = 0; it < n_iter; it++) {
            s = v->attention_block(be, &attn_block);
            if (s == GEIST_OK) { s = v->ffn_geglu_block(be, &ffn_block); }
            if (s != GEIST_OK) { break; }
        }
        const double dt_ms = (now_ms() - t0) / n_iter;
        if (s == GEIST_OK) {
            s = v->buffer_download(d_model * sizeof(float),
                                   (uint8_t *) out_host, out_b);
        }
        const double logical =
            (double) q_t->nbytes + (double) k_t->nbytes +
            (double) v_t->nbytes + (double) o_t->nbytes +
            (double) gate_t->nbytes + (double) up_t->nbytes +
            (double) down_t->nbytes +
            (double) kv_len * (double) kv_out * 2.0 * sizeof(float);
        if (s == GEIST_OK) {
            printf("  %-32s [Metal LAYER Gemma kv=%zu] logical %7.1f MB  %7.2f ms  %5.2f GB/s  (%d it)\n",
                   name, kv_len, logical / (1024.0 * 1024.0), dt_ms,
                   logical / (dt_ms * 1e6), n_iter);
            fflush(stdout);
        }
        if (s == GEIST_OK && getenv("GEIST_Q4K_BENCH_METAL_SEQUENCE") != NULL &&
            v->command_sequence_begin != NULL && v->command_sequence_end != NULL) {
            int seq_iter = n_iter;
            if (seq_iter > 128) { seq_iter = 128; }
            int token = 0;
            const double ts0 = now_ms();
            s = v->command_sequence_begin(be, GEIST_COMMAND_SEQUENCE_DECODE_LAYER_LOOP, &token);
            if (s == GEIST_OK) {
                for (int it = 0; it < seq_iter; it++) {
                    s = v->attention_block(be, &attn_block);
                    if (s == GEIST_OK) { s = v->ffn_geglu_block(be, &ffn_block); }
                    if (s != GEIST_OK) { break; }
                }
                enum geist_status es = v->command_sequence_end(be, token, s == GEIST_OK);
                if (s == GEIST_OK) { s = es; }
            }
            const double seq_dt_ms = (now_ms() - ts0) / (double) seq_iter;
            if (s == GEIST_OK) {
                printf("  %-32s [Metal LAYER Gemma kv=%zu seq] logical %7.1f MB  %7.2f ms  %5.2f GB/s  (%d blocks)\n",
                       name, kv_len, logical / (1024.0 * 1024.0),
                       seq_dt_ms, logical / (seq_dt_ms * 1e6), seq_iter);
                fflush(stdout);
            }
        }
    }
    if (s != GEIST_OK) {
        fprintf(stderr, "  %-32s [Metal LAYER Gemma kv=%zu] FAIL (%s)\n",
                name, kv_len, geist_backend_errmsg(be));
    }

layer_done:
#undef CREATE_LAYER_BUF
#undef UPLOAD_LAYER_BUF
    struct geist_buffer *bufs[] = {
        res_b, attn_norm_b, attn_post_norm_b, ffn_norm_b, ffn_post_norm_b,
        q_norm_b, k_norm_b, v_norm_b, cos_b, sin_b, q_w_b, k_w_b, v_w_b,
        o_w_b, gate_w_b, up_w_b, down_w_b, k_cache_b, v_cache_b,
        attn_normed_b, q_b, k_b, vv_b, attn_b, o_b, attn_post_b, attn_out_b,
        ffn_pre_b, gate_s_b, up_s_b, ffn_out_b, ffn_post_b, out_b,
    };
    for (size_t i = 0; i < sizeof(bufs) / sizeof(bufs[0]); i++) {
        if (bufs[i] != NULL) { v->buffer_destroy(be, bufs[i]); }
    }

done:
    free(residual_host);
    free(attn_norm_host);
    free(attn_post_norm_host);
    free(ffn_norm_host);
    free(ffn_post_norm_host);
    free(q_norm_host);
    free(k_norm_host);
    free(v_norm_host);
    free(cos_host);
    free(sin_host);
    free(k_cache_host);
    free(v_cache_host);
    free(out_host);
    geist_backend_destroy(be);
}

static void bench_metal_ffn_gemma_q4q6(const struct gguf_ctx *ctx,
                                       const struct gguf_tensor_t *gate_t,
                                       const char *name,
                                       size_t rows) {
    if (getenv("GEIST_Q4K_BENCH_METAL_FFN") == NULL) {
        return;
    }
    if (ctx == NULL || gate_t == NULL || gate_t->dtype != GGUF_TYPE_Q4_K ||
        strstr(name, "ffn_gate.weight") == NULL || rows == 0) {
        return;
    }

    char up_name[256];
    char down_name[256];
    if (!bench_ffn_sibling_name(name, "ffn_up.weight",
                                sizeof(up_name), up_name) ||
        !bench_ffn_sibling_name(name, "ffn_down.weight",
                                sizeof(down_name), down_name)) {
        return;
    }
    const struct gguf_tensor_t *up_t = gguf_get_tensor(ctx, up_name);
    const struct gguf_tensor_t *down_t = gguf_get_tensor(ctx, down_name);
    if (up_t == NULL || down_t == NULL || up_t->dtype != GGUF_TYPE_Q4_K ||
        (down_t->dtype != GGUF_TYPE_Q6_K && down_t->dtype != GGUF_TYPE_Q4_K)) {
        fprintf(stderr,
                "  %-32s [Metal FFN Gemma m=%zu] SKIP (sibling weights missing)\n",
                name, rows);
        return;
    }

    const size_t d_model = gate_t->dims[0];
    const size_t inter = gate_t->dims[1];
    if (up_t->dims[0] != d_model || up_t->dims[1] != inter ||
        down_t->dims[0] != inter || down_t->dims[1] != d_model) {
        fprintf(stderr,
                "  %-32s [Metal FFN Gemma m=%zu] SKIP (shape mismatch)\n",
                name, rows);
        return;
    }

    struct geist_backend *be = NULL;
    enum geist_status s = geist_backend_create("metal", NULL, NULL, &be);
    if (s != GEIST_OK) {
        fprintf(stderr, "  %-32s [Metal FFN Gemma m=%zu] SKIP (%s)\n",
                name, rows, geist_last_create_error());
        return;
    }
    const struct geist_backend_vtbl *v = be->desc->vtbl;
    if (v == NULL || v->ffn_geglu_block == NULL) {
        fprintf(stderr,
                "  %-32s [Metal FFN Gemma m=%zu] SKIP (ffn block missing)\n",
                name, rows);
        geist_backend_destroy(be);
        return;
    }

    float *residual_host =
        (float *) aligned_alloc(64, rows * d_model * sizeof(float));
    float *norm_host = (float *) aligned_alloc(64, d_model * sizeof(float));
    float *post_norm_host =
        (float *) aligned_alloc(64, d_model * sizeof(float));
    float *out_host =
        (float *) aligned_alloc(64, rows * d_model * sizeof(float));
    if (residual_host == NULL || norm_host == NULL ||
        post_norm_host == NULL || out_host == NULL) {
        fprintf(stderr,
                "  %-32s [Metal FFN Gemma m=%zu] SKIP (host alloc fail)\n",
                name, rows);
        goto done;
    }
    for (size_t i = 0; i < rows * d_model; i++) {
        residual_host[i] =
            ((float) ((i * 11u) % 4096u) * 0.00003125f) - 0.0625f;
    }
    for (size_t i = 0; i < d_model; i++) {
        norm_host[i] = 0.0075f + (float) (i % 7u) * 0.00025f;
        post_norm_host[i] = 0.25f + (float) (i % 5u) * 0.015625f;
    }

    struct geist_buffer *residual_b = NULL;
    struct geist_buffer *norm_b = NULL;
    struct geist_buffer *post_norm_b = NULL;
    struct geist_buffer *gate_b = NULL;
    struct geist_buffer *up_b = NULL;
    struct geist_buffer *down_b = NULL;
    struct geist_buffer *pre_b = NULL;
    struct geist_buffer *gate_s_b = NULL;
    struct geist_buffer *up_s_b = NULL;
    struct geist_buffer *ffn_out_b = NULL;
    struct geist_buffer *post_b = NULL;
    struct geist_buffer *out_b = NULL;

#define CREATE_REAL_FFN(ptr_, bytes_, kind_)                                      \
    do {                                                                          \
        if (s == GEIST_OK) {                                                      \
            s = v->buffer_create(be, (bytes_), (kind_), GEIST_MEMORY_DEVICE, &(ptr_)); \
        }                                                                         \
    } while (0)
#define UPLOAD_REAL_FFN(ptr_, bytes_, data_)                                      \
    do {                                                                          \
        if (s == GEIST_OK) {                                                      \
            s = v->buffer_upload((ptr_), (bytes_), (const uint8_t *) (data_));     \
        }                                                                         \
    } while (0)

    CREATE_REAL_FFN(residual_b, rows * d_model * sizeof(float),
                    GEIST_BUFFER_ACTIVATION);
    CREATE_REAL_FFN(norm_b, d_model * sizeof(float), GEIST_BUFFER_WEIGHT);
    CREATE_REAL_FFN(post_norm_b, d_model * sizeof(float), GEIST_BUFFER_WEIGHT);
    CREATE_REAL_FFN(gate_b, gate_t->nbytes, GEIST_BUFFER_WEIGHT);
    CREATE_REAL_FFN(up_b, up_t->nbytes, GEIST_BUFFER_WEIGHT);
    CREATE_REAL_FFN(down_b, down_t->nbytes, GEIST_BUFFER_WEIGHT);
    CREATE_REAL_FFN(pre_b, rows * d_model * sizeof(float), GEIST_BUFFER_SCRATCH);
    CREATE_REAL_FFN(gate_s_b, rows * inter * sizeof(float), GEIST_BUFFER_SCRATCH);
    CREATE_REAL_FFN(up_s_b, rows * inter * sizeof(float), GEIST_BUFFER_SCRATCH);
    CREATE_REAL_FFN(ffn_out_b, rows * d_model * sizeof(float),
                    GEIST_BUFFER_SCRATCH);
    CREATE_REAL_FFN(post_b, rows * d_model * sizeof(float), GEIST_BUFFER_SCRATCH);
    CREATE_REAL_FFN(out_b, rows * d_model * sizeof(float),
                    GEIST_BUFFER_ACTIVATION);

    UPLOAD_REAL_FFN(residual_b, rows * d_model * sizeof(float), residual_host);
    UPLOAD_REAL_FFN(norm_b, d_model * sizeof(float), norm_host);
    UPLOAD_REAL_FFN(post_norm_b, d_model * sizeof(float), post_norm_host);
    UPLOAD_REAL_FFN(gate_b, gate_t->nbytes, gate_t->data);
    UPLOAD_REAL_FFN(up_b, up_t->nbytes, up_t->data);
    UPLOAD_REAL_FFN(down_b, down_t->nbytes, down_t->data);
    if (s != GEIST_OK) {
        fprintf(stderr, "  %-32s [Metal FFN Gemma m=%zu] SKIP (setup: %s)\n",
                name, rows, geist_backend_errmsg(be));
        goto ffn_done;
    }

    struct geist_tensor t_residual =
        bench_tensor_2d(residual_b, GEIST_DTYPE_F32,
                        GEIST_LAYOUT_DENSE, rows, d_model);
    struct geist_tensor t_norm =
        bench_tensor_2d(norm_b, GEIST_DTYPE_F32,
                        GEIST_LAYOUT_DENSE, 1, d_model);
    struct geist_tensor t_post_norm =
        bench_tensor_2d(post_norm_b, GEIST_DTYPE_F32,
                        GEIST_LAYOUT_DENSE, 1, d_model);
    struct geist_tensor t_gate =
        bench_tensor_2d(gate_b, GEIST_DTYPE_Q4_K,
                        GEIST_LAYOUT_BLOCK_QUANTIZED, inter, d_model);
    struct geist_tensor t_up =
        bench_tensor_2d(up_b, GEIST_DTYPE_Q4_K,
                        GEIST_LAYOUT_BLOCK_QUANTIZED, inter, d_model);
    struct geist_tensor t_down =
        bench_tensor_2d(down_b,
                        down_t->dtype == GGUF_TYPE_Q6_K ? GEIST_DTYPE_Q6_K
                                                        : GEIST_DTYPE_Q4_K,
                        GEIST_LAYOUT_BLOCK_QUANTIZED, d_model, inter);
    struct geist_tensor t_pre =
        bench_tensor_2d(pre_b, GEIST_DTYPE_F32,
                        GEIST_LAYOUT_DENSE, rows, d_model);
    struct geist_tensor t_gate_s =
        bench_tensor_2d(gate_s_b, GEIST_DTYPE_F32,
                        GEIST_LAYOUT_DENSE, rows, inter);
    struct geist_tensor t_up_s =
        bench_tensor_2d(up_s_b, GEIST_DTYPE_F32,
                        GEIST_LAYOUT_DENSE, rows, inter);
    struct geist_tensor t_ffn_out =
        bench_tensor_2d(ffn_out_b, GEIST_DTYPE_F32,
                        GEIST_LAYOUT_DENSE, rows, d_model);
    struct geist_tensor t_post =
        bench_tensor_2d(post_b, GEIST_DTYPE_F32,
                        GEIST_LAYOUT_DENSE, rows, d_model);
    struct geist_tensor t_out =
        bench_tensor_2d(out_b, GEIST_DTYPE_F32,
                        GEIST_LAYOUT_DENSE, rows, d_model);
    const struct geist_backend_ffn_geglu_block block = {
        .struct_size = sizeof(block),
        .seq = rows,
        .d_model = d_model,
        .inter = inter,
        .eps = 1e-6f,
        .residual = &t_residual,
        .ffn_norm_weight = &t_norm,
        .gate_weight = &t_gate,
        .up_weight = &t_up,
        .down_weight = &t_down,
        .post_ffw_norm_weight = &t_post_norm,
        .pre_ff_scratch = &t_pre,
        .gate_scratch = &t_gate_s,
        .up_scratch = &t_up_s,
        .ffn_out_scratch = &t_ffn_out,
        .post_ff_scratch = &t_post,
        .out = &t_out,
    };

    if (v->prepare_weight_layout != NULL) {
        s = v->prepare_weight_layout(be, &t_gate);
        if (s == GEIST_OK) {
            s = v->prepare_weight_layout(be, &t_up);
        }
        if (s == GEIST_OK &&
            (t_down.dtype == GEIST_DTYPE_Q4_K ||
             t_down.dtype == GEIST_DTYPE_Q6_K)) {
            s = v->prepare_weight_layout(be, &t_down);
        }
        if (s != GEIST_OK) {
            fprintf(stderr,
                    "  [Metal FFN Gemma m=%zu] SKIP (prepare: %s)\n",
                    rows, geist_backend_errmsg(be));
            goto done;
        }
    }

    s = v->ffn_geglu_block(be, &block);
    if (s == GEIST_OK) {
        const double twarm = now_ms();
        s = v->ffn_geglu_block(be, &block);
        const double single_ms = now_ms() - twarm;
        int n_iter = (int) (200.0 / (single_ms + 0.001)) + 3;
        if (n_iter > 1000) { n_iter = 1000; }
        if (n_iter < 3) { n_iter = 3; }
        const double t0 = now_ms();
        for (int it = 0; it < n_iter; it++) {
            s = v->ffn_geglu_block(be, &block);
            if (s != GEIST_OK) { break; }
        }
        const double dt_ms = (now_ms() - t0) / n_iter;
        if (s == GEIST_OK) {
            s = v->buffer_download(rows * d_model * sizeof(float),
                                   (uint8_t *) out_host, out_b);
        }
        if (s == GEIST_OK) {
            const double logical =
                (double) rows *
                ((double) gate_t->nbytes + (double) up_t->nbytes +
                 (double) down_t->nbytes);
            printf("  %-32s [Metal FFN Gemma m=%zu] logical %7.1f MB  %7.2f ms  %5.2f GB/s  (%d it)\n",
                   name, rows, logical / (1024.0 * 1024.0), dt_ms,
                   logical / (dt_ms * 1e6), n_iter);
            fflush(stdout);
        }
        if (s == GEIST_OK &&
            getenv("GEIST_Q4K_BENCH_METAL_SEQUENCE") != NULL &&
            v->command_sequence_begin != NULL &&
            v->command_sequence_end != NULL) {
            int seq_iter = n_iter;
            if (seq_iter > 128) { seq_iter = 128; }
            int token = 0;
            const double ts0 = now_ms();
            s = v->command_sequence_begin(
                be, GEIST_COMMAND_SEQUENCE_DECODE_LAYER_LOOP, &token);
            if (s == GEIST_OK) {
                for (int it = 0; it < seq_iter; it++) {
                    s = v->ffn_geglu_block(be, &block);
                    if (s != GEIST_OK) { break; }
                }
                enum geist_status es =
                    v->command_sequence_end(be, token, s == GEIST_OK);
                if (s == GEIST_OK) { s = es; }
            }
            const double seq_dt_ms = (now_ms() - ts0) / (double) seq_iter;
            if (s == GEIST_OK) {
                const double logical =
                    (double) rows *
                    ((double) gate_t->nbytes + (double) up_t->nbytes +
                     (double) down_t->nbytes);
                printf("  %-32s [Metal FFN Gemma m=%zu seq] logical %7.1f MB  %7.2f ms  %5.2f GB/s  (%d blocks)\n",
                       name, rows, logical / (1024.0 * 1024.0),
                       seq_dt_ms, logical / (seq_dt_ms * 1e6), seq_iter);
                fflush(stdout);
            }
        }
    }
    if (s != GEIST_OK) {
        fprintf(stderr, "  %-32s [Metal FFN Gemma m=%zu] FAIL (%s)\n",
                name, rows, geist_backend_errmsg(be));
    }

ffn_done:
#undef CREATE_REAL_FFN
#undef UPLOAD_REAL_FFN
    if (residual_b != NULL) { v->buffer_destroy(be, residual_b); }
    if (norm_b != NULL) { v->buffer_destroy(be, norm_b); }
    if (post_norm_b != NULL) { v->buffer_destroy(be, post_norm_b); }
    if (gate_b != NULL) { v->buffer_destroy(be, gate_b); }
    if (up_b != NULL) { v->buffer_destroy(be, up_b); }
    if (down_b != NULL) { v->buffer_destroy(be, down_b); }
    if (pre_b != NULL) { v->buffer_destroy(be, pre_b); }
    if (gate_s_b != NULL) { v->buffer_destroy(be, gate_s_b); }
    if (up_s_b != NULL) { v->buffer_destroy(be, up_s_b); }
    if (ffn_out_b != NULL) { v->buffer_destroy(be, ffn_out_b); }
    if (post_b != NULL) { v->buffer_destroy(be, post_b); }
    if (out_b != NULL) { v->buffer_destroy(be, out_b); }

done:
    free(residual_host);
    free(norm_host);
    free(post_norm_host);
    free(out_host);
    geist_backend_destroy(be);
}

static void bench_metal_q6k(const struct gguf_tensor_t *t,
                            const char *name,
                            size_t weight_bytes,
                            size_t matmul_rows) {
    if (getenv("GEIST_Q4K_BENCH_METAL") == NULL) {
        return;
    }
    if (t == NULL || t->dtype != GGUF_TYPE_Q6_K) {
        return;
    }

    const size_t n_in = t->dims[0];
    const size_t n_out = t->dims[1];
    struct geist_backend *be = NULL;
    enum geist_status s = geist_backend_create("metal", NULL, NULL, &be);
    if (s != GEIST_OK) {
        fprintf(stderr, "  %-32s [Metal Q6_K m=1] SKIP (%s)\n",
                name, geist_last_create_error());
        return;
    }
    const struct geist_backend_vtbl *v = be->desc->vtbl;
    if (v == NULL || v->matvec_q6k == NULL || v->matmul_q6k == NULL) {
        fprintf(stderr, "  %-32s [Metal Q6_K] SKIP (Q6 missing)\n", name);
        geist_backend_destroy(be);
        return;
    }

    float *x_host = (float *) aligned_alloc(64, matmul_rows * n_in * sizeof(float));
    float *y_host = (float *) aligned_alloc(64, matmul_rows * n_out * sizeof(float));
    struct geist_buffer *x = NULL;
    struct geist_buffer *w = NULL;
    struct geist_buffer *y = NULL;
    if (x_host == NULL || y_host == NULL) {
        fprintf(stderr, "  %-32s [Metal Q6_K] SKIP (host alloc fail)\n", name);
        goto done;
    }
    for (size_t r = 0; r < matmul_rows; r++) {
        for (size_t i = 0; i < n_in; i++) {
            x_host[r * n_in + i] =
                ((float) ((r * 17u + i * 13u) % 4096u) * 0.0019f) - 3.9f;
        }
    }
    s = v->buffer_create(be, matmul_rows * n_in * sizeof(float),
                         GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_DEVICE, &x);
    if (s == GEIST_OK) {
        s = v->buffer_create(be, weight_bytes, GEIST_BUFFER_WEIGHT,
                             GEIST_MEMORY_DEVICE, &w);
    }
    if (s == GEIST_OK) {
        s = v->buffer_create(be, matmul_rows * n_out * sizeof(float),
                             GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_DEVICE, &y);
    }
    if (s == GEIST_OK) {
        s = v->buffer_upload(x, matmul_rows * n_in * sizeof(float),
                             (const uint8_t *) x_host);
    }
    if (s == GEIST_OK) {
        s = v->buffer_upload(w, weight_bytes, (const uint8_t *) t->data);
    }
    if (s != GEIST_OK) {
        fprintf(stderr, "  %-32s [Metal Q6_K] SKIP (setup: %s)\n",
                name, geist_backend_errmsg(be));
        goto done;
    }

    struct geist_tensor tw =
        bench_tensor_2d(w, GEIST_DTYPE_Q6_K,
                        GEIST_LAYOUT_BLOCK_QUANTIZED, n_out, n_in);
    if (v->prepare_weight_layout != NULL) {
        s = v->prepare_weight_layout(be, &tw);
        if (s != GEIST_OK) {
            fprintf(stderr,
                    "  %-32s [Metal Q6_K] SKIP (prepare: %s)\n",
                    name, geist_backend_errmsg(be));
            goto done;
        }
    }
    struct geist_tensor tx1 =
        bench_tensor_2d(x, GEIST_DTYPE_F32, GEIST_LAYOUT_DENSE, 1, n_in);
    struct geist_tensor ty1 =
        bench_tensor_2d(y, GEIST_DTYPE_F32, GEIST_LAYOUT_DENSE, 1, n_out);
    const char *q6_raw_env = getenv("GEIST_METAL_Q6K_LINEAR_RAW");
    const char *q6_mode =
        q6_raw_env != NULL && strcmp(q6_raw_env, "1") == 0 ? "raw"
                                                           : "packed";
    s = v->matvec_q6k(be, &tx1, &tw, &ty1);
    if (s == GEIST_OK) {
        const double twarm = now_ms();
        s = v->matvec_q6k(be, &tx1, &tw, &ty1);
        const double single_ms = now_ms() - twarm;
        int n_iter = (int) (200.0 / (single_ms + 0.001)) + 3;
        if (n_iter > 1000) { n_iter = 1000; }
        if (n_iter < 3) { n_iter = 3; }
        const double t0 = now_ms();
        for (int it = 0; it < n_iter; it++) {
            s = v->matvec_q6k(be, &tx1, &tw, &ty1);
            if (s != GEIST_OK) { break; }
        }
        const double dt_ms = (now_ms() - t0) / n_iter;
        if (s == GEIST_OK) {
            printf("  %-32s [Metal Q6_K %s m=1] logical %7.1f MB  %7.2f ms  %5.2f GB/s  (%d it)\n",
                   name,
                   q6_mode,
                   (double) weight_bytes / (1024.0 * 1024.0),
                   dt_ms,
                   (double) weight_bytes / (dt_ms * 1e6),
                   n_iter);
            fflush(stdout);
        }
        if (s == GEIST_OK &&
            getenv("GEIST_Q4K_BENCH_METAL_SEQUENCE") != NULL &&
            v->command_sequence_begin != NULL &&
            v->command_sequence_end != NULL) {
            int token = 0;
            const double ts0 = now_ms();
            s = v->command_sequence_begin(
                be, GEIST_COMMAND_SEQUENCE_VERIFY_GREEDY, &token);
            if (s == GEIST_OK) {
                for (int it = 0; it < n_iter; it++) {
                    s = v->matvec_q6k(be, &tx1, &tw, &ty1);
                    if (s != GEIST_OK) { break; }
                }
                enum geist_status es =
                    v->command_sequence_end(be, token, s == GEIST_OK);
                if (s == GEIST_OK) {
                    s = es;
                }
            }
            const double seq_dt_ms = (now_ms() - ts0) / n_iter;
            if (s == GEIST_OK) {
                printf("  %-32s [Metal Q6_K %s m=1 seq] logical %7.1f MB  %7.2f ms  %5.2f GB/s  (%d dispatches)\n",
                       name,
                       q6_mode,
                       (double) weight_bytes / (1024.0 * 1024.0),
                       seq_dt_ms,
                       (double) weight_bytes / (seq_dt_ms * 1e6),
                       n_iter);
                fflush(stdout);
            }
        }
    }
    if (s != GEIST_OK) {
        fprintf(stderr, "  %-32s [Metal Q6_K m=1] FAIL (%s)\n",
                name, geist_backend_errmsg(be));
        goto done;
    }

    if (matmul_rows > 1) {
        struct geist_tensor txm =
            bench_tensor_2d(x, GEIST_DTYPE_F32, GEIST_LAYOUT_DENSE,
                            matmul_rows, n_in);
        struct geist_tensor tym =
            bench_tensor_2d(y, GEIST_DTYPE_F32, GEIST_LAYOUT_DENSE,
                            matmul_rows, n_out);
        s = v->matmul_q6k(be, &txm, &tw, &tym);
        if (s == GEIST_OK) {
            const double twarm = now_ms();
            s = v->matmul_q6k(be, &txm, &tw, &tym);
            const double single_ms = now_ms() - twarm;
            int n_iter = (int) (200.0 / (single_ms + 0.001)) + 3;
            if (n_iter > 1000) { n_iter = 1000; }
            if (n_iter < 3) { n_iter = 3; }
            const double t0 = now_ms();
            for (int it = 0; it < n_iter; it++) {
                s = v->matmul_q6k(be, &txm, &tw, &tym);
                if (s != GEIST_OK) { break; }
            }
            const double dt_ms = (now_ms() - t0) / n_iter;
            if (s == GEIST_OK) {
                const double logical =
                    (double) weight_bytes * (double) matmul_rows;
                printf("  %-32s [Metal Q6_K %s m=%zu] logical %7.1f MB  %7.2f ms  %5.2f GB/s  (%d it)\n",
                       name,
                       q6_mode,
                       matmul_rows,
                       logical / (1024.0 * 1024.0),
                       dt_ms,
                       logical / (dt_ms * 1e6),
                       n_iter);
                fflush(stdout);
            }
        }
        if (s != GEIST_OK) {
            fprintf(stderr, "  %-32s [Metal Q6_K m=%zu] FAIL (%s)\n",
                    name, matmul_rows, geist_backend_errmsg(be));
        }
    }

done:
    if (x != NULL) { v->buffer_destroy(be, x); }
    if (w != NULL) { v->buffer_destroy(be, w); }
    if (y != NULL) { v->buffer_destroy(be, y); }
    free(x_host);
    free(y_host);
    geist_backend_destroy(be);
}

static void bench_metal_q4k(const struct gguf_tensor_t *t,
                            const char *name,
                            size_t weight_bytes,
                            size_t matmul_rows) {
    if (getenv("GEIST_Q4K_BENCH_METAL") == NULL) {
        return;
    }
    if (t == NULL || t->dtype != GGUF_TYPE_Q4_K) {
        return;
    }

    const size_t n_in = t->dims[0];
    const size_t n_out = t->dims[1];
    struct geist_backend *be = NULL;
    enum geist_status s = geist_backend_create("metal", NULL, NULL, &be);
    if (s != GEIST_OK) {
        fprintf(stderr, "  %-32s [Metal Q4_K m=1] SKIP (%s)\n",
                name, geist_last_create_error());
        return;
    }
    const struct geist_backend_vtbl *v = be->desc->vtbl;
    if (v == NULL || v->matvec_q4k == NULL) {
        fprintf(stderr, "  %-32s [Metal Q4_K m=1] SKIP (matvec missing)\n",
                name);
        geist_backend_destroy(be);
        return;
    }

    float *x_host = (float *) aligned_alloc(64, n_in * sizeof(float));
    float *y_host = (float *) aligned_alloc(64, n_out * sizeof(float));
    struct geist_buffer *x = NULL;
    struct geist_buffer *w = NULL;
    struct geist_buffer *y = NULL;
    if (x_host == NULL || y_host == NULL) {
        fprintf(stderr, "  %-32s [Metal Q4_K m=1] SKIP (host alloc fail)\n",
                name);
        goto done;
    }
    for (size_t i = 0; i < n_in; i++) {
        x_host[i] = ((float) ((i * 13u) % 4096u) * 0.0019f) - 3.9f;
    }

    s = v->buffer_create(be, n_in * sizeof(float),
                         GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_DEVICE, &x);
    if (s == GEIST_OK) {
        s = v->buffer_create(be, weight_bytes, GEIST_BUFFER_WEIGHT,
                             GEIST_MEMORY_DEVICE, &w);
    }
    if (s == GEIST_OK) {
        s = v->buffer_create(be, n_out * sizeof(float),
                             GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_DEVICE, &y);
    }
    if (s != GEIST_OK) {
        fprintf(stderr, "  %-32s [Metal Q4_K m=1] SKIP (buffer_create: %s)\n",
                name, geist_backend_errmsg(be));
        goto done;
    }
    s = v->buffer_upload(x, n_in * sizeof(float),
                         (const uint8_t *) x_host);
    if (s == GEIST_OK) {
        s = v->buffer_upload(w, weight_bytes, (const uint8_t *) t->data);
    }
    if (s != GEIST_OK) {
        fprintf(stderr, "  %-32s [Metal Q4_K m=1] SKIP (upload: %s)\n",
                name, geist_backend_errmsg(be));
        goto done;
    }

    struct geist_tensor tx =
        bench_tensor_2d(x, GEIST_DTYPE_F32, GEIST_LAYOUT_DENSE, 1, n_in);
    struct geist_tensor tw =
        bench_tensor_2d(w, GEIST_DTYPE_Q4_K,
                        GEIST_LAYOUT_BLOCK_QUANTIZED, n_out, n_in);
    struct geist_tensor ty =
        bench_tensor_2d(y, GEIST_DTYPE_F32, GEIST_LAYOUT_DENSE, 1, n_out);

    if (v->prepare_weight_layout != NULL) {
        s = v->prepare_weight_layout(be, &tw);
        if (s != GEIST_OK) {
            fprintf(stderr,
                    "  %-32s [Metal Q4_K m=1] SKIP (prepare: %s)\n",
                    name, geist_backend_errmsg(be));
            goto done;
        }
    }

    s = v->matvec_q4k(be, &tx, &tw, &ty);
    if (s != GEIST_OK) {
        fprintf(stderr, "  %-32s [Metal Q4_K m=1] SKIP (matvec: %s)\n",
                name, geist_backend_errmsg(be));
        goto done;
    }
    const double twarm = now_ms();
    s = v->matvec_q4k(be, &tx, &tw, &ty);
    const double single_ms = now_ms() - twarm;
    if (s != GEIST_OK) {
        fprintf(stderr, "  %-32s [Metal Q4_K m=1] SKIP (matvec warm: %s)\n",
                name, geist_backend_errmsg(be));
        goto done;
    }
    int n_iter = (int) (200.0 / (single_ms + 0.001)) + 3;
    if (n_iter > 1000) {
        n_iter = 1000;
    }
    if (n_iter < 3) {
        n_iter = 3;
    }
    const double t0 = now_ms();
    for (int it = 0; it < n_iter; it++) {
        s = v->matvec_q4k(be, &tx, &tw, &ty);
        if (s != GEIST_OK) {
            break;
        }
    }
    const double dt_ms = (now_ms() - t0) / n_iter;
    if (s != GEIST_OK) {
        fprintf(stderr, "  %-32s [Metal Q4_K m=1] FAIL (matvec: %s)\n",
                name, geist_backend_errmsg(be));
        goto done;
    }
    s = v->buffer_download(n_out * sizeof(float), (uint8_t *) y_host, y);
    if (s != GEIST_OK) {
        fprintf(stderr, "  %-32s [Metal Q4_K m=1] FAIL (download: %s)\n",
                name, geist_backend_errmsg(be));
        goto done;
    }
    const double logical_gbps = (double) weight_bytes / (dt_ms * 1e6);
    printf("  %-32s [Metal Q4_K m=1]  logical %7.1f MB  %7.2f ms  %5.2f GB/s  (%d it)\n",
           name,
           (double) weight_bytes / (1024.0 * 1024.0),
           dt_ms,
           logical_gbps,
           n_iter);
    fflush(stdout);
    if (y_host[0] == 0.0f && y_host[n_out - 1] == 0.0f) {
        fprintf(stderr, "(zero Metal output)\n");
    }

    if (getenv("GEIST_Q4K_BENCH_METAL_SEQUENCE") != NULL &&
        v->command_sequence_begin != NULL &&
        v->command_sequence_end != NULL) {
        int token = 0;
        const double ts0 = now_ms();
        s = v->command_sequence_begin(
            be, GEIST_COMMAND_SEQUENCE_VERIFY_GREEDY, &token);
        if (s == GEIST_OK) {
            for (int it = 0; it < n_iter; it++) {
                s = v->matvec_q4k(be, &tx, &tw, &ty);
                if (s != GEIST_OK) {
                    break;
                }
            }
            enum geist_status es =
                v->command_sequence_end(be, token, s == GEIST_OK);
            if (s == GEIST_OK) {
                s = es;
            }
        }
        const double seq_dt_ms = (now_ms() - ts0) / n_iter;
        if (s == GEIST_OK) {
            const double seq_gbps = (double) weight_bytes / (seq_dt_ms * 1e6);
            printf("  %-32s [Metal Q4_K m=1 seq] logical %7.1f MB  %7.2f ms  %5.2f GB/s  (%d dispatches)\n",
                   name,
                   (double) weight_bytes / (1024.0 * 1024.0),
                   seq_dt_ms,
                   seq_gbps,
                   n_iter);
            fflush(stdout);
        } else {
            fprintf(stderr,
                    "  %-32s [Metal Q4_K m=1 seq] FAIL (%s)\n",
                    name, geist_backend_errmsg(be));
        }
    }

    if (matmul_rows > 1 && v->matmul_q4k != NULL) {
        float *xm_host = (float *) aligned_alloc(
            64, matmul_rows * n_in * sizeof(float));
        float *ym_host = (float *) aligned_alloc(
            64, matmul_rows * n_out * sizeof(float));
        struct geist_buffer *xm = NULL;
        struct geist_buffer *ym = NULL;
        if (xm_host == NULL || ym_host == NULL) {
            fprintf(stderr, "  %-32s [Metal Q4_K m=%zu] SKIP (host alloc fail)\n",
                    name, matmul_rows);
            free(xm_host);
            free(ym_host);
            goto done;
        }
        for (size_t r = 0; r < matmul_rows; r++) {
            for (size_t i = 0; i < n_in; i++) {
                xm_host[r * n_in + i] =
                    ((float) ((r * 17u + i * 13u) % 4096u) * 0.0019f) - 3.9f;
            }
        }

        s = v->buffer_create(be, matmul_rows * n_in * sizeof(float),
                             GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_DEVICE,
                             &xm);
        if (s == GEIST_OK) {
            s = v->buffer_create(be, matmul_rows * n_out * sizeof(float),
                                 GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_DEVICE,
                                 &ym);
        }
        if (s == GEIST_OK) {
            s = v->buffer_upload(xm, matmul_rows * n_in * sizeof(float),
                                 (const uint8_t *) xm_host);
        }
        if (s != GEIST_OK) {
            fprintf(stderr, "  %-32s [Metal Q4_K m=%zu] SKIP (setup: %s)\n",
                    name, matmul_rows, geist_backend_errmsg(be));
            if (xm != NULL) { v->buffer_destroy(be, xm); }
            if (ym != NULL) { v->buffer_destroy(be, ym); }
            free(xm_host);
            free(ym_host);
            goto done;
        }

        struct geist_tensor txm =
            bench_tensor_2d(xm, GEIST_DTYPE_F32, GEIST_LAYOUT_DENSE,
                            matmul_rows, n_in);
        struct geist_tensor tym =
            bench_tensor_2d(ym, GEIST_DTYPE_F32, GEIST_LAYOUT_DENSE,
                            matmul_rows, n_out);
        s = v->matmul_q4k(be, &txm, &tw, &tym);
        if (s == GEIST_OK) {
            const double tmwarm = now_ms();
            s = v->matmul_q4k(be, &txm, &tw, &tym);
            const double matmul_single_ms = now_ms() - tmwarm;
            int matmul_iter =
                (int) (200.0 / (matmul_single_ms + 0.001)) + 3;
            if (matmul_iter > 1000) {
                matmul_iter = 1000;
            }
            if (matmul_iter < 3) {
                matmul_iter = 3;
            }
            const double tm0 = now_ms();
            for (int it = 0; it < matmul_iter; it++) {
                s = v->matmul_q4k(be, &txm, &tw, &tym);
                if (s != GEIST_OK) {
                    break;
                }
            }
            const double matmul_dt_ms = (now_ms() - tm0) / matmul_iter;
            if (s == GEIST_OK) {
                s = v->buffer_download(matmul_rows * n_out * sizeof(float),
                                       (uint8_t *) ym_host, ym);
            }
            if (s == GEIST_OK) {
                const double matmul_gbps =
                    ((double) weight_bytes * (double) matmul_rows) /
                    (matmul_dt_ms * 1e6);
                printf("  %-32s [Metal Q4_K m=%zu] logical %7.1f MB  %7.2f ms  %5.2f GB/s  (%d it)\n",
                       name,
                       matmul_rows,
                       ((double) weight_bytes * (double) matmul_rows) /
                           (1024.0 * 1024.0),
                       matmul_dt_ms,
                       matmul_gbps,
                       matmul_iter);
                fflush(stdout);
            }
        }
        if (s != GEIST_OK) {
            fprintf(stderr, "  %-32s [Metal Q4_K m=%zu] FAIL (%s)\n",
                    name, matmul_rows, geist_backend_errmsg(be));
        }
        if (xm != NULL) { v->buffer_destroy(be, xm); }
        if (ym != NULL) { v->buffer_destroy(be, ym); }
        free(xm_host);
        free(ym_host);
    }

    if (matmul_rows > 1 && getenv("GEIST_Q4K_BENCH_METAL_FFN") != NULL &&
        v->ffn_geglu_block != NULL && n_out % Q4_K_BLOCK_ELEMS == 0) {
        const size_t rows = matmul_rows;
        const size_t d_model = n_in;
        const size_t inter = n_out;
        const size_t down_bytes =
            d_model * (inter / Q4_K_BLOCK_ELEMS) * Q4_K_BLOCK_BYTES;
        float *residual_host =
            (float *) aligned_alloc(64, rows * d_model * sizeof(float));
        float *norm_host =
            (float *) aligned_alloc(64, d_model * sizeof(float));
        float *post_norm_host =
            (float *) aligned_alloc(64, d_model * sizeof(float));
        float *out_host =
            (float *) aligned_alloc(64, rows * d_model * sizeof(float));
        uint8_t *down_host = (uint8_t *) aligned_alloc(64, down_bytes);
        struct geist_buffer *residual_b = NULL;
        struct geist_buffer *norm_b = NULL;
        struct geist_buffer *post_norm_b = NULL;
        struct geist_buffer *down_b = NULL;
        struct geist_buffer *pre_b = NULL;
        struct geist_buffer *gate_b = NULL;
        struct geist_buffer *up_b = NULL;
        struct geist_buffer *ffn_out_b = NULL;
        struct geist_buffer *post_b = NULL;
        struct geist_buffer *out_b = NULL;
        if (residual_host == NULL || norm_host == NULL ||
            post_norm_host == NULL || out_host == NULL ||
            down_host == NULL) {
            fprintf(stderr,
                    "  %-32s [Metal FFN Q4_K m=%zu] SKIP (host alloc fail)\n",
                    name, rows);
            goto ffn_done;
        }
        for (size_t i = 0; i < rows * d_model; i++) {
            residual_host[i] =
                ((float) ((i * 11u) % 4096u) * 0.00003125f) - 0.0625f;
        }
        for (size_t i = 0; i < d_model; i++) {
            norm_host[i] = 0.0075f + (float) (i % 7u) * 0.00025f;
            post_norm_host[i] = 0.25f + (float) (i % 5u) * 0.015625f;
        }
        bench_pack_q4k_matrix(inter, d_model, down_host);

        s = v->buffer_create(be, rows * d_model * sizeof(float),
                             GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_DEVICE,
                             &residual_b);
        if (s == GEIST_OK) {
            s = v->buffer_create(be, d_model * sizeof(float),
                                 GEIST_BUFFER_WEIGHT, GEIST_MEMORY_DEVICE,
                                 &norm_b);
        }
        if (s == GEIST_OK) {
            s = v->buffer_create(be, d_model * sizeof(float),
                                 GEIST_BUFFER_WEIGHT, GEIST_MEMORY_DEVICE,
                                 &post_norm_b);
        }
        if (s == GEIST_OK) {
            s = v->buffer_create(be, down_bytes, GEIST_BUFFER_WEIGHT,
                                 GEIST_MEMORY_DEVICE, &down_b);
        }
        if (s == GEIST_OK) {
            s = v->buffer_create(be, rows * d_model * sizeof(float),
                                 GEIST_BUFFER_SCRATCH, GEIST_MEMORY_DEVICE,
                                 &pre_b);
        }
        if (s == GEIST_OK) {
            s = v->buffer_create(be, rows * inter * sizeof(float),
                                 GEIST_BUFFER_SCRATCH, GEIST_MEMORY_DEVICE,
                                 &gate_b);
        }
        if (s == GEIST_OK) {
            s = v->buffer_create(be, rows * inter * sizeof(float),
                                 GEIST_BUFFER_SCRATCH, GEIST_MEMORY_DEVICE,
                                 &up_b);
        }
        if (s == GEIST_OK) {
            s = v->buffer_create(be, rows * d_model * sizeof(float),
                                 GEIST_BUFFER_SCRATCH, GEIST_MEMORY_DEVICE,
                                 &ffn_out_b);
        }
        if (s == GEIST_OK) {
            s = v->buffer_create(be, rows * d_model * sizeof(float),
                                 GEIST_BUFFER_SCRATCH, GEIST_MEMORY_DEVICE,
                                 &post_b);
        }
        if (s == GEIST_OK) {
            s = v->buffer_create(be, rows * d_model * sizeof(float),
                                 GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_DEVICE,
                                 &out_b);
        }
        if (s == GEIST_OK) {
            s = v->buffer_upload(residual_b, rows * d_model * sizeof(float),
                                 (const uint8_t *) residual_host);
        }
        if (s == GEIST_OK) {
            s = v->buffer_upload(norm_b, d_model * sizeof(float),
                                 (const uint8_t *) norm_host);
        }
        if (s == GEIST_OK) {
            s = v->buffer_upload(post_norm_b, d_model * sizeof(float),
                                 (const uint8_t *) post_norm_host);
        }
        if (s == GEIST_OK) {
            s = v->buffer_upload(down_b, down_bytes, down_host);
        }
        if (s != GEIST_OK) {
            fprintf(stderr,
                    "  %-32s [Metal FFN Q4_K m=%zu] SKIP (setup: %s)\n",
                    name, rows, geist_backend_errmsg(be));
            goto ffn_done;
        }

        struct geist_tensor t_residual =
            bench_tensor_2d(residual_b, GEIST_DTYPE_F32,
                            GEIST_LAYOUT_DENSE, rows, d_model);
        struct geist_tensor t_norm =
            bench_tensor_2d(norm_b, GEIST_DTYPE_F32,
                            GEIST_LAYOUT_DENSE, 1, d_model);
        struct geist_tensor t_post_norm =
            bench_tensor_2d(post_norm_b, GEIST_DTYPE_F32,
                            GEIST_LAYOUT_DENSE, 1, d_model);
        struct geist_tensor t_down =
            bench_tensor_2d(down_b, GEIST_DTYPE_Q4_K,
                            GEIST_LAYOUT_BLOCK_QUANTIZED, d_model, inter);
        struct geist_tensor t_pre =
            bench_tensor_2d(pre_b, GEIST_DTYPE_F32,
                            GEIST_LAYOUT_DENSE, rows, d_model);
        struct geist_tensor t_gate =
            bench_tensor_2d(gate_b, GEIST_DTYPE_F32,
                            GEIST_LAYOUT_DENSE, rows, inter);
        struct geist_tensor t_up =
            bench_tensor_2d(up_b, GEIST_DTYPE_F32,
                            GEIST_LAYOUT_DENSE, rows, inter);
        struct geist_tensor t_ffn_out =
            bench_tensor_2d(ffn_out_b, GEIST_DTYPE_F32,
                            GEIST_LAYOUT_DENSE, rows, d_model);
        struct geist_tensor t_post =
            bench_tensor_2d(post_b, GEIST_DTYPE_F32,
                            GEIST_LAYOUT_DENSE, rows, d_model);
        struct geist_tensor t_out =
            bench_tensor_2d(out_b, GEIST_DTYPE_F32,
                            GEIST_LAYOUT_DENSE, rows, d_model);
        const struct geist_backend_ffn_geglu_block block = {
            .struct_size = sizeof(block),
            .seq = rows,
            .d_model = d_model,
            .inter = inter,
            .eps = 1e-6f,
            .residual = &t_residual,
            .ffn_norm_weight = &t_norm,
            .gate_weight = &tw,
            .up_weight = &tw,
            .down_weight = &t_down,
            .post_ffw_norm_weight = &t_post_norm,
            .pre_ff_scratch = &t_pre,
            .gate_scratch = &t_gate,
            .up_scratch = &t_up,
            .ffn_out_scratch = &t_ffn_out,
            .post_ff_scratch = &t_post,
            .out = &t_out,
        };
        s = v->ffn_geglu_block(be, &block);
        if (s == GEIST_OK) {
            const double tfwarm = now_ms();
            s = v->ffn_geglu_block(be, &block);
            const double ffn_single_ms = now_ms() - tfwarm;
            int ffn_iter = (int) (200.0 / (ffn_single_ms + 0.001)) + 3;
            if (ffn_iter > 1000) {
                ffn_iter = 1000;
            }
            if (ffn_iter < 3) {
                ffn_iter = 3;
            }
            const double tf0 = now_ms();
            for (int it = 0; it < ffn_iter; it++) {
                s = v->ffn_geglu_block(be, &block);
                if (s != GEIST_OK) {
                    break;
                }
            }
            const double ffn_dt_ms = (now_ms() - tf0) / ffn_iter;
            if (s == GEIST_OK) {
                s = v->buffer_download(rows * d_model * sizeof(float),
                                       (uint8_t *) out_host, out_b);
            }
            if (s == GEIST_OK) {
                const double logical_bytes =
                    (double) rows *
                    (2.0 * (double) weight_bytes + (double) down_bytes);
                printf("  %-32s [Metal FFN Q4_K m=%zu] logical %7.1f MB  %7.2f ms  %5.2f GB/s  (%d it)\n",
                       name,
                       rows,
                       logical_bytes / (1024.0 * 1024.0),
                       ffn_dt_ms,
                       logical_bytes / (ffn_dt_ms * 1e6),
                       ffn_iter);
                fflush(stdout);
                if (out_host[0] == 0.0f &&
                    out_host[rows * d_model - 1] == 0.0f) {
                    fprintf(stderr, "(zero Metal FFN output)\n");
                }
            }
            if (s == GEIST_OK &&
                getenv("GEIST_Q4K_BENCH_METAL_SEQUENCE") != NULL &&
                v->command_sequence_begin != NULL &&
                v->command_sequence_end != NULL) {
                int seq_iter = ffn_iter;
                if (seq_iter > 128) {
                    seq_iter = 128;
                }
                int token = 0;
                const double ts0 = now_ms();
                s = v->command_sequence_begin(
                    be, GEIST_COMMAND_SEQUENCE_DECODE_LAYER_LOOP, &token);
                if (s == GEIST_OK) {
                    for (int it = 0; it < seq_iter; it++) {
                        s = v->ffn_geglu_block(be, &block);
                        if (s != GEIST_OK) {
                            break;
                        }
                    }
                    enum geist_status es =
                        v->command_sequence_end(be, token, s == GEIST_OK);
                    if (s == GEIST_OK) {
                        s = es;
                    }
                }
                const double seq_dt_ms = (now_ms() - ts0) / (double) seq_iter;
                if (s == GEIST_OK) {
                    const double logical_bytes =
                        (double) rows *
                        (2.0 * (double) weight_bytes + (double) down_bytes);
                    printf("  %-32s [Metal FFN Q4_K m=%zu seq] logical %7.1f MB  %7.2f ms  %5.2f GB/s  (%d blocks)\n",
                           name,
                           rows,
                           logical_bytes / (1024.0 * 1024.0),
                           seq_dt_ms,
                           logical_bytes / (seq_dt_ms * 1e6),
                           seq_iter);
                    fflush(stdout);
                } else {
                    fprintf(stderr,
                            "  %-32s [Metal FFN Q4_K m=%zu seq] FAIL (%s)\n",
                            name, rows, geist_backend_errmsg(be));
                }
            }
        }
        if (s != GEIST_OK) {
            fprintf(stderr, "  %-32s [Metal FFN Q4_K m=%zu] FAIL (%s)\n",
                    name, rows, geist_backend_errmsg(be));
        }

ffn_done:
        if (residual_b != NULL) { v->buffer_destroy(be, residual_b); }
        if (norm_b != NULL) { v->buffer_destroy(be, norm_b); }
        if (post_norm_b != NULL) { v->buffer_destroy(be, post_norm_b); }
        if (down_b != NULL) { v->buffer_destroy(be, down_b); }
        if (pre_b != NULL) { v->buffer_destroy(be, pre_b); }
        if (gate_b != NULL) { v->buffer_destroy(be, gate_b); }
        if (up_b != NULL) { v->buffer_destroy(be, up_b); }
        if (ffn_out_b != NULL) { v->buffer_destroy(be, ffn_out_b); }
        if (post_b != NULL) { v->buffer_destroy(be, post_b); }
        if (out_b != NULL) { v->buffer_destroy(be, out_b); }
        free(residual_host);
        free(norm_host);
        free(post_norm_host);
        free(out_host);
        free(down_host);
    }

done:
    if (x != NULL) { v->buffer_destroy(be, x); }
    if (w != NULL) { v->buffer_destroy(be, w); }
    if (y != NULL) { v->buffer_destroy(be, y); }
    free(x_host);
    free(y_host);
    geist_backend_destroy(be);
}

static void bench_one(const struct gguf_ctx *ctx,
                      const struct gguf_tensor_t* t,
                      const char* name) {
    const size_t n_in = t->dims[0];
    const size_t n_out = t->dims[1];
    const size_t m_bench =
            (getenv("GEIST_BENCH_M") != NULL) ? (size_t) atoi(getenv("GEIST_BENCH_M")) : 8;
    size_t bytes_per_call;
    const char* kind;
    switch (t->dtype) {
    case GGUF_TYPE_Q4_K:
        if (n_in % Q4_K_BLOCK_ELEMS != 0)
            goto skip_align;
        bytes_per_call = (n_out * n_in / Q4_K_BLOCK_ELEMS) * Q4_K_BLOCK_BYTES;
        kind = "Q4_K W4A8";
        break;
    case GGUF_TYPE_Q6_K:
        if (n_in % Q6_K_BLOCK_ELEMS != 0)
            goto skip_align;
        bytes_per_call = (n_out * n_in / Q6_K_BLOCK_ELEMS) * Q6_K_BLOCK_BYTES;
        kind = "Q6_K fp32 ";
        break;
    default:
        fprintf(stderr, "  %-32s SKIP (dtype %d)\n", name, t->dtype);
        return;
    }

    /* Allocate workspace. Q4_K path also needs pre-quantized x; Q6_K is FP32-in. */
    float* x = (float*) aligned_alloc(64, n_in * sizeof(float));
    int8_t* x_q8 = (int8_t*) aligned_alloc(64, n_in * sizeof(int8_t));
    int32_t* sum32 = (int32_t*) aligned_alloc(64, (n_in / 32) * sizeof(int32_t));
    float* y = (float*) aligned_alloc(64, n_out * sizeof(float));
    if (!x || !x_q8 || !sum32 || !y) {
        fprintf(stderr, "  %-32s SKIP (alloc fail)\n", name);
        free(x);
        free(x_q8);
        free(sum32);
        free(y);
        return;
    }
    for (size_t i = 0; i < n_in; i++)
        x[i] = ((float) i * 0.0137f) - 7.3f;
    float scale_x = 0.0f;
    if (t->dtype == GGUF_TYPE_Q4_K)
        scale_x = quantize_x_for_q4k(x, n_in, x_q8, sum32);

#define CALL_KERNEL()                                                                  \
    do {                                                                               \
        if (t->dtype == GGUF_TYPE_Q4_K)                                                \
            linear_q4k_decode_w4a8_pre(x_q8, scale_x, sum32, t->data, n_in, n_out, y); \
        else                                                                           \
            linear_q6k_decode_fp32(x, t->data, n_in, n_out, y);                        \
    } while (0)

    /* Calibrate iter count so the run takes ≥ 200ms. */
    const double t_warm = now_ms();
    CALL_KERNEL();
    const double single_ms = now_ms() - t_warm;
    int n_iter = (int) (200.0 / (single_ms + 0.001)) + 5;
    if (n_iter > 5000)
        n_iter = 5000;
    if (n_iter < 3)
        n_iter = 3;

    /* Hot loop. */
    const double t0 = now_ms();
    for (int it = 0; it < n_iter; it++)
        CALL_KERNEL();
    const double dt_ms = (now_ms() - t0) / n_iter;
    const double gbps = (double) bytes_per_call / (dt_ms * 1e6);
    const double mb_per_call = (double) bytes_per_call / (1024.0 * 1024.0);

    printf("  %-32s [%s] n_out=%6zu n_in=%5zu %7.1f MB  %7.2f ms  %5.2f GB/s  (%d it)\n",
           name,
           kind,
           n_out,
           n_in,
           mb_per_call,
           dt_ms,
           gbps,
           n_iter);
    fflush(stdout);

    /* Anti-DCE sink. */
    if (y[0] == 0.0f && y[n_out - 1] == 0.0f)
        fprintf(stderr, "(zero output)\n");

    /* For Q4_K only: bench m=8 prefill mtile4 vs mtile8 against the
     * pre-decoded weight format, single-thread. Comparison is on the
     * same dispatch path with the same activation quant; only the
     * inner M-tile width differs. */
    if (t->dtype == GGUF_TYPE_Q4_K) {
        const size_t M_BENCH = m_bench;
        const size_t n_chunks = n_in / 32;
        const size_t pd_bytes = q4k_predecode_size_bytes(n_in, n_out);
        const size_t pd_nt_bytes = q4k_predecode_ntile4_size_bytes(n_in, n_out);
        void* packed = malloc(pd_bytes);
        void* packed_nt = malloc(pd_nt_bytes);
        int8_t* xm_q8 = (int8_t*) malloc(M_BENCH * n_in * sizeof(int8_t));
        int32_t* xm_sum = (int32_t*) malloc(M_BENCH * n_chunks * sizeof(int32_t));
        float* xm_sx = (float*) malloc(M_BENCH * sizeof(float));
        float* ym4 = (float*) malloc(M_BENCH * n_out * sizeof(float));
        float* ym8 = (float*) malloc(M_BENCH * n_out * sizeof(float));
        float* ymn84 = (float*) malloc(M_BENCH * n_out * sizeof(float));
        int pack_rc = 1;
        int pack_nt_rc = 1;
        if (packed && packed_nt && xm_q8 && xm_sum && xm_sx && ym4 && ym8 && ymn84) {
            pack_rc = q4k_predecode_pack(t->data, n_in, n_out, packed);
            pack_nt_rc = q4k_predecode_ntile4_pack(t->data, n_in, n_out, packed_nt);
        } else {
            fprintf(stderr,
                    "  %-32s [Q4_K m=%zu] alloc failed (packed=%p xm_q8=%p xm_sum=%p xm_sx=%p "
                    "ym4=%p ym8=%p)\n",
                    name,
                    M_BENCH,
                    (void*) packed,
                    (void*) xm_q8,
                    (void*) xm_sum,
                    (void*) xm_sx,
                    (void*) ym4,
                    (void*) ym8);
        }
        if (pack_rc != 0) {
            fprintf(stderr, "  %-32s [Q4_K m=%zu] predecode_pack rc=%d\n", name, M_BENCH, pack_rc);
        }
        if (pack_rc == 0) {
            for (size_t r = 0; r < M_BENCH; r++) {
                for (size_t i = 0; i < n_in; i++) {
                    x[i] = ((float) ((r * 13 + i) % 4096) * 0.0019f) - 3.9f;
                }
                xm_sx[r] = quantize_x_for_q4k(x, n_in, xm_q8 + r * n_in, xm_sum + r * n_chunks);
            }
            const double tw4 = now_ms();
            linear_q4k_w4a8_prefill_predecoded_mtile4(
                    xm_q8, xm_sx, xm_sum, M_BENCH, packed, n_in, n_out, ym4);
            const double s4 = now_ms() - tw4;
            int it4 = (int) (200.0 / (s4 + 0.001)) + 3;
            if (it4 > 2000)
                it4 = 2000;
            if (it4 < 3)
                it4 = 3;
            const double t04 = now_ms();
            for (int it = 0; it < it4; it++)
                linear_q4k_w4a8_prefill_predecoded_mtile4(
                        xm_q8, xm_sx, xm_sum, M_BENCH, packed, n_in, n_out, ym4);
            const double dt4 = (now_ms() - t04) / it4;

            const double tw8 = now_ms();
            linear_q4k_w4a8_prefill_predecoded_mtile8(
                    xm_q8, xm_sx, xm_sum, M_BENCH, packed, n_in, n_out, ym8);
            const double s8 = now_ms() - tw8;
            int it8 = (int) (200.0 / (s8 + 0.001)) + 3;
            if (it8 > 2000)
                it8 = 2000;
            if (it8 < 3)
                it8 = 3;
            const double t08 = now_ms();
            for (int it = 0; it < it8; it++)
                linear_q4k_w4a8_prefill_predecoded_mtile8(
                        xm_q8, xm_sx, xm_sum, M_BENCH, packed, n_in, n_out, ym8);
            const double dt8 = (now_ms() - t08) / it8;
            (void) (dt4 / dt8); /* legacy speedup (mt4 → mt8) — see new columns below */

            /* SGEMM path: dequant Q4_K → fp32 + cblas_sgemm per tile. */
            extern void cblas_sgemm(int,
                                    int,
                                    int,
                                    int,
                                    int,
                                    int,
                                    float,
                                    const float*,
                                    int,
                                    const float*,
                                    int,
                                    float,
                                    float*,
                                    int);
            extern void dequant_q4_K_row(const void*, float*, size_t);
            const int Cb_RowMajor = 101, Cb_NoTrans = 111, Cb_Trans = 112;
            const size_t DEQ_TILE = 32;
            double dt_sg = 0.0;
            float* tile = (float*) malloc(DEQ_TILE * n_in * sizeof(float));
            float* x_fp32 = (float*) malloc(M_BENCH * n_in * sizeof(float));
            if (tile && x_fp32) {
                for (size_t i = 0; i < M_BENCH * n_in; i++) {
                    x_fp32[i] = ((float) ((i * 17) % 4096) * 0.0019f) - 3.9f;
                }
                const size_t blk_bytes = (n_in / Q4_K_BLOCK_ELEMS) * 144; /* Q4_K_BLOCK_BYTES */
                /* warm */
                for (size_t r0 = 0; r0 < n_out; r0 += DEQ_TILE) {
                    const size_t tr = (n_out - r0 < DEQ_TILE) ? (n_out - r0) : DEQ_TILE;
                    dequant_q4_K_row((const uint8_t*) t->data + r0 * blk_bytes, tile, tr * n_in);
                    cblas_sgemm(Cb_RowMajor,
                                Cb_NoTrans,
                                Cb_Trans,
                                (int) M_BENCH,
                                (int) tr,
                                (int) n_in,
                                1.0f,
                                x_fp32,
                                (int) n_in,
                                tile,
                                (int) n_in,
                                0.0f,
                                ym8 + r0,
                                (int) n_out);
                }
                const double tw_sg = now_ms();
                for (size_t r0 = 0; r0 < n_out; r0 += DEQ_TILE) {
                    const size_t tr = (n_out - r0 < DEQ_TILE) ? (n_out - r0) : DEQ_TILE;
                    dequant_q4_K_row((const uint8_t*) t->data + r0 * blk_bytes, tile, tr * n_in);
                    cblas_sgemm(Cb_RowMajor,
                                Cb_NoTrans,
                                Cb_Trans,
                                (int) M_BENCH,
                                (int) tr,
                                (int) n_in,
                                1.0f,
                                x_fp32,
                                (int) n_in,
                                tile,
                                (int) n_in,
                                0.0f,
                                ym8 + r0,
                                (int) n_out);
                }
                const double s_sg = now_ms() - tw_sg;
                int it_sg = (int) (200.0 / (s_sg + 0.001)) + 3;
                if (it_sg > 1000)
                    it_sg = 1000;
                if (it_sg < 3)
                    it_sg = 3;
                const double t0_sg = now_ms();
                for (int it = 0; it < it_sg; it++) {
                    for (size_t r0 = 0; r0 < n_out; r0 += DEQ_TILE) {
                        const size_t tr = (n_out - r0 < DEQ_TILE) ? (n_out - r0) : DEQ_TILE;
                        dequant_q4_K_row(
                                (const uint8_t*) t->data + r0 * blk_bytes, tile, tr * n_in);
                        cblas_sgemm(Cb_RowMajor,
                                    Cb_NoTrans,
                                    Cb_Trans,
                                    (int) M_BENCH,
                                    (int) tr,
                                    (int) n_in,
                                    1.0f,
                                    x_fp32,
                                    (int) n_in,
                                    tile,
                                    (int) n_in,
                                    0.0f,
                                    ym8 + r0,
                                    (int) n_out);
                    }
                }
                dt_sg = (now_ms() - t0_sg) / it_sg;
            }
            free(tile);
            free(x_fp32);
            printf("  %-32s [Q4_K m=%zu sgemm path] dt_sgemm=%6.2f ms (vs mt8=%5.2f → "
                   "Δ=%+5.1f%%)\n",
                   name,
                   M_BENCH,
                   dt_sg,
                   dt8,
                   dt_sg > 0 ? (dt8 / dt_sg - 1.0) * 100.0 : 0.0);
            fflush(stdout);

            /* mtile4_ntile4_packed on the ntile4 packed format. */
            double dt44n = 0.0;
            if (pack_nt_rc == 0) {
                const double tw44n = now_ms();
                linear_q4k_w4a8_prefill_predecoded_mtile4_ntile4_packed(
                        xm_q8, xm_sx, xm_sum, M_BENCH, packed_nt, n_in, n_out, ymn84);
                const double s44n = now_ms() - tw44n;
                int it44n = (int) (200.0 / (s44n + 0.001)) + 3;
                if (it44n > 2000)
                    it44n = 2000;
                if (it44n < 3)
                    it44n = 3;
                const double t044n = now_ms();
                for (int it = 0; it < it44n; it++)
                    linear_q4k_w4a8_prefill_predecoded_mtile4_ntile4_packed(
                            xm_q8, xm_sx, xm_sum, M_BENCH, packed_nt, n_in, n_out, ymn84);
                dt44n = (now_ms() - t044n) / it44n;
            }
            /* mtile8_ntile4_packed on the ntile4 packed format. */
            double dt84 = 0.0;
            if (pack_nt_rc == 0) {
                const double tw84 = now_ms();
                linear_q4k_w4a8_prefill_predecoded_mtile8_ntile4_packed(
                        xm_q8, xm_sx, xm_sum, M_BENCH, packed_nt, n_in, n_out, ymn84);
                const double s84 = now_ms() - tw84;
                int it84 = (int) (200.0 / (s84 + 0.001)) + 3;
                if (it84 > 2000)
                    it84 = 2000;
                if (it84 < 3)
                    it84 = 3;
                const double t084 = now_ms();
                for (int it = 0; it < it84; it++)
                    linear_q4k_w4a8_prefill_predecoded_mtile8_ntile4_packed(
                            xm_q8, xm_sx, xm_sum, M_BENCH, packed_nt, n_in, n_out, ymn84);
                dt84 = (now_ms() - t084) / it84;
            }
            const double speedup84 = (dt84 > 0) ? dt44n / dt84 : 0.0;
            const double speedup8_vs_44n = (dt84 > 0) ? dt44n / dt8 : 0.0;
            printf("  %-32s [Q4_K m=%zu] mt4=%5.2f mt4_nt4p=%5.2f mt8=%5.2f mt8_nt4p=%5.2f ms | "
                   "mt8 vs mt4_nt4p Δ=%+5.1f%% | mt8_nt4p vs mt4_nt4p Δ=%+5.1f%%\n",
                   name,
                   M_BENCH,
                   dt4,
                   dt44n,
                   dt8,
                   dt84,
                   (speedup8_vs_44n - 1.0) * 100.0,
                   (speedup84 - 1.0) * 100.0);
            fflush(stdout);
            bench_vulkan_qk(t, name, 1, bytes_per_call,
                            GEIST_DTYPE_Q4_K, "Q4_K");
            bench_vulkan_qk(t, name, M_BENCH, bytes_per_call,
                            GEIST_DTYPE_Q4_K, "Q4_K");
            bench_metal_q4k(t, name, bytes_per_call, M_BENCH);
            bench_metal_attention_q4k(ctx, t, name);
            bench_metal_layer_gemma_q4q6(ctx, t, name);
            bench_metal_ffn_gemma_q4q6(ctx, t, name, M_BENCH);
        }
        free(packed);
        free(packed_nt);
        free(xm_q8);
        free(xm_sum);
        free(xm_sx);
        free(ym4);
        free(ym8);
        free(ymn84);
    }

    /* For Q6_K only: also bench (a) the W6A8 NEON variant and (b) the
     * dequant→FP32 sgemv path used today for lm_head. Verify W6A8 matches
     * the FP32 reference numerically (cosine similarity ≥ 0.999). */
    if (t->dtype == GGUF_TYPE_Q6_K) {
        /* Reference output from FP32 path stored in `y` from the calibration
         * runs above. Make a copy then re-compute via W6A8 to compare. */
        float* y_ref = (float*) aligned_alloc(64, n_out * sizeof(float));
        memcpy(y_ref, y, n_out * sizeof(float));

        /* W6A8: needs symmetric int8 quantization of x (no sum32 needed). */
        const float scale_x6 = quantize_x_int8_sym(x, n_in, x_q8);
        linear_q6k_decode_w6a8_pre(x_q8, scale_x6, t->data, n_in, n_out, y);

        /* Cosine similarity vs reference. */
        double dot = 0, na = 0, nb = 0;
        for (size_t i = 0; i < n_out; i++) {
            dot += (double) y_ref[i] * y[i];
            na += (double) y_ref[i] * y_ref[i];
            nb += (double) y[i] * y[i];
        }
        const double cos_sim = dot / sqrt(na * nb);

        /* Bench W6A8 hot loop. */
        const double tw = now_ms();
        linear_q6k_decode_w6a8_pre(x_q8, scale_x6, t->data, n_in, n_out, y);
        const double single_ms3 = now_ms() - tw;
        int n_iter3 = (int) (200.0 / (single_ms3 + 0.001)) + 5;
        if (n_iter3 > 5000)
            n_iter3 = 5000;
        if (n_iter3 < 3)
            n_iter3 = 3;
        const double t03 = now_ms();
        for (int it = 0; it < n_iter3; it++)
            linear_q6k_decode_w6a8_pre(x_q8, scale_x6, t->data, n_in, n_out, y);
        const double dt_ms3 = (now_ms() - t03) / n_iter3;
        const double gbps3 = (double) bytes_per_call / (dt_ms3 * 1e6);
        printf("  %-32s [Q6_K W6A8] (same shape)         %7.1f MB  %7.2f ms  %5.2f GB/s  (%d it)  "
               "cos=%.6f\n",
               name,
               (double) bytes_per_call / (1024.0 * 1024.0),
               dt_ms3,
               gbps3,
               n_iter3,
               cos_sim);
        fflush(stdout);
        bench_vulkan_qk(t, name, 1, bytes_per_call,
                        GEIST_DTYPE_Q6_K, "Q6_K");
        bench_vulkan_qk(t, name, m_bench, bytes_per_call,
                        GEIST_DTYPE_Q6_K, "Q6_K");
        if (getenv("GEIST_Q4K_BENCH_METAL_Q6_SWEEP") != NULL) {
            (void) setenv("GEIST_METAL_Q6K_LINEAR_RAW", "0", 1);
            bench_metal_q6k(t, name, bytes_per_call, m_bench);
            (void) setenv("GEIST_METAL_Q6K_LINEAR_RAW", "1", 1);
            bench_metal_q6k(t, name, bytes_per_call, m_bench);
            (void) unsetenv("GEIST_METAL_Q6K_LINEAR_RAW");
        } else {
            bench_metal_q6k(t, name, bytes_per_call, m_bench);
        }

        free(y_ref);

        /* FP32 sgemv comparison only worth it for the lm_head size. */
        if (n_out * n_in >= 1024 * 1024) {
            float* w_fp32 = gguf_dequant_to_fp32(t);
            if (w_fp32) {
                const double t_warm2 = now_ms();
                linear_fp32(x, w_fp32, nullptr, 1, n_in, n_out, y);
                const double single_ms2 = now_ms() - t_warm2;
                int n_iter2 = (int) (200.0 / (single_ms2 + 0.001)) + 5;
                if (n_iter2 > 5000)
                    n_iter2 = 5000;
                if (n_iter2 < 3)
                    n_iter2 = 3;
                const double t02 = now_ms();
                for (int it = 0; it < n_iter2; it++)
                    linear_fp32(x, w_fp32, nullptr, 1, n_in, n_out, y);
                const double dt_ms2 = (now_ms() - t02) / n_iter2;
                const size_t bytes2 = n_out * n_in * sizeof(float);
                const double gbps2 = (double) bytes2 / (dt_ms2 * 1e6);
                const double mb2 = (double) bytes2 / (1024.0 * 1024.0);
                printf("  %-32s [FP32 sgemv] (same shape)        %7.1f MB  %7.2f ms  %5.2f GB/s  "
                       "(%d it)\n",
                       name,
                       mb2,
                       dt_ms2,
                       gbps2,
                       n_iter2);
                fflush(stdout);
                free(w_fp32);
            }
        }
    }

    free(x);
    free(x_q8);
    free(sum32);
    free(y);
    return;

skip_align:
    fprintf(stderr, "  %-32s SKIP (n_in %zu not block-aligned)\n", name, n_in);
}

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        fprintf(stderr, "Usage: %s <gguf> [<tensor_name>]\n", argv[0]);
        return 2;
    }
    const char* err = nullptr;
    struct gguf_ctx* ctx = gguf_open(argv[1], &err);
    if (!ctx) {
        fprintf(stderr, "gguf_open: %s\n", err);
        return 1;
    }

    printf("bench_q4k_kernel — single-thread W4A8 throughput on %s\n", argv[1]);
    printf("  shape                            (n_out × n_in)         size   wall            "
           "bandwidth\n");

    if (argc == 3) {
        const struct gguf_tensor_t* t = gguf_get_tensor(ctx, argv[2]);
        if (!t) {
            fprintf(stderr, "tensor not found: %s\n", argv[2]);
            return 1;
        }
        bench_one(ctx, t, argv[2]);
    } else {
        /* Canonical Gemma 4 E2B per-decode-call shapes (layer 0 representative)
         * + lm_head (token_embd is tied; Q6_K in Q4_K_M model). */
        static const char* shapes[] = {
                "blk.0.attn_q.weight",
                "blk.0.attn_k.weight",
                "blk.0.attn_output.weight",
                "blk.0.ffn_gate.weight",
                "blk.0.ffn_up.weight",
                "blk.0.ffn_down.weight",
                "token_embd.weight", /* aliased as lm_head — DOMINANT decode cost */
                nullptr,
        };
        for (int i = 0; shapes[i]; i++) {
            const struct gguf_tensor_t* t = gguf_get_tensor(ctx, shapes[i]);
            if (!t) {
                fprintf(stderr, "  %-32s NOT FOUND\n", shapes[i]);
                continue;
            }
            bench_one(ctx, t, shapes[i]);
        }
    }

    gguf_close(ctx);
    return 0;
}
