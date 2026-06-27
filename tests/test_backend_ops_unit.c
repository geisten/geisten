/*
 * test_backend_ops_unit — verifies the elementwise + rmsnorm ops in both
 * cpu_scalar and cpu_neon backends produce the same outputs.
 *
 * Phase B-4e foundation: with these ops in the backend vtable, lm.c's
 * forward pass can be migrated op-by-op through backend->vtbl->* instead
 * of direct calls into gemma4_kernels.c.
 */
#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int check(bool cond, const char* what) {
    if (!cond) {
        fprintf(stderr, "FAIL: %s\n", what);
        return 1;
    }
    return 0;
}

static bool backend_available(const char *backend_name) {
    struct geist_backend *be = nullptr;
    const enum geist_status s =
        geist_backend_create(backend_name, nullptr, nullptr, &be);
    if (s != GEIST_OK) {
        return false;
    }
    geist_backend_destroy(be);
    return true;
}

/* Build a 1-row F32 DENSE tensor backed by an existing buffer. */
static struct geist_tensor make_tensor_1d(struct geist_buffer* buf, size_t n) {
    return (struct geist_tensor) {
            .buffer = buf,
            .offset = 0,
            .dtype = GEIST_DTYPE_F32,
            .layout = GEIST_LAYOUT_DENSE,
            .ndim = 1,
            .shape = {(int64_t) n},
            .stride = {1},
    };
}

static struct geist_tensor make_tensor_2d(struct geist_buffer* buf,
                                          size_t rows,
                                          size_t cols) {
    return (struct geist_tensor) {
            .buffer = buf,
            .offset = 0,
            .dtype = GEIST_DTYPE_F32,
            .layout = GEIST_LAYOUT_DENSE,
            .ndim = 2,
            .shape = {(int64_t) rows, (int64_t) cols},
            .stride = {(int64_t) cols, 1},
    };
}

static struct geist_tensor make_tensor_q4k_2d(struct geist_buffer* buf,
                                              size_t rows,
                                              size_t cols) {
    return (struct geist_tensor) {
            .buffer = buf,
            .offset = 0,
            .dtype = GEIST_DTYPE_Q4_K,
            .layout = GEIST_LAYOUT_BLOCK_QUANTIZED,
            .ndim = 2,
            .shape = {(int64_t) rows, (int64_t) cols},
            .stride = {0, 0},
    };
}

static struct geist_tensor make_tensor_q6k_2d(struct geist_buffer* buf,
                                              size_t rows,
                                              size_t cols) {
    return (struct geist_tensor) {
            .buffer = buf,
            .offset = 0,
            .dtype = GEIST_DTYPE_Q6_K,
            .layout = GEIST_LAYOUT_BLOCK_QUANTIZED,
            .ndim = 2,
            .shape = {(int64_t) rows, (int64_t) cols},
            .stride = {0, 0},
    };
}

static struct geist_tensor make_tensor_3d(struct geist_buffer* buf,
                                          size_t d0,
                                          size_t d1,
                                          size_t d2) {
    return (struct geist_tensor) {
            .buffer = buf,
            .offset = 0,
            .dtype = GEIST_DTYPE_F32,
            .layout = GEIST_LAYOUT_DENSE,
            .ndim = 3,
            .shape = {(int64_t) d0, (int64_t) d1, (int64_t) d2},
            .stride = {(int64_t) (d1 * d2), (int64_t) d2, 1},
    };
}

/* Cross-reference a single op across the two backends. */
static int compare_op_outputs(const char* op_name,
                              const float* out_a,
                              const float* out_b,
                              size_t n,
                              float rtol,
                              float atol) {
    ptrdiff_t bad = geist_fp32_close_array(out_a, out_b, n, rtol, atol);
    if (bad >= 0) {
        fprintf(stderr,
                "FAIL %s: idx %td: scalar=%.6f neon=%.6f\n",
                op_name,
                bad,
                (double) out_a[bad],
                (double) out_b[bad]);
        return 1;
    }
    return 0;
}

/* Helper: build a backend, run an op of choice, get scalar host buffer back.
 * Returns 0 on success, -1 on error. */
static int
run_add(const char* backend_name, const float* a_in, const float* b_in, size_t n, float* out_host) {
    struct geist_backend* be = nullptr;
    if (geist_backend_create(backend_name, nullptr, nullptr, &be) != GEIST_OK) {
        return -1;
    }
    struct geist_buffer *ba = nullptr, *bb = nullptr, *by = nullptr;
    (void) be->desc->vtbl->buffer_create(
            be, n * sizeof(float), GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_AUTO, &ba);
    (void) be->desc->vtbl->buffer_create(
            be, n * sizeof(float), GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_AUTO, &bb);
    (void) be->desc->vtbl->buffer_create(
            be, n * sizeof(float), GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_AUTO, &by);
    (void) be->desc->vtbl->buffer_upload(ba, n * sizeof(float), (const uint8_t*) a_in);
    (void) be->desc->vtbl->buffer_upload(bb, n * sizeof(float), (const uint8_t*) b_in);
    struct geist_tensor a = make_tensor_1d(ba, n);
    struct geist_tensor b = make_tensor_1d(bb, n);
    struct geist_tensor y = make_tensor_1d(by, n);
    enum geist_status s = be->desc->vtbl->add(be, &a, &b, &y);
    if (s != GEIST_OK) {
        geist_backend_destroy(be);
        return -1;
    }
    (void) be->desc->vtbl->buffer_download(n * sizeof(float), (uint8_t*) out_host, by);
    (void) be->desc->vtbl->buffer_destroy(be, ba);
    (void) be->desc->vtbl->buffer_destroy(be, bb);
    (void) be->desc->vtbl->buffer_destroy(be, by);
    geist_backend_destroy(be);
    return 0;
}

static int run_mul(const char *backend_name,
                   const float *a_in,
                   const float *b_in,
                   size_t n,
                   unsigned int memory_flags,
                   bool expect_device_only,
                   float *out_host) {
    struct geist_backend *be = nullptr;
    if (geist_backend_create(backend_name, nullptr, nullptr, &be) != GEIST_OK) {
        return -1;
    }
    if (be->desc->vtbl->mul == nullptr) {
        geist_backend_destroy(be);
        return -1;
    }

    struct geist_buffer *ba = nullptr;
    struct geist_buffer *bb = nullptr;
    struct geist_buffer *by = nullptr;
    enum geist_status s = be->desc->vtbl->buffer_create(
            be, n * sizeof(float), GEIST_BUFFER_ACTIVATION,
            memory_flags, &ba);
    if (s == GEIST_OK) {
        s = be->desc->vtbl->buffer_create(
                be, n * sizeof(float), GEIST_BUFFER_ACTIVATION,
                memory_flags, &bb);
    }
    if (s == GEIST_OK) {
        s = be->desc->vtbl->buffer_create(
                be, n * sizeof(float), GEIST_BUFFER_ACTIVATION,
                memory_flags, &by);
    }
    if (s == GEIST_OK && expect_device_only &&
        be->desc->vtbl->buffer_map(ba) != nullptr) {
        s = GEIST_E_BACKEND;
    }
    if (s == GEIST_OK) {
        s = be->desc->vtbl->buffer_upload(
                ba, n * sizeof(float), (const uint8_t *) a_in);
    }
    if (s == GEIST_OK) {
        s = be->desc->vtbl->buffer_upload(
                bb, n * sizeof(float), (const uint8_t *) b_in);
    }
    if (s == GEIST_OK) {
        struct geist_tensor a = make_tensor_1d(ba, n);
        struct geist_tensor b = make_tensor_1d(bb, n);
        struct geist_tensor y = make_tensor_1d(by, n);
        s = be->desc->vtbl->mul(be, &a, &b, &y);
    }
    if (s == GEIST_OK) {
        s = be->desc->vtbl->buffer_download(
                n * sizeof(float), (uint8_t *) out_host, by);
    }

    if (ba != nullptr) {
        be->desc->vtbl->buffer_destroy(be, ba);
    }
    if (bb != nullptr) {
        be->desc->vtbl->buffer_destroy(be, bb);
    }
    if (by != nullptr) {
        be->desc->vtbl->buffer_destroy(be, by);
    }
    geist_backend_destroy(be);
    return s == GEIST_OK ? 0 : -1;
}

static int run_scale(const char *backend_name,
                     const float *x_in,
                     size_t n,
                     float scale,
                     unsigned int memory_flags,
                     bool expect_device_only,
                     float *out_host) {
    struct geist_backend *be = nullptr;
    if (geist_backend_create(backend_name, nullptr, nullptr, &be) != GEIST_OK) {
        return -1;
    }
    if (be->desc->vtbl->scale_f32 == nullptr) {
        geist_backend_destroy(be);
        return -1;
    }

    struct geist_buffer *bx = nullptr;
    struct geist_buffer *by = nullptr;
    enum geist_status s = be->desc->vtbl->buffer_create(
            be, n * sizeof(float), GEIST_BUFFER_ACTIVATION,
            memory_flags, &bx);
    if (s == GEIST_OK) {
        s = be->desc->vtbl->buffer_create(
                be, n * sizeof(float), GEIST_BUFFER_ACTIVATION,
                memory_flags, &by);
    }
    if (s == GEIST_OK && expect_device_only &&
        be->desc->vtbl->buffer_map(bx) != nullptr) {
        s = GEIST_E_BACKEND;
    }
    if (s == GEIST_OK) {
        s = be->desc->vtbl->buffer_upload(
                bx, n * sizeof(float), (const uint8_t *) x_in);
    }
    if (s == GEIST_OK) {
        struct geist_tensor x = make_tensor_2d(bx, 2, n / 2u);
        struct geist_tensor y = make_tensor_2d(by, 2, n / 2u);
        s = be->desc->vtbl->scale_f32(be, &x, scale, &y);
    }
    if (s == GEIST_OK) {
        s = be->desc->vtbl->buffer_download(
                n * sizeof(float), (uint8_t *) out_host, by);
    }

    if (bx != nullptr) {
        be->desc->vtbl->buffer_destroy(be, bx);
    }
    if (by != nullptr) {
        be->desc->vtbl->buffer_destroy(be, by);
    }
    geist_backend_destroy(be);
    return s == GEIST_OK ? 0 : -1;
}

static int run_gelu(const char* backend_name,
                    const float* x_in,
                    size_t n,
                    unsigned int memory_flags,
                    bool expect_device_only,
                    float* out_host) {
    struct geist_backend* be = nullptr;
    if (geist_backend_create(backend_name, nullptr, nullptr, &be) != GEIST_OK) {
        return -1;
    }
    if (be->desc->vtbl->gelu_tanh == nullptr) {
        geist_backend_destroy(be);
        return -1;
    }
    struct geist_buffer *bx = nullptr, *by = nullptr;
    enum geist_status s = be->desc->vtbl->buffer_create(
            be, n * sizeof(float), GEIST_BUFFER_ACTIVATION,
            memory_flags, &bx);
    if (s == GEIST_OK) {
        s = be->desc->vtbl->buffer_create(
                be, n * sizeof(float), GEIST_BUFFER_ACTIVATION,
                memory_flags, &by);
    }
    if (s == GEIST_OK && expect_device_only &&
        be->desc->vtbl->buffer_map(bx) != nullptr) {
        s = GEIST_E_BACKEND;
    }
    if (s == GEIST_OK) {
        s = be->desc->vtbl->buffer_upload(
                bx, n * sizeof(float), (const uint8_t*) x_in);
    }
    if (s == GEIST_OK) {
        struct geist_tensor x = make_tensor_1d(bx, n);
        struct geist_tensor y = make_tensor_1d(by, n);
        s = be->desc->vtbl->gelu_tanh(be, &x, &y);
    }
    if (s == GEIST_OK) {
        s = be->desc->vtbl->buffer_download(
                n * sizeof(float), (uint8_t*) out_host, by);
    }
    if (bx != nullptr) {
        be->desc->vtbl->buffer_destroy(be, bx);
    }
    if (by != nullptr) {
        be->desc->vtbl->buffer_destroy(be, by);
    }
    geist_backend_destroy(be);
    return s == GEIST_OK ? 0 : -1;
}

static int run_gelu_mul(const char *backend_name,
                        const float *x_in,
                        const float *z_in,
                        size_t n,
                        unsigned int memory_flags,
                        float *out_host) {
    struct geist_backend *be = nullptr;
    if (geist_backend_create(backend_name, nullptr, nullptr, &be) != GEIST_OK) {
        return -1;
    }
    if (be->desc->vtbl->gelu_tanh_mul == nullptr) {
        geist_backend_destroy(be);
        return -1;
    }
    struct geist_buffer *bx = nullptr, *bz = nullptr, *by = nullptr;
    enum geist_status s = be->desc->vtbl->buffer_create(
            be, n * sizeof(float), GEIST_BUFFER_ACTIVATION, memory_flags, &bx);
    if (s == GEIST_OK) {
        s = be->desc->vtbl->buffer_create(
                be, n * sizeof(float), GEIST_BUFFER_ACTIVATION, memory_flags, &bz);
    }
    if (s == GEIST_OK) {
        s = be->desc->vtbl->buffer_create(
                be, n * sizeof(float), GEIST_BUFFER_ACTIVATION, memory_flags, &by);
    }
    if (s == GEIST_OK) {
        s = be->desc->vtbl->buffer_upload(bx, n * sizeof(float),
                                          (const uint8_t*) x_in);
    }
    if (s == GEIST_OK) {
        s = be->desc->vtbl->buffer_upload(bz, n * sizeof(float),
                                          (const uint8_t*) z_in);
    }
    if (s == GEIST_OK) {
        struct geist_tensor x = make_tensor_2d(bx, 1, n);
        struct geist_tensor z = make_tensor_2d(bz, 1, n);
        struct geist_tensor y = make_tensor_2d(by, 1, n);
        s = be->desc->vtbl->gelu_tanh_mul(be, &x, &z, &y);
    }
    if (s == GEIST_OK) {
        s = be->desc->vtbl->buffer_download(n * sizeof(float),
                                            (uint8_t*) out_host, by);
    }
    if (bx != nullptr) {
        be->desc->vtbl->buffer_destroy(be, bx);
    }
    if (bz != nullptr) {
        be->desc->vtbl->buffer_destroy(be, bz);
    }
    if (by != nullptr) {
        be->desc->vtbl->buffer_destroy(be, by);
    }
    geist_backend_destroy(be);
    return s == GEIST_OK ? 0 : -1;
}

static int run_rmsnorm(const char* backend_name,
                       const float* x_in,
                       const float* w_in,
                       size_t n_rows,
                       size_t feat,
                       float eps,
                       float* out_host) {
    struct geist_backend* be = nullptr;
    if (geist_backend_create(backend_name, nullptr, nullptr, &be) != GEIST_OK) {
        return -1;
    }
    struct geist_buffer *bx = nullptr, *bw = nullptr, *by = nullptr;
    (void) be->desc->vtbl->buffer_create(
            be, n_rows * feat * sizeof(float), GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_AUTO, &bx);
    (void) be->desc->vtbl->buffer_create(
            be, feat * sizeof(float), GEIST_BUFFER_WEIGHT, GEIST_MEMORY_AUTO, &bw);
    (void) be->desc->vtbl->buffer_create(
            be, n_rows * feat * sizeof(float), GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_AUTO, &by);
    (void) be->desc->vtbl->buffer_upload(bx, n_rows * feat * sizeof(float), (const uint8_t*) x_in);
    (void) be->desc->vtbl->buffer_upload(bw, feat * sizeof(float), (const uint8_t*) w_in);
    struct geist_tensor x = {
            .buffer = bx,
            .offset = 0,
            .dtype = GEIST_DTYPE_F32,
            .layout = GEIST_LAYOUT_DENSE,
            .ndim = 2,
            .shape = {(int64_t) n_rows, (int64_t) feat},
            .stride = {(int64_t) feat, 1},
    };
    struct geist_tensor w = make_tensor_1d(bw, feat);
    struct geist_tensor y = x;
    y.buffer = by;
    enum geist_status s = be->desc->vtbl->rmsnorm(be, &x, &w, eps, &y);
    if (s != GEIST_OK) {
        geist_backend_destroy(be);
        return -1;
    }
    (void) be->desc->vtbl->buffer_download(n_rows * feat * sizeof(float), (uint8_t*) out_host, by);
    (void) be->desc->vtbl->buffer_destroy(be, bx);
    (void) be->desc->vtbl->buffer_destroy(be, bw);
    (void) be->desc->vtbl->buffer_destroy(be, by);
    geist_backend_destroy(be);
    return 0;
}

static int run_elementwise_reuse_chain(const char *backend_name,
                                       const float *a_in,
                                       const float *b_in,
                                       const float *w_in,
                                       size_t n_rows,
                                       size_t feat,
                                       unsigned int memory_flags,
                                       bool expect_device_only,
                                       float *out_host) {
    const size_t n = n_rows * feat;
    struct geist_backend *be = nullptr;
    if (geist_backend_create(backend_name, nullptr, nullptr, &be) != GEIST_OK) {
        return -1;
    }
    if (be->desc->vtbl->add == nullptr ||
        be->desc->vtbl->mul == nullptr ||
        be->desc->vtbl->scale_f32 == nullptr ||
        be->desc->vtbl->gelu_tanh == nullptr ||
        be->desc->vtbl->gelu_tanh_mul == nullptr ||
        be->desc->vtbl->rmsnorm == nullptr) {
        geist_backend_destroy(be);
        return -1;
    }

    struct geist_buffer *ba = nullptr;
    struct geist_buffer *bb = nullptr;
    struct geist_buffer *bc = nullptr;
    struct geist_buffer *bw = nullptr;
    enum geist_status s = be->desc->vtbl->buffer_create(
            be, n * sizeof(float), GEIST_BUFFER_ACTIVATION,
            memory_flags, &ba);
    if (s == GEIST_OK) {
        s = be->desc->vtbl->buffer_create(
                be, n * sizeof(float), GEIST_BUFFER_ACTIVATION,
                memory_flags, &bb);
    }
    if (s == GEIST_OK) {
        s = be->desc->vtbl->buffer_create(
                be, n * sizeof(float), GEIST_BUFFER_ACTIVATION,
                memory_flags, &bc);
    }
    if (s == GEIST_OK) {
        s = be->desc->vtbl->buffer_create(
                be, feat * sizeof(float), GEIST_BUFFER_WEIGHT,
                memory_flags, &bw);
    }
    if (s == GEIST_OK && expect_device_only &&
        be->desc->vtbl->buffer_map(ba) != nullptr) {
        s = GEIST_E_BACKEND;
    }
    if (s == GEIST_OK) {
        s = be->desc->vtbl->buffer_upload(
                ba, n * sizeof(float), (const uint8_t *) a_in);
    }
    if (s == GEIST_OK) {
        s = be->desc->vtbl->buffer_upload(
                bb, n * sizeof(float), (const uint8_t *) b_in);
    }
    if (s == GEIST_OK) {
        s = be->desc->vtbl->buffer_upload(
                bw, feat * sizeof(float), (const uint8_t *) w_in);
    }

    struct geist_tensor a = make_tensor_1d(ba, n);
    struct geist_tensor b = make_tensor_1d(bb, n);
    struct geist_tensor c = make_tensor_1d(bc, n);
    if (s == GEIST_OK) {
        s = be->desc->vtbl->add(be, &a, &b, &c);
    }
    if (s == GEIST_OK) {
        s = be->desc->vtbl->mul(be, &c, &b, &a);
    }
    if (s == GEIST_OK) {
        s = be->desc->vtbl->scale_f32(be, &a, 0.5f, &c);
    }
    if (s == GEIST_OK) {
        s = be->desc->vtbl->gelu_tanh(be, &c, &a);
    }
    if (s == GEIST_OK) {
        s = be->desc->vtbl->gelu_tanh_mul(be, &a, &b, &c);
    }
    if (s == GEIST_OK) {
        struct geist_tensor x = make_tensor_2d(bc, n_rows, feat);
        struct geist_tensor w = make_tensor_1d(bw, feat);
        struct geist_tensor y = make_tensor_2d(ba, n_rows, feat);
        s = be->desc->vtbl->rmsnorm(be, &x, &w, 1e-6f, &y);
    }
    if (s == GEIST_OK) {
        s = be->desc->vtbl->buffer_download(
                n * sizeof(float), (uint8_t *) out_host, ba);
    }

    if (ba != nullptr) {
        be->desc->vtbl->buffer_destroy(be, ba);
    }
    if (bb != nullptr) {
        be->desc->vtbl->buffer_destroy(be, bb);
    }
    if (bc != nullptr) {
        be->desc->vtbl->buffer_destroy(be, bc);
    }
    if (bw != nullptr) {
        be->desc->vtbl->buffer_destroy(be, bw);
    }
    geist_backend_destroy(be);
    return s == GEIST_OK ? 0 : -1;
}

static void ref_matvec_f32(const float *x,
                           const float *w,
                           size_t n_in,
                           size_t n_out,
                           float *y) {
    for (size_t row = 0; row < n_out; row++) {
        double acc = 0.0;
        for (size_t k = 0; k < n_in; k++) {
            acc += (double) x[k] * (double) w[row * n_in + k];
        }
        y[row] = (float) acc;
    }
}

static uint8_t q4k_test_value(size_t row,
                              size_t block,
                              size_t sub,
                              size_t idx) {
    return (uint8_t) ((row * 3u + block * 5u + sub * 7u + idx) & 15u);
}

static void pack_q4k_test_matrix(size_t n_in,
                                 size_t n_out,
                                 uint8_t *dst) {
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
                        q4k_test_value(row, block, pair * 2u, idx);
                    const uint8_t hi =
                        q4k_test_value(row, block, pair * 2u + 1u, idx);
                    b[16u + pair * 32u + idx] = (uint8_t) (lo | (hi << 4u));
                }
            }
        }
    }
}

static void ref_matvec_q4k_test(const float *x,
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
                           (double) q4k_test_value(row, block, sub, idx);
                }
            }
        }
        y[row] = (float) acc;
    }
}

static void ref_embedding_q4k_test(size_t n_in,
                                   size_t row,
                                   float scale,
                                   float *y) {
    const size_t blocks_per_row = n_in / 256u;
    for (size_t block = 0; block < blocks_per_row; block++) {
        for (size_t sub = 0; sub < 8u; sub++) {
            for (size_t idx = 0; idx < 32u; idx++) {
                const size_t k = block * 256u + sub * 32u + idx;
                y[k] = (float) q4k_test_value(row, block, sub, idx) * scale;
            }
        }
    }
}

static int8_t q6k_test_value(size_t row,
                             size_t block,
                             size_t half_idx,
                             size_t stream,
                             size_t idx) {
    const size_t v = row * 5u + block * 7u + half_idx * 11u +
                     stream * 3u + idx;
    return (int8_t) ((int) (v & 15u) - 8);
}

static void pack_q6k_test_matrix(size_t n_in,
                                 size_t n_out,
                                 uint8_t *dst) {
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
                    const uint8_t q0 = (uint8_t)
                        (q6k_test_value(row, block, half_idx, 0u, idx) + 32);
                    const uint8_t q1 = (uint8_t)
                        (q6k_test_value(row, block, half_idx, 1u, idx) + 32);
                    const uint8_t q2 = (uint8_t)
                        (q6k_test_value(row, block, half_idx, 2u, idx) + 32);
                    const uint8_t q3 = (uint8_t)
                        (q6k_test_value(row, block, half_idx, 3u, idx) + 32);
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

static void ref_matvec_q6k_test(const float *x,
                                size_t n_in,
                                size_t n_out,
                                float *y) {
    const size_t blocks_per_row = n_in / 256u;
    for (size_t row = 0; row < n_out; row++) {
        double acc = 0.0;
        for (size_t block = 0; block < blocks_per_row; block++) {
            for (size_t half_idx = 0; half_idx < 2u; half_idx++) {
                for (size_t stream = 0; stream < 4u; stream++) {
                    for (size_t idx = 0; idx < 32u; idx++) {
                        const size_t k = block * 256u + half_idx * 128u +
                                         stream * 32u + idx;
                        acc += (double) x[k] *
                               (double) q6k_test_value(row, block, half_idx,
                                                       stream, idx);
                    }
                }
            }
        }
        y[row] = (float) acc;
    }
}

static void ref_embedding_q6k_test(size_t n_in,
                                   size_t row,
                                   float scale,
                                   float *y) {
    const size_t blocks_per_row = n_in / 256u;
    for (size_t block = 0; block < blocks_per_row; block++) {
        for (size_t half_idx = 0; half_idx < 2u; half_idx++) {
            for (size_t stream = 0; stream < 4u; stream++) {
                for (size_t idx = 0; idx < 32u; idx++) {
                    const size_t k = block * 256u + half_idx * 128u +
                                     stream * 32u + idx;
                    y[k] = (float) q6k_test_value(row, block, half_idx,
                                                  stream, idx) * scale;
                }
            }
        }
    }
}

static void ref_rope_f32(float *x,
                         const float *cos_v,
                         const float *sin_v,
                         size_t seq,
                         size_t heads,
                         size_t head_dim) {
    const size_t half_dim = head_dim / 2u;
    for (size_t s = 0; s < seq; s++) {
        const float *cos_s = cos_v + s * head_dim;
        const float *sin_s = sin_v + s * head_dim;
        for (size_t h = 0; h < heads; h++) {
            float *xh = x + (s * heads + h) * head_dim;
            for (size_t i = 0; i < half_dim; i++) {
                const float a = xh[i];
                const float b = xh[i + half_dim];
                xh[i] = a * cos_s[i] - b * sin_s[i];
                xh[i + half_dim] = b * cos_s[i + half_dim] +
                                    a * sin_s[i + half_dim];
            }
        }
    }
}

static void ref_attention_f32(const float *q,
                              const float *k,
                              const float *v,
                              size_t n_q,
                              size_t n_kv,
                              size_t q_offset,
                              size_t n_q_heads,
                              size_t n_kv_heads,
                              size_t head_dim,
                              size_t sliding_window,
                              float *out) {
    const size_t kv_group_size = n_q_heads / n_kv_heads;
    for (size_t t = 0; t < n_q; t++) {
        const size_t q_pos = q_offset + t;
        size_t s_lo = 0;
        if (sliding_window > 0 && q_pos + 1 > sliding_window) {
            s_lo = q_pos + 1 - sliding_window;
        }
        size_t s_hi = q_pos;
        if (s_hi >= n_kv) {
            s_hi = n_kv - 1;
        }
        for (size_t h = 0; h < n_q_heads; h++) {
            const size_t kv_h = h / kv_group_size;
            const float *qv = q + (t * n_q_heads + h) * head_dim;
            float max_score = -INFINITY;
            for (size_t s = s_lo; s <= s_hi; s++) {
                const float *kv = k + (s * n_kv_heads + kv_h) * head_dim;
                float dot = 0.0f;
                for (size_t i = 0; i < head_dim; i++) {
                    dot += qv[i] * kv[i];
                }
                if (dot > max_score) {
                    max_score = dot;
                }
            }
            float denom = 0.0f;
            for (size_t s = s_lo; s <= s_hi; s++) {
                const float *kv = k + (s * n_kv_heads + kv_h) * head_dim;
                float dot = 0.0f;
                for (size_t i = 0; i < head_dim; i++) {
                    dot += qv[i] * kv[i];
                }
                denom += expf(dot - max_score);
            }
            float *outv = out + (t * n_q_heads + h) * head_dim;
            for (size_t i = 0; i < head_dim; i++) {
                outv[i] = 0.0f;
            }
            for (size_t s = s_lo; s <= s_hi; s++) {
                const float *kv = k + (s * n_kv_heads + kv_h) * head_dim;
                float dot = 0.0f;
                for (size_t i = 0; i < head_dim; i++) {
                    dot += qv[i] * kv[i];
                }
                const float weight = expf(dot - max_score) / denom;
                const float *vv = v + (s * n_kv_heads + kv_h) * head_dim;
                for (size_t i = 0; i < head_dim; i++) {
                    outv[i] += weight * vv[i];
                }
            }
        }
    }
}

static int run_matvec(const char* backend_name,
                      const float* x_in,
                      const float* w_in,
                      size_t n_in,
                      size_t n_out,
                      float* out_host) {
    struct geist_backend* be = nullptr;
    if (geist_backend_create(backend_name, nullptr, nullptr, &be) != GEIST_OK) {
        return -1;
    }
    if (be->desc->vtbl->matvec_f32_dense == nullptr) {
        geist_backend_destroy(be);
        return -1;
    }

    struct geist_buffer *bx = nullptr, *bw = nullptr, *by = nullptr;
    enum geist_status s = be->desc->vtbl->buffer_create(
            be, n_in * sizeof(float), GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_AUTO, &bx);
    if (s == GEIST_OK) {
        s = be->desc->vtbl->buffer_create(
                be, n_out * n_in * sizeof(float), GEIST_BUFFER_WEIGHT, GEIST_MEMORY_AUTO, &bw);
    }
    if (s == GEIST_OK) {
        s = be->desc->vtbl->buffer_create(
                be, n_out * sizeof(float), GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_AUTO, &by);
    }
    if (s == GEIST_OK) {
        s = be->desc->vtbl->buffer_upload(bx, n_in * sizeof(float), (const uint8_t*) x_in);
    }
    if (s == GEIST_OK) {
        s = be->desc->vtbl->buffer_upload(bw, n_out * n_in * sizeof(float), (const uint8_t*) w_in);
    }
    if (s == GEIST_OK) {
        struct geist_tensor x = make_tensor_1d(bx, n_in);
        struct geist_tensor w = make_tensor_2d(bw, n_out, n_in);
        struct geist_tensor y = make_tensor_1d(by, n_out);
        s = be->desc->vtbl->matvec_f32_dense(be, &x, &w, &y);
    }
    if (s == GEIST_OK) {
        s = be->desc->vtbl->buffer_download(n_out * sizeof(float), (uint8_t*) out_host, by);
    }

    if (bx != nullptr) {
        be->desc->vtbl->buffer_destroy(be, bx);
    }
    if (bw != nullptr) {
        be->desc->vtbl->buffer_destroy(be, bw);
    }
    if (by != nullptr) {
        be->desc->vtbl->buffer_destroy(be, by);
    }
    geist_backend_destroy(be);
    return s == GEIST_OK ? 0 : -1;
}

static int run_matvec_q4k(const char* backend_name,
                          const float* x_in,
                          const uint8_t* w_in,
                          size_t n_in,
                          size_t n_out,
                          float* out_host) {
    struct geist_backend* be = nullptr;
    if (geist_backend_create(backend_name, nullptr, nullptr, &be) != GEIST_OK) {
        return -1;
    }
    if (be->desc->vtbl->matvec_q4k == nullptr) {
        geist_backend_destroy(be);
        return -1;
    }

    const size_t w_bytes = n_out * (n_in / 256u) * 144u;
    struct geist_buffer *bx = nullptr, *bw = nullptr, *by = nullptr;
    enum geist_status s = be->desc->vtbl->buffer_create(
            be, n_in * sizeof(float), GEIST_BUFFER_ACTIVATION,
            GEIST_MEMORY_DEVICE, &bx);
    if (s == GEIST_OK) {
        s = be->desc->vtbl->buffer_create(
                be, w_bytes, GEIST_BUFFER_WEIGHT, GEIST_MEMORY_DEVICE, &bw);
    }
    if (s == GEIST_OK) {
        s = be->desc->vtbl->buffer_create(
                be, n_out * sizeof(float), GEIST_BUFFER_ACTIVATION,
                GEIST_MEMORY_DEVICE, &by);
    }
    if (s == GEIST_OK) {
        s = be->desc->vtbl->buffer_upload(bx, n_in * sizeof(float),
                                          (const uint8_t*) x_in);
    }
    if (s == GEIST_OK) {
        s = be->desc->vtbl->buffer_upload(bw, w_bytes, w_in);
    }
    if (s == GEIST_OK) {
        struct geist_tensor x = make_tensor_1d(bx, n_in);
        struct geist_tensor w = make_tensor_q4k_2d(bw, n_out, n_in);
        struct geist_tensor y = make_tensor_1d(by, n_out);
        s = be->desc->vtbl->matvec_q4k(be, &x, &w, &y);
    }
    if (s == GEIST_OK) {
        s = be->desc->vtbl->buffer_download(n_out * sizeof(float),
                                            (uint8_t*) out_host, by);
    }

    if (bx != nullptr) {
        be->desc->vtbl->buffer_destroy(be, bx);
    }
    if (bw != nullptr) {
        be->desc->vtbl->buffer_destroy(be, bw);
    }
    if (by != nullptr) {
        be->desc->vtbl->buffer_destroy(be, by);
    }
    geist_backend_destroy(be);
    return s == GEIST_OK ? 0 : -1;
}

static int run_embedding_lookup(const char *backend_name,
                                const float *table_in,
                                size_t vocab,
                                size_t d_model,
                                geist_token_t token_id,
                                bool expect_device_only,
                                float *out_host) {
    struct geist_backend *be = nullptr;
    if (geist_backend_create(backend_name, nullptr, nullptr, &be) != GEIST_OK) {
        return -1;
    }
    if (be->desc->vtbl->embedding_lookup == nullptr) {
        geist_backend_destroy(be);
        return -1;
    }

    struct geist_buffer *bt = nullptr;
    struct geist_buffer *bo = nullptr;
    enum geist_status s = be->desc->vtbl->buffer_create(
        be, vocab * d_model * sizeof(float), GEIST_BUFFER_WEIGHT,
        expect_device_only ? GEIST_MEMORY_DEVICE : GEIST_MEMORY_AUTO, &bt);
    if (s == GEIST_OK) {
        s = be->desc->vtbl->buffer_create(
            be, d_model * sizeof(float), GEIST_BUFFER_ACTIVATION,
            expect_device_only ? GEIST_MEMORY_DEVICE : GEIST_MEMORY_AUTO, &bo);
    }
    if (s == GEIST_OK && expect_device_only &&
        be->desc->vtbl->buffer_map(bt) != nullptr) {
        s = GEIST_E_BACKEND;
    }
    if (s == GEIST_OK) {
        s = be->desc->vtbl->buffer_upload(
            bt, vocab * d_model * sizeof(float), (const uint8_t *) table_in);
    }
    if (s == GEIST_OK) {
        struct geist_tensor table = make_tensor_2d(bt, vocab, d_model);
        struct geist_tensor out = make_tensor_1d(bo, d_model);
        s = be->desc->vtbl->embedding_lookup(be, &table, token_id, &out);
    }
    if (s == GEIST_OK) {
        s = be->desc->vtbl->buffer_download(d_model * sizeof(float),
                                            (uint8_t *) out_host, bo);
    }

    if (bt != nullptr) {
        be->desc->vtbl->buffer_destroy(be, bt);
    }
    if (bo != nullptr) {
        be->desc->vtbl->buffer_destroy(be, bo);
    }
    geist_backend_destroy(be);
    return s == GEIST_OK ? 0 : -1;
}

static int run_embedding_lookup_typed(const char *backend_name,
                                      const void *table_in,
                                      size_t table_bytes,
                                      enum geist_dtype dtype,
                                      enum geist_layout layout,
                                      size_t vocab,
                                      size_t d_model,
                                      geist_token_t token_id,
                                      bool expect_device_only,
                                      float *out_host) {
    struct geist_backend *be = nullptr;
    if (geist_backend_create(backend_name, nullptr, nullptr, &be) != GEIST_OK) {
        return -1;
    }
    if (be->desc->vtbl->embedding_lookup == nullptr) {
        geist_backend_destroy(be);
        return -1;
    }

    struct geist_buffer *bt = nullptr;
    struct geist_buffer *bo = nullptr;
    enum geist_status s = be->desc->vtbl->buffer_create(
        be, table_bytes, GEIST_BUFFER_WEIGHT,
        expect_device_only ? GEIST_MEMORY_DEVICE : GEIST_MEMORY_AUTO, &bt);
    if (s == GEIST_OK) {
        s = be->desc->vtbl->buffer_create(
            be, d_model * sizeof(float), GEIST_BUFFER_ACTIVATION,
            expect_device_only ? GEIST_MEMORY_DEVICE : GEIST_MEMORY_AUTO, &bo);
    }
    if (s == GEIST_OK && expect_device_only &&
        be->desc->vtbl->buffer_map(bt) != nullptr) {
        s = GEIST_E_BACKEND;
    }
    if (s == GEIST_OK) {
        s = be->desc->vtbl->buffer_upload(
            bt, table_bytes, (const uint8_t *) table_in);
    }
    if (s == GEIST_OK) {
        struct geist_tensor table = {
            .buffer = bt,
            .offset = 0,
            .dtype = dtype,
            .layout = layout,
            .ndim = 2,
            .shape = {(int64_t) vocab, (int64_t) d_model},
        };
        if (layout == GEIST_LAYOUT_DENSE) {
            table.stride[0] = (int64_t) d_model;
            table.stride[1] = 1;
        }
        struct geist_tensor out = make_tensor_1d(bo, d_model);
        s = be->desc->vtbl->embedding_lookup(be, &table, token_id, &out);
    }
    if (s == GEIST_OK) {
        s = be->desc->vtbl->buffer_download(d_model * sizeof(float),
                                            (uint8_t *) out_host, bo);
    }

    if (bt != nullptr) {
        be->desc->vtbl->buffer_destroy(be, bt);
    }
    if (bo != nullptr) {
        be->desc->vtbl->buffer_destroy(be, bo);
    }
    geist_backend_destroy(be);
    return s == GEIST_OK ? 0 : -1;
}

static int run_embedding_lookup_scaled(const char *backend_name,
                                       const void *table_in,
                                       size_t table_bytes,
                                       enum geist_dtype dtype,
                                       enum geist_layout layout,
                                       size_t vocab,
                                       size_t d_model,
                                       geist_token_t token_id,
                                       float scale,
                                       bool expect_device_only,
                                       float *out_host) {
    struct geist_backend *be = nullptr;
    if (geist_backend_create(backend_name, nullptr, nullptr, &be) != GEIST_OK) {
        return -1;
    }
    if (be->desc->vtbl->embedding_lookup_scaled == nullptr) {
        geist_backend_destroy(be);
        return -1;
    }

    struct geist_buffer *bt = nullptr;
    struct geist_buffer *bo = nullptr;
    enum geist_status s = be->desc->vtbl->buffer_create(
        be, table_bytes, GEIST_BUFFER_WEIGHT,
        expect_device_only ? GEIST_MEMORY_DEVICE : GEIST_MEMORY_AUTO, &bt);
    if (s == GEIST_OK) {
        s = be->desc->vtbl->buffer_create(
            be, d_model * sizeof(float), GEIST_BUFFER_ACTIVATION,
            expect_device_only ? GEIST_MEMORY_DEVICE : GEIST_MEMORY_AUTO, &bo);
    }
    if (s == GEIST_OK && expect_device_only &&
        be->desc->vtbl->buffer_map(bt) != nullptr) {
        s = GEIST_E_BACKEND;
    }
    if (s == GEIST_OK) {
        s = be->desc->vtbl->buffer_upload(
            bt, table_bytes, (const uint8_t *) table_in);
    }
    if (s == GEIST_OK) {
        struct geist_tensor table = {
            .buffer = bt,
            .offset = 0,
            .dtype = dtype,
            .layout = layout,
            .ndim = 2,
            .shape = {(int64_t) vocab, (int64_t) d_model},
        };
        if (layout == GEIST_LAYOUT_DENSE) {
            table.stride[0] = (int64_t) d_model;
            table.stride[1] = 1;
        }
        struct geist_tensor out = make_tensor_1d(bo, d_model);
        s = be->desc->vtbl->embedding_lookup_scaled(
            be, &table, token_id, scale, &out);
    }
    if (s == GEIST_OK) {
        s = be->desc->vtbl->buffer_download(d_model * sizeof(float),
                                            (uint8_t *) out_host, bo);
    }

    if (bt != nullptr) {
        be->desc->vtbl->buffer_destroy(be, bt);
    }
    if (bo != nullptr) {
        be->desc->vtbl->buffer_destroy(be, bo);
    }
    geist_backend_destroy(be);
    return s == GEIST_OK ? 0 : -1;
}

static int run_matvec_q6k(const char* backend_name,
                          const float* x_in,
                          const uint8_t* w_in,
                          size_t n_in,
                          size_t n_out,
                          float* out_host) {
    struct geist_backend* be = nullptr;
    if (geist_backend_create(backend_name, nullptr, nullptr, &be) != GEIST_OK) {
        return -1;
    }
    if (be->desc->vtbl->matvec_q6k == nullptr) {
        geist_backend_destroy(be);
        return -1;
    }

    const size_t w_bytes = n_out * (n_in / 256u) * 210u;
    struct geist_buffer *bx = nullptr, *bw = nullptr, *by = nullptr;
    enum geist_status s = be->desc->vtbl->buffer_create(
            be, n_in * sizeof(float), GEIST_BUFFER_ACTIVATION,
            GEIST_MEMORY_DEVICE, &bx);
    if (s == GEIST_OK) {
        s = be->desc->vtbl->buffer_create(
                be, w_bytes, GEIST_BUFFER_WEIGHT, GEIST_MEMORY_DEVICE, &bw);
    }
    if (s == GEIST_OK) {
        s = be->desc->vtbl->buffer_create(
                be, n_out * sizeof(float), GEIST_BUFFER_ACTIVATION,
                GEIST_MEMORY_DEVICE, &by);
    }
    if (s == GEIST_OK) {
        s = be->desc->vtbl->buffer_upload(bx, n_in * sizeof(float),
                                          (const uint8_t*) x_in);
    }
    if (s == GEIST_OK) {
        s = be->desc->vtbl->buffer_upload(bw, w_bytes, w_in);
    }
    if (s == GEIST_OK) {
        struct geist_tensor x = make_tensor_1d(bx, n_in);
        struct geist_tensor w = make_tensor_q6k_2d(bw, n_out, n_in);
        struct geist_tensor y = make_tensor_1d(by, n_out);
        s = be->desc->vtbl->matvec_q6k(be, &x, &w, &y);
    }
    if (s == GEIST_OK) {
        s = be->desc->vtbl->buffer_download(n_out * sizeof(float),
                                            (uint8_t*) out_host, by);
    }

    if (bx != nullptr) {
        be->desc->vtbl->buffer_destroy(be, bx);
    }
    if (bw != nullptr) {
        be->desc->vtbl->buffer_destroy(be, bw);
    }
    if (by != nullptr) {
        be->desc->vtbl->buffer_destroy(be, by);
    }
    geist_backend_destroy(be);
    return s == GEIST_OK ? 0 : -1;
}

static int run_rope(const char* backend_name,
                    const float* x_in,
                    const float* cos_in,
                    const float* sin_in,
                    size_t seq,
                    size_t heads,
                    size_t head_dim,
                    float* out_host) {
    struct geist_backend* be = nullptr;
    if (geist_backend_create(backend_name, nullptr, nullptr, &be) != GEIST_OK) {
        return -1;
    }
    if (be->desc->vtbl->rope_apply == nullptr) {
        geist_backend_destroy(be);
        return -1;
    }

    const size_t x_n = seq * heads * head_dim;
    const size_t rope_n = seq * head_dim;
    struct geist_buffer *bx = nullptr, *bc = nullptr, *bs = nullptr;
    enum geist_status s = be->desc->vtbl->buffer_create(
            be, x_n * sizeof(float), GEIST_BUFFER_ACTIVATION,
            GEIST_MEMORY_DEVICE, &bx);
    if (s == GEIST_OK) {
        s = be->desc->vtbl->buffer_create(
                be, rope_n * sizeof(float), GEIST_BUFFER_ACTIVATION,
                GEIST_MEMORY_DEVICE, &bc);
    }
    if (s == GEIST_OK) {
        s = be->desc->vtbl->buffer_create(
                be, rope_n * sizeof(float), GEIST_BUFFER_ACTIVATION,
                GEIST_MEMORY_DEVICE, &bs);
    }
    if (s == GEIST_OK) {
        s = be->desc->vtbl->buffer_upload(bx, x_n * sizeof(float),
                                          (const uint8_t*) x_in);
    }
    if (s == GEIST_OK) {
        s = be->desc->vtbl->buffer_upload(bc, rope_n * sizeof(float),
                                          (const uint8_t*) cos_in);
    }
    if (s == GEIST_OK) {
        s = be->desc->vtbl->buffer_upload(bs, rope_n * sizeof(float),
                                          (const uint8_t*) sin_in);
    }
    if (s == GEIST_OK) {
        struct geist_tensor x = make_tensor_3d(bx, seq, heads, head_dim);
        struct geist_tensor cos_t = make_tensor_2d(bc, seq, head_dim);
        struct geist_tensor sin_t = make_tensor_2d(bs, seq, head_dim);
        s = be->desc->vtbl->rope_apply(be, &x, &cos_t, &sin_t);
    }
    if (s == GEIST_OK) {
        s = be->desc->vtbl->buffer_download(x_n * sizeof(float),
                                            (uint8_t*) out_host, bx);
    }

    if (bx != nullptr) {
        be->desc->vtbl->buffer_destroy(be, bx);
    }
    if (bc != nullptr) {
        be->desc->vtbl->buffer_destroy(be, bc);
    }
    if (bs != nullptr) {
        be->desc->vtbl->buffer_destroy(be, bs);
    }
    geist_backend_destroy(be);
    return s == GEIST_OK ? 0 : -1;
}

static int run_attention(const char* backend_name,
                         const float* q_in,
                         const float* k_in,
                         const float* v_in,
                         size_t n_q,
                         size_t n_kv,
                         size_t q_offset,
                         size_t n_q_heads,
                         size_t n_kv_heads,
                         size_t head_dim,
                         size_t sliding_window,
                         float* out_host) {
    struct geist_backend* be = nullptr;
    if (geist_backend_create(backend_name, nullptr, nullptr, &be) != GEIST_OK) {
        return -1;
    }
    if (be->desc->vtbl->attention == nullptr) {
        geist_backend_destroy(be);
        return -1;
    }

    const size_t q_n = n_q * n_q_heads * head_dim;
    const size_t kv_n = n_kv * n_kv_heads * head_dim;
    struct geist_buffer *bq = nullptr, *bk = nullptr, *bv = nullptr, *bo = nullptr;
    enum geist_status s = be->desc->vtbl->buffer_create(
            be, q_n * sizeof(float), GEIST_BUFFER_ACTIVATION,
            GEIST_MEMORY_DEVICE, &bq);
    if (s == GEIST_OK) {
        s = be->desc->vtbl->buffer_create(
                be, kv_n * sizeof(float), GEIST_BUFFER_ACTIVATION,
                GEIST_MEMORY_DEVICE, &bk);
    }
    if (s == GEIST_OK) {
        s = be->desc->vtbl->buffer_create(
                be, kv_n * sizeof(float), GEIST_BUFFER_ACTIVATION,
                GEIST_MEMORY_DEVICE, &bv);
    }
    if (s == GEIST_OK) {
        s = be->desc->vtbl->buffer_create(
                be, q_n * sizeof(float), GEIST_BUFFER_ACTIVATION,
                GEIST_MEMORY_DEVICE, &bo);
    }
    if (s == GEIST_OK) {
        s = be->desc->vtbl->buffer_upload(bq, q_n * sizeof(float),
                                          (const uint8_t*) q_in);
    }
    if (s == GEIST_OK) {
        s = be->desc->vtbl->buffer_upload(bk, kv_n * sizeof(float),
                                          (const uint8_t*) k_in);
    }
    if (s == GEIST_OK) {
        s = be->desc->vtbl->buffer_upload(bv, kv_n * sizeof(float),
                                          (const uint8_t*) v_in);
    }
    if (s == GEIST_OK) {
        struct geist_tensor tq = make_tensor_3d(bq, n_q, n_q_heads, head_dim);
        struct geist_tensor tk = make_tensor_3d(bk, n_kv, n_kv_heads, head_dim);
        struct geist_tensor tv = make_tensor_3d(bv, n_kv, n_kv_heads, head_dim);
        struct geist_tensor to = make_tensor_3d(bo, n_q, n_q_heads, head_dim);
        s = be->desc->vtbl->attention(be, &tq, &tk, &tv,
                                      q_offset, sliding_window, &to);
    }
    if (s == GEIST_OK) {
        s = be->desc->vtbl->buffer_download(q_n * sizeof(float),
                                            (uint8_t*) out_host, bo);
    }

    if (bq != nullptr) {
        be->desc->vtbl->buffer_destroy(be, bq);
    }
    if (bk != nullptr) {
        be->desc->vtbl->buffer_destroy(be, bk);
    }
    if (bv != nullptr) {
        be->desc->vtbl->buffer_destroy(be, bv);
    }
    if (bo != nullptr) {
        be->desc->vtbl->buffer_destroy(be, bo);
    }
    geist_backend_destroy(be);
    return s == GEIST_OK ? 0 : -1;
}

static int run_argmax(const char* backend_name,
                      const float* logits_in,
                      size_t n,
                      geist_token_t* out_token) {
    struct geist_backend* be = nullptr;
    if (geist_backend_create(backend_name, nullptr, nullptr, &be) != GEIST_OK) {
        return -1;
    }
    if (be->desc->vtbl->argmax_f32 == nullptr) {
        geist_backend_destroy(be);
        return -1;
    }

    struct geist_buffer *blogits = nullptr;
    enum geist_status s = be->desc->vtbl->buffer_create(
            be, n * sizeof(float), GEIST_BUFFER_ACTIVATION,
            GEIST_MEMORY_DEVICE, &blogits);
    if (s == GEIST_OK) {
        s = be->desc->vtbl->buffer_upload(blogits, n * sizeof(float),
                                          (const uint8_t*) logits_in);
    }
    if (s == GEIST_OK) {
        struct geist_tensor logits = make_tensor_2d(blogits, 1, n);
        s = be->desc->vtbl->argmax_f32(be, &logits, out_token);
    }

    if (blogits != nullptr) {
        be->desc->vtbl->buffer_destroy(be, blogits);
    }
    geist_backend_destroy(be);
    return s == GEIST_OK ? 0 : -1;
}

static int run_argmax_reuse(const char* backend_name,
                            const float* logits_a,
                            const float* logits_b,
                            size_t n,
                            geist_token_t out_tokens[static 2]) {
    struct geist_backend* be = nullptr;
    if (geist_backend_create(backend_name, nullptr, nullptr, &be) != GEIST_OK) {
        return -1;
    }
    if (be->desc->vtbl->argmax_f32 == nullptr) {
        geist_backend_destroy(be);
        return -1;
    }

    struct geist_buffer *ba = nullptr, *bb = nullptr;
    enum geist_status s = be->desc->vtbl->buffer_create(
            be, n * sizeof(float), GEIST_BUFFER_ACTIVATION,
            GEIST_MEMORY_DEVICE, &ba);
    if (s == GEIST_OK) {
        s = be->desc->vtbl->buffer_create(
                be, n * sizeof(float), GEIST_BUFFER_ACTIVATION,
                GEIST_MEMORY_DEVICE, &bb);
    }
    if (s == GEIST_OK) {
        s = be->desc->vtbl->buffer_upload(ba, n * sizeof(float),
                                          (const uint8_t*) logits_a);
    }
    if (s == GEIST_OK) {
        s = be->desc->vtbl->buffer_upload(bb, n * sizeof(float),
                                          (const uint8_t*) logits_b);
    }
    if (s == GEIST_OK) {
        struct geist_tensor ta = make_tensor_2d(ba, 1, n);
        s = be->desc->vtbl->argmax_f32(be, &ta, &out_tokens[0]);
    }
    if (s == GEIST_OK) {
        struct geist_tensor tb = make_tensor_2d(bb, 1, n);
        s = be->desc->vtbl->argmax_f32(be, &tb, &out_tokens[1]);
    }

    if (ba != nullptr) {
        be->desc->vtbl->buffer_destroy(be, ba);
    }
    if (bb != nullptr) {
        be->desc->vtbl->buffer_destroy(be, bb);
    }
    geist_backend_destroy(be);
    return s == GEIST_OK ? 0 : -1;
}

static int run_argmax_batch(const char *backend_name,
                            const float *logits_in,
                            size_t rows,
                            size_t n,
                            geist_token_t out_tokens[static rows]) {
    struct geist_backend *be = nullptr;
    if (geist_backend_create(backend_name, nullptr, nullptr, &be) != GEIST_OK) {
        return -1;
    }
    if (be->desc->vtbl->argmax_f32_batch == nullptr) {
        geist_backend_destroy(be);
        return -1;
    }

    struct geist_buffer *blogits = nullptr;
    enum geist_status s = be->desc->vtbl->buffer_create(
        be, rows * n * sizeof(float), GEIST_BUFFER_ACTIVATION,
        GEIST_MEMORY_DEVICE, &blogits);
    if (s == GEIST_OK) {
        s = be->desc->vtbl->buffer_upload(
            blogits, rows * n * sizeof(float), (const uint8_t *) logits_in);
    }
    if (s == GEIST_OK) {
        struct geist_tensor logits = make_tensor_2d(blogits, rows, n);
        s = be->desc->vtbl->argmax_f32_batch(be, &logits, out_tokens);
    }

    if (blogits != nullptr) {
        be->desc->vtbl->buffer_destroy(be, blogits);
    }
    geist_backend_destroy(be);
    return s == GEIST_OK ? 0 : -1;
}

static void ref_greedy_head_batch(const float *hidden,
                                  const float *norm_weight,
                                  const float *lm_head_weight,
                                  size_t rows,
                                  size_t d_model,
                                  size_t vocab_size,
                                  float eps,
                                  geist_token_t *out_tokens) {
    for (size_t row = 0; row < rows; row++) {
        const float *h = hidden + row * d_model;
        float ss = 0.0f;
        for (size_t i = 0; i < d_model; i++) {
            ss += h[i] * h[i];
        }
        const float scale = 1.0f / sqrtf(ss / (float) d_model + eps);
        float best = -INFINITY;
        geist_token_t best_id = 0;
        for (size_t tok = 0; tok < vocab_size; tok++) {
            const float *wrow = lm_head_weight + tok * d_model;
            float dot = 0.0f;
            for (size_t i = 0; i < d_model; i++) {
                dot += wrow[i] * h[i] * norm_weight[i] * scale;
            }
            if (tok == 0 || dot > best) {
                best = dot;
                best_id = (geist_token_t) tok;
            }
        }
        out_tokens[row] = best_id;
    }
}

static int run_greedy_head_batch(const char *backend_name,
                                 const float *hidden_in,
                                 const float *norm_in,
                                 const void *weight_in,
                                 size_t weight_bytes,
                                 enum geist_dtype weight_dtype,
                                 enum geist_layout weight_layout,
                                 size_t rows,
                                 size_t d_model,
                                 size_t vocab_size,
                                 float eps,
                                 geist_token_t *out_tokens) {
    struct geist_backend *be = nullptr;
    if (geist_backend_create(backend_name, nullptr, nullptr, &be) != GEIST_OK) {
        return -1;
    }
    if (be->desc->vtbl->greedy_head_batch == nullptr) {
        geist_backend_destroy(be);
        return -1;
    }

    struct geist_buffer *hidden_buf = nullptr;
    struct geist_buffer *norm_buf = nullptr;
    struct geist_buffer *weight_buf = nullptr;
    struct geist_buffer *normed_buf = nullptr;
    struct geist_buffer *logits_buf = nullptr;
    enum geist_status s = be->desc->vtbl->buffer_create(
        be, rows * d_model * sizeof(float), GEIST_BUFFER_ACTIVATION,
        GEIST_MEMORY_DEVICE, &hidden_buf);
    if (s == GEIST_OK) {
        s = be->desc->vtbl->buffer_create(
            be, d_model * sizeof(float), GEIST_BUFFER_WEIGHT,
            GEIST_MEMORY_DEVICE, &norm_buf);
    }
    if (s == GEIST_OK) {
        s = be->desc->vtbl->buffer_create(
            be, weight_bytes, GEIST_BUFFER_WEIGHT,
            GEIST_MEMORY_DEVICE, &weight_buf);
    }
    if (s == GEIST_OK) {
        s = be->desc->vtbl->buffer_create(
            be, rows * d_model * sizeof(float), GEIST_BUFFER_SCRATCH,
            GEIST_MEMORY_DEVICE, &normed_buf);
    }
    if (s == GEIST_OK) {
        s = be->desc->vtbl->buffer_create(
            be, rows * vocab_size * sizeof(float), GEIST_BUFFER_SCRATCH,
            GEIST_MEMORY_DEVICE, &logits_buf);
    }
    if (s == GEIST_OK) {
        s = be->desc->vtbl->buffer_upload(
            hidden_buf, rows * d_model * sizeof(float),
            (const uint8_t *) hidden_in);
    }
    if (s == GEIST_OK) {
        s = be->desc->vtbl->buffer_upload(
            norm_buf, d_model * sizeof(float), (const uint8_t *) norm_in);
    }
    if (s == GEIST_OK) {
        s = be->desc->vtbl->buffer_upload(
            weight_buf, weight_bytes,
            (const uint8_t *) weight_in);
    }
    if (s == GEIST_OK) {
        struct geist_tensor hidden = make_tensor_2d(hidden_buf, rows, d_model);
        struct geist_tensor norm = make_tensor_1d(norm_buf, d_model);
        struct geist_tensor weight = {
            .buffer = weight_buf,
            .offset = 0,
            .dtype = weight_dtype,
            .layout = weight_layout,
            .ndim = 2,
            .shape = {(int64_t) vocab_size, (int64_t) d_model},
        };
        if (weight_layout == GEIST_LAYOUT_DENSE) {
            weight.stride[0] = (int64_t) d_model;
            weight.stride[1] = 1;
        }
        struct geist_tensor normed = make_tensor_2d(normed_buf, rows, d_model);
        struct geist_tensor logits =
            make_tensor_2d(logits_buf, rows, vocab_size);
        const struct geist_backend_greedy_head_batch head = {
            .struct_size = sizeof(head),
            .d_model = d_model,
            .vocab_size = vocab_size,
            .row_count = rows,
            .token_output_offset = 0,
            .eps = eps,
            .hidden = &hidden,
            .norm_weight = &norm,
            .lm_head_weight = &weight,
            .normed_scratch = &normed,
            .logits = &logits,
        };
        s = be->desc->vtbl->greedy_head_batch(be, &head, out_tokens);
    }

    if (hidden_buf != nullptr) be->desc->vtbl->buffer_destroy(be, hidden_buf);
    if (norm_buf != nullptr) be->desc->vtbl->buffer_destroy(be, norm_buf);
    if (weight_buf != nullptr) be->desc->vtbl->buffer_destroy(be, weight_buf);
    if (normed_buf != nullptr) be->desc->vtbl->buffer_destroy(be, normed_buf);
    if (logits_buf != nullptr) be->desc->vtbl->buffer_destroy(be, logits_buf);
    geist_backend_destroy(be);
    return s == GEIST_OK ? 0 : -1;
}

int main(void) {
    int fails = 0;
    const size_t N = 256;
    const bool have_neon = backend_available("cpu_neon");
    const bool have_vulkan = backend_available("vulkan");

    /* Deterministic input. */
    float a[N], b[N], x[N], w[N];
    for (size_t i = 0; i < N; i++) {
        a[i] = (float) i * 0.01f - 1.2f;
        b[i] = (float) (i % 7) * 0.05f - 0.15f;
        x[i] = sinf((float) i * 0.31415f) * 2.0f;
        w[i] = 1.0f + (float) (i % 11) * 0.07f;
    }

    /* ---- add: scalar ≡ neon, bit-identical ---- */
    float ya_scalar[N], ya_neon[N];
    fails += check(run_add("cpu_scalar", a, b, N, ya_scalar) == 0, "scalar add ran");
    if (have_neon) {
        fails += check(run_add("cpu_neon", a, b, N, ya_neon) == 0, "neon add ran");
        fails += compare_op_outputs("add", ya_scalar, ya_neon, N, 0.0f, 0.0f);
    }
    if (have_vulkan) {
        float ya_vulkan[N];
        fails += check(run_add("vulkan", a, b, N, ya_vulkan) == 0,
                       "vulkan add ran");
        fails += compare_op_outputs("add", ya_scalar, ya_vulkan, N,
                                    0.0f, 0.0f);
    }

    /* Spot-check correctness vs hand-computed. */
    for (size_t i = 0; i < 5; i++) {
        float expected = a[i] + b[i];
        fails += check(fabsf(ya_scalar[i] - expected) < 1e-6f, "add correctness");
    }

    /* ---- mul: decompose fallback primitive for PLE/FFN paths ---- */
    float ym_scalar[N];
    fails += check(run_mul("cpu_scalar", a, b, N, GEIST_MEMORY_AUTO,
                           false, ym_scalar) == 0,
                   "scalar mul ran");
    for (size_t i = 0; i < N; i++) {
        const float expected = a[i] * b[i];
        fails += check(fabsf(ym_scalar[i] - expected) < 1e-6f,
                       "mul scalar correctness");
    }
    if (have_neon) {
        float ym_neon[N];
        fails += check(run_mul("cpu_neon", a, b, N, GEIST_MEMORY_AUTO,
                               false, ym_neon) == 0,
                       "neon mul ran");
        fails += compare_op_outputs("mul", ym_scalar, ym_neon,
                                    N, 0.0f, 0.0f);
    }
    if (have_vulkan) {
        float ym_vulkan[N];
        fails += check(run_mul("vulkan", a, b, N, GEIST_MEMORY_DEVICE,
                               true, ym_vulkan) == 0,
                       "vulkan mul ran on device buffers");
        fails += compare_op_outputs("mul", ym_scalar, ym_vulkan,
                                    N, 0.0f, 0.0f);
    }

    /* ---- scale_f32: layer output scaling without host mapping ---- */
    float ys_scalar[N];
    fails += check(run_scale("cpu_scalar", x, N, -0.375f,
                             GEIST_MEMORY_AUTO, false, ys_scalar) == 0,
                   "scalar scale_f32 ran");
    for (size_t i = 0; i < N; i++) {
        const float expected = x[i] * -0.375f;
        fails += check(fabsf(ys_scalar[i] - expected) < 1e-6f,
                       "scale_f32 scalar correctness");
    }
    if (have_neon) {
        float ys_neon[N];
        fails += check(run_scale("cpu_neon", x, N, -0.375f,
                                 GEIST_MEMORY_AUTO, false, ys_neon) == 0,
                       "neon scale_f32 ran");
        fails += compare_op_outputs("scale_f32", ys_scalar, ys_neon,
                                    N, 0.0f, 0.0f);
    }
    if (have_vulkan) {
        float ys_vulkan[N];
        fails += check(run_scale("vulkan", x, N, -0.375f,
                                 GEIST_MEMORY_DEVICE, true, ys_vulkan) == 0,
                       "vulkan scale_f32 ran on device buffers");
        fails += compare_op_outputs("scale_f32", ys_scalar, ys_vulkan,
                                    N, 0.0f, 0.0f);
    }

    /* ---- gelu_tanh: tight tolerance (same scalar kernel both sides) ---- */
    float yg_scalar[N], yg_neon[N];
    fails += check(run_gelu("cpu_scalar", x, N, GEIST_MEMORY_AUTO,
                            false, yg_scalar) == 0,
                   "scalar gelu ran");
    if (have_neon) {
        fails += check(run_gelu("cpu_neon", x, N, GEIST_MEMORY_AUTO,
                                false, yg_neon) == 0,
                       "neon gelu ran");
        fails += compare_op_outputs("gelu_tanh", yg_scalar, yg_neon, N, 1e-5f, 1e-6f);
    }
    if (have_vulkan) {
        float yg_vulkan[N];
        fails += check(run_gelu("vulkan", x, N, GEIST_MEMORY_DEVICE,
                                true, yg_vulkan) == 0,
                       "vulkan gelu_tanh ran on device buffers");
        fails += compare_op_outputs("gelu_tanh", yg_scalar, yg_vulkan,
                                    N, 2e-5f, 2e-6f);
    }

    /* gelu(0) = 0; gelu(very-positive) ≈ x; gelu(very-negative) ≈ 0. */
    fails += check(fabsf(yg_scalar[0]) < 1e-5f || x[0] != 0.0f, "gelu(0) sanity");

    /* ---- gelu_tanh_mul: Gemma GEGLU FFN middle op ---- */
    float ygm_scalar[N];
    fails += check(run_gelu_mul("cpu_scalar", x, b, N, GEIST_MEMORY_AUTO,
                                ygm_scalar) == 0,
                   "scalar gelu_tanh_mul ran");
    for (size_t i = 0; i < N; i++) {
        const float expected = yg_scalar[i] * b[i];
        fails += check(fabsf(ygm_scalar[i] - expected) < 2e-6f,
                       "gelu_tanh_mul scalar correctness");
    }
    if (have_vulkan) {
        float ygm_vulkan[N];
        fails += check(run_gelu_mul("vulkan", x, b, N, GEIST_MEMORY_DEVICE,
                                    ygm_vulkan) == 0,
                       "vulkan gelu_tanh_mul ran on device buffers");
        fails += compare_op_outputs("gelu_tanh_mul", ygm_scalar, ygm_vulkan,
                                    N, 1e-5f, 2e-6f);
    }

    /* ---- rmsnorm: scalar vs neon, tolerate reduction order ---- */
    float yr_scalar[N * 4], yr_neon[N * 4];
    /* 4 rows × 64 feat = 256 elements; we want n_rows=4, feat=64. */
    const size_t n_rows = 4, feat = 64;
    /* Reuse x as 4×64 input. */
    float rms_x[256], rms_w[64];
    for (size_t i = 0; i < 256; i++)
        rms_x[i] = x[i];
    for (size_t i = 0; i < 64; i++)
        rms_w[i] = w[i];
    fails += check(run_rmsnorm("cpu_scalar", rms_x, rms_w, n_rows, feat, 1e-6f, yr_scalar) == 0,
                   "scalar rmsnorm ran");
    if (have_neon) {
        fails += check(run_rmsnorm("cpu_neon", rms_x, rms_w, n_rows, feat, 1e-6f, yr_neon) == 0,
                       "neon rmsnorm ran");
        fails += compare_op_outputs("rmsnorm", yr_scalar, yr_neon, n_rows * feat, 1e-4f, 1e-5f);
    }
    if (have_vulkan) {
        float yr_vulkan[N * 4];
        fails += check(run_rmsnorm("vulkan", rms_x, rms_w, n_rows, feat, 1e-6f, yr_vulkan) == 0,
                       "vulkan rmsnorm ran");
        fails += compare_op_outputs("rmsnorm", yr_scalar, yr_vulkan, n_rows * feat,
                                    1e-4f, 2e-5f);

        float reuse_scalar[N];
        float reuse_vulkan[N];
        fails += check(run_elementwise_reuse_chain(
                           "cpu_scalar", a, b, rms_w, n_rows, feat,
                           GEIST_MEMORY_AUTO, false, reuse_scalar) == 0,
                       "scalar elementwise reuse chain ran");
        fails += check(run_elementwise_reuse_chain(
                           "vulkan", a, b, rms_w, n_rows, feat,
                           GEIST_MEMORY_DEVICE, true, reuse_vulkan) == 0,
                       "vulkan elementwise reuse chain ran on device buffers");
        fails += compare_op_outputs("elementwise reuse chain",
                                    reuse_scalar, reuse_vulkan, N,
                                    1e-4f, 2e-5f);
    }

    /* ---- matvec: narrow Vulkan device-resident F32 decode primitive ---- */
    if (have_vulkan) {
        constexpr size_t mv_in = 96;
        constexpr size_t mv_out = 37;
        float mv_x[mv_in], mv_w[mv_out * mv_in];
        float mv_ref[mv_out], mv_vulkan[mv_out];
        for (size_t i = 0; i < mv_in; i++) {
            mv_x[i] = sinf((float) i * 0.17f) * 0.75f;
        }
        for (size_t i = 0; i < mv_out * mv_in; i++) {
            mv_w[i] = cosf((float) i * 0.031f) * 0.25f;
        }
        ref_matvec_f32(mv_x, mv_w, mv_in, mv_out, mv_ref);
        fails += check(run_matvec("vulkan", mv_x, mv_w, mv_in, mv_out, mv_vulkan) == 0,
                       "vulkan matvec ran");
        fails += compare_op_outputs("matvec", mv_ref, mv_vulkan, mv_out,
                                    1e-4f, 2e-5f);

        constexpr size_t gh_rows = 2;
        constexpr size_t gh_d = 32;
        constexpr size_t gh_vocab = 19;
        constexpr float gh_eps = 1e-6f;
        float gh_hidden[gh_rows * gh_d];
        float gh_norm[gh_d];
        float gh_weight[gh_vocab * gh_d];
        geist_token_t gh_ref[gh_rows] = {-1, -1};
        geist_token_t gh_vulkan[gh_rows] = {-1, -1};
        for (size_t i = 0; i < gh_rows * gh_d; i++) {
            gh_hidden[i] = sinf((float) i * 0.19f) * 0.7f +
                           cosf((float) i * 0.11f) * 0.2f;
        }
        for (size_t i = 0; i < gh_d; i++) {
            gh_norm[i] = 0.75f + 0.01f * (float) (i % 7);
        }
        for (size_t i = 0; i < gh_vocab * gh_d; i++) {
            gh_weight[i] = sinf((float) i * 0.053f) * 0.3f -
                           cosf((float) i * 0.017f) * 0.1f;
        }
        ref_greedy_head_batch(gh_hidden, gh_norm, gh_weight,
                              gh_rows, gh_d, gh_vocab, gh_eps, gh_ref);
        fails += check(run_greedy_head_batch(
                           "vulkan", gh_hidden, gh_norm, gh_weight,
                           sizeof(gh_weight), GEIST_DTYPE_F32,
                           GEIST_LAYOUT_DENSE,
                           gh_rows, gh_d, gh_vocab, gh_eps, gh_vulkan) == 0,
                       "vulkan greedy_head_batch ran on device buffers");
        fails += check(gh_vulkan[0] == gh_ref[0] &&
                       gh_vulkan[1] == gh_ref[1],
                       "vulkan greedy_head_batch argmax matched");

        constexpr size_t gh_q4_rows = 2;
        constexpr size_t gh_q4_d = 512;
        constexpr size_t gh_q4_vocab = 11;
        constexpr size_t gh_q4_bytes =
            gh_q4_vocab * (gh_q4_d / 256u) * 144u;
        float gh_q4_hidden[gh_q4_rows * gh_q4_d];
        float gh_q4_norm[gh_q4_d];
        float gh_q4_normed[gh_q4_d];
        float gh_q4_logits[gh_q4_vocab];
        uint8_t gh_q4_weight[gh_q4_bytes];
        geist_token_t gh_q4_ref[gh_q4_rows] = {-1, -1};
        geist_token_t gh_q4_vulkan[gh_q4_rows] = {-1, -1};
        for (size_t i = 0; i < gh_q4_rows * gh_q4_d; i++) {
            gh_q4_hidden[i] = sinf((float) i * 0.029f) * 0.55f -
                              cosf((float) i * 0.041f) * 0.18f;
        }
        for (size_t i = 0; i < gh_q4_d; i++) {
            gh_q4_norm[i] = 0.65f + 0.012f * (float) (i % 13);
        }
        pack_q4k_test_matrix(gh_q4_d, gh_q4_vocab, gh_q4_weight);
        for (size_t row = 0; row < gh_q4_rows; row++) {
            const float *h = gh_q4_hidden + row * gh_q4_d;
            float ss = 0.0f;
            for (size_t i = 0; i < gh_q4_d; i++) {
                ss += h[i] * h[i];
            }
            const float scale = 1.0f / sqrtf(ss / (float) gh_q4_d + gh_eps);
            for (size_t i = 0; i < gh_q4_d; i++) {
                gh_q4_normed[i] = h[i] * gh_q4_norm[i] * scale;
            }
            ref_matvec_q4k_test(gh_q4_normed, gh_q4_d, gh_q4_vocab,
                                gh_q4_logits);
            float best = gh_q4_logits[0];
            geist_token_t best_id = 0;
            for (size_t tok = 1; tok < gh_q4_vocab; tok++) {
                if (gh_q4_logits[tok] > best) {
                    best = gh_q4_logits[tok];
                    best_id = (geist_token_t) tok;
                }
            }
            gh_q4_ref[row] = best_id;
        }
        fails += check(run_greedy_head_batch(
                           "vulkan", gh_q4_hidden, gh_q4_norm,
                           gh_q4_weight, sizeof(gh_q4_weight),
                           GEIST_DTYPE_Q4_K, GEIST_LAYOUT_BLOCK_QUANTIZED,
                           gh_q4_rows, gh_q4_d, gh_q4_vocab, gh_eps,
                           gh_q4_vulkan) == 0,
                       "vulkan Q4_K greedy_head_batch ran on device buffers");
        fails += check(gh_q4_vulkan[0] == gh_q4_ref[0] &&
                       gh_q4_vulkan[1] == gh_q4_ref[1],
                       "vulkan Q4_K greedy_head_batch argmax matched");

        constexpr size_t gh_q6_rows = 2;
        constexpr size_t gh_q6_d = 512;
        constexpr size_t gh_q6_vocab = 9;
        constexpr size_t gh_q6_bytes =
            gh_q6_vocab * (gh_q6_d / 256u) * 210u;
        float gh_q6_hidden[gh_q6_rows * gh_q6_d];
        float gh_q6_norm[gh_q6_d];
        float gh_q6_normed[gh_q6_d];
        float gh_q6_logits[gh_q6_vocab];
        uint8_t gh_q6_weight[gh_q6_bytes];
        geist_token_t gh_q6_ref[gh_q6_rows] = {-1, -1};
        geist_token_t gh_q6_vulkan[gh_q6_rows] = {-1, -1};
        for (size_t i = 0; i < gh_q6_rows * gh_q6_d; i++) {
            gh_q6_hidden[i] = sinf((float) i * 0.021f) * 0.6f +
                              cosf((float) i * 0.037f) * 0.2f;
        }
        for (size_t i = 0; i < gh_q6_d; i++) {
            gh_q6_norm[i] = 0.5f + 0.015f * (float) (i % 11);
        }
        pack_q6k_test_matrix(gh_q6_d, gh_q6_vocab, gh_q6_weight);
        for (size_t row = 0; row < gh_q6_rows; row++) {
            const float *h = gh_q6_hidden + row * gh_q6_d;
            float ss = 0.0f;
            for (size_t i = 0; i < gh_q6_d; i++) {
                ss += h[i] * h[i];
            }
            const float scale = 1.0f / sqrtf(ss / (float) gh_q6_d + gh_eps);
            for (size_t i = 0; i < gh_q6_d; i++) {
                gh_q6_normed[i] = h[i] * gh_q6_norm[i] * scale;
            }
            ref_matvec_q6k_test(gh_q6_normed, gh_q6_d, gh_q6_vocab,
                                gh_q6_logits);
            float best = gh_q6_logits[0];
            geist_token_t best_id = 0;
            for (size_t tok = 1; tok < gh_q6_vocab; tok++) {
                if (gh_q6_logits[tok] > best) {
                    best = gh_q6_logits[tok];
                    best_id = (geist_token_t) tok;
                }
            }
            gh_q6_ref[row] = best_id;
        }
        fails += check(run_greedy_head_batch(
                           "vulkan", gh_q6_hidden, gh_q6_norm,
                           gh_q6_weight, sizeof(gh_q6_weight),
                           GEIST_DTYPE_Q6_K, GEIST_LAYOUT_BLOCK_QUANTIZED,
                           gh_q6_rows, gh_q6_d, gh_q6_vocab, gh_eps,
                           gh_q6_vulkan) == 0,
                       "vulkan Q6_K greedy_head_batch ran on device buffers");
        fails += check(gh_q6_vulkan[0] == gh_q6_ref[0] &&
                       gh_q6_vulkan[1] == gh_q6_ref[1],
                       "vulkan Q6_K greedy_head_batch argmax matched");

        constexpr size_t emb_vocab = 7;
        constexpr size_t emb_dim = 13;
        constexpr geist_token_t emb_token = 4;
        float emb_table[emb_vocab * emb_dim];
        float emb_ref[emb_dim], emb_vulkan[emb_dim];
        for (size_t i = 0; i < emb_vocab * emb_dim; i++) {
            emb_table[i] = sinf((float) i * 0.13f) * 0.5f +
                           cosf((float) i * 0.07f) * 0.25f;
        }
        memcpy(emb_ref, emb_table + (size_t) emb_token * emb_dim,
               sizeof(emb_ref));
        fails += check(run_embedding_lookup("vulkan", emb_table,
                                            emb_vocab, emb_dim, emb_token,
                                            true, emb_vulkan) == 0,
                       "vulkan embedding lookup copied a device-only row");
        fails += compare_op_outputs("embedding_lookup", emb_ref, emb_vulkan,
                                    emb_dim, 0.0f, 0.0f);

        constexpr size_t emb_q_in = 512;
        constexpr size_t emb_q_vocab = 5;
        constexpr geist_token_t emb_q_token = 3;
        constexpr float emb_q_scale = 0.125f;
        constexpr size_t emb_q4_bytes =
            emb_q_vocab * (emb_q_in / 256u) * 144u;
        constexpr size_t emb_q6_bytes =
            emb_q_vocab * (emb_q_in / 256u) * 210u;
        uint8_t emb_q4[emb_q4_bytes];
        uint8_t emb_q6[emb_q6_bytes];
        float emb_q_ref[emb_q_in], emb_q_vulkan[emb_q_in];

        pack_q4k_test_matrix(emb_q_in, emb_q_vocab, emb_q4);
        ref_embedding_q4k_test(emb_q_in, (size_t) emb_q_token,
                               1.0f, emb_q_ref);
        fails += check(run_embedding_lookup_typed(
                           "vulkan", emb_q4, sizeof(emb_q4),
                           GEIST_DTYPE_Q4_K,
                           GEIST_LAYOUT_BLOCK_QUANTIZED,
                           emb_q_vocab, emb_q_in, emb_q_token,
                           true, emb_q_vulkan) == 0,
                       "vulkan Q4_K embedding lookup ran on device buffers");
        fails += compare_op_outputs("embedding_lookup_q4k",
                                    emb_q_ref, emb_q_vulkan,
                                    emb_q_in, 0.0f, 0.0f);
        ref_embedding_q4k_test(emb_q_in, (size_t) emb_q_token,
                               emb_q_scale, emb_q_ref);
        fails += check(run_embedding_lookup_scaled(
                           "vulkan", emb_q4, sizeof(emb_q4),
                           GEIST_DTYPE_Q4_K,
                           GEIST_LAYOUT_BLOCK_QUANTIZED,
                           emb_q_vocab, emb_q_in, emb_q_token,
                           emb_q_scale, true, emb_q_vulkan) == 0,
                       "vulkan Q4_K scaled embedding lookup ran on device buffers");
        fails += compare_op_outputs("embedding_lookup_scaled_q4k",
                                    emb_q_ref, emb_q_vulkan,
                                    emb_q_in, 0.0f, 0.0f);

        pack_q6k_test_matrix(emb_q_in, emb_q_vocab, emb_q6);
        ref_embedding_q6k_test(emb_q_in, (size_t) emb_q_token,
                               1.0f, emb_q_ref);
        fails += check(run_embedding_lookup_typed(
                           "vulkan", emb_q6, sizeof(emb_q6),
                           GEIST_DTYPE_Q6_K,
                           GEIST_LAYOUT_BLOCK_QUANTIZED,
                           emb_q_vocab, emb_q_in, emb_q_token,
                           true, emb_q_vulkan) == 0,
                       "vulkan Q6_K embedding lookup ran on device buffers");
        fails += compare_op_outputs("embedding_lookup_q6k",
                                    emb_q_ref, emb_q_vulkan,
                                    emb_q_in, 0.0f, 0.0f);
        ref_embedding_q6k_test(emb_q_in, (size_t) emb_q_token,
                               emb_q_scale, emb_q_ref);
        fails += check(run_embedding_lookup_scaled(
                           "vulkan", emb_q6, sizeof(emb_q6),
                           GEIST_DTYPE_Q6_K,
                           GEIST_LAYOUT_BLOCK_QUANTIZED,
                           emb_q_vocab, emb_q_in, emb_q_token,
                           emb_q_scale, true, emb_q_vulkan) == 0,
                       "vulkan Q6_K scaled embedding lookup ran on device buffers");
        fails += compare_op_outputs("embedding_lookup_scaled_q6k",
                                    emb_q_ref, emb_q_vulkan,
                                    emb_q_in, 0.0f, 0.0f);

        constexpr size_t q4_in = 512;
        constexpr size_t q4_out = 9;
        constexpr size_t q4_bytes = q4_out * (q4_in / 256u) * 144u;
        float q4_x[q4_in], q4_ref[q4_out], q4_vulkan[q4_out];
        uint8_t q4_w[q4_bytes];
        for (size_t i = 0; i < q4_in; i++) {
            q4_x[i] = sinf((float) i * 0.091f) * 0.5f +
                      cosf((float) i * 0.017f) * 0.125f;
        }
        pack_q4k_test_matrix(q4_in, q4_out, q4_w);
        ref_matvec_q4k_test(q4_x, q4_in, q4_out, q4_ref);
        fails += check(run_matvec_q4k("vulkan", q4_x, q4_w, q4_in, q4_out,
                                      q4_vulkan) == 0,
                       "vulkan Q4_K matvec ran on device buffers");
        fails += compare_op_outputs("matvec_q4k", q4_ref, q4_vulkan,
                                    q4_out, 2e-4f, 4e-5f);

        constexpr size_t q6_in = 512;
        constexpr size_t q6_out = 9;
        constexpr size_t q6_bytes = q6_out * (q6_in / 256u) * 210u;
        float q6_x[q6_in], q6_ref[q6_out], q6_vulkan[q6_out];
        uint8_t q6_w[q6_bytes];
        for (size_t i = 0; i < q6_in; i++) {
            q6_x[i] = cosf((float) i * 0.067f) * 0.375f -
                      sinf((float) i * 0.019f) * 0.25f;
        }
        pack_q6k_test_matrix(q6_in, q6_out, q6_w);
        ref_matvec_q6k_test(q6_x, q6_in, q6_out, q6_ref);
        fails += check(run_matvec_q6k("vulkan", q6_x, q6_w, q6_in, q6_out,
                                      q6_vulkan) == 0,
                       "vulkan Q6_K matvec ran on device buffers");
        fails += compare_op_outputs("matvec_q6k", q6_ref, q6_vulkan,
                                    q6_out, 2e-4f, 4e-5f);

        constexpr size_t rope_seq = 3;
        constexpr size_t rope_heads = 2;
        constexpr size_t rope_dim = 8;
        float rope_x[rope_seq * rope_heads * rope_dim];
        float rope_cos[rope_seq * rope_dim];
        float rope_sin[rope_seq * rope_dim];
        float rope_ref[rope_seq * rope_heads * rope_dim];
        float rope_vulkan[rope_seq * rope_heads * rope_dim];
        for (size_t i = 0; i < rope_seq * rope_heads * rope_dim; i++) {
            rope_x[i] = sinf((float) i * 0.13f) * 0.5f +
                        cosf((float) i * 0.07f) * 0.25f;
            rope_ref[i] = rope_x[i];
        }
        for (size_t s = 0; s < rope_seq; s++) {
            for (size_t i = 0; i < rope_dim / 2u; i++) {
                const float angle = (float) (s + 1u) * (float) (i + 1u) * 0.11f;
                const float c = cosf(angle);
                const float si = sinf(angle);
                rope_cos[s * rope_dim + i] = c;
                rope_cos[s * rope_dim + rope_dim / 2u + i] = c;
                rope_sin[s * rope_dim + i] = si;
                rope_sin[s * rope_dim + rope_dim / 2u + i] = si;
            }
        }
        ref_rope_f32(rope_ref, rope_cos, rope_sin,
                     rope_seq, rope_heads, rope_dim);
        fails += check(run_rope("vulkan", rope_x, rope_cos, rope_sin,
                                rope_seq, rope_heads, rope_dim,
                                rope_vulkan) == 0,
                       "vulkan rope ran");
        fails += compare_op_outputs("rope", rope_ref, rope_vulkan,
                                    rope_seq * rope_heads * rope_dim,
                                    1e-5f, 1e-6f);

        constexpr size_t attn_q = 3;
        constexpr size_t attn_kv = 5;
        constexpr size_t attn_q_heads = 4;
        constexpr size_t attn_kv_heads = 2;
        constexpr size_t attn_dim = 8;
        constexpr size_t attn_q_offset = 1;
        constexpr size_t attn_window = 3;
        float attn_qv[attn_q * attn_q_heads * attn_dim];
        float attn_kv_k[attn_kv * attn_kv_heads * attn_dim];
        float attn_kv_v[attn_kv * attn_kv_heads * attn_dim];
        float attn_ref[attn_q * attn_q_heads * attn_dim];
        float attn_vulkan[attn_q * attn_q_heads * attn_dim];
        for (size_t i = 0; i < attn_q * attn_q_heads * attn_dim; i++) {
            attn_qv[i] = sinf((float) i * 0.09f) * 0.20f;
        }
        for (size_t i = 0; i < attn_kv * attn_kv_heads * attn_dim; i++) {
            attn_kv_k[i] = cosf((float) i * 0.05f) * 0.18f;
            attn_kv_v[i] = sinf((float) i * 0.07f) * 0.31f +
                           cosf((float) i * 0.03f) * 0.11f;
        }
        ref_attention_f32(attn_qv, attn_kv_k, attn_kv_v,
                          attn_q, attn_kv, attn_q_offset,
                          attn_q_heads, attn_kv_heads, attn_dim,
                          attn_window, attn_ref);
        fails += check(run_attention("vulkan", attn_qv, attn_kv_k, attn_kv_v,
                                     attn_q, attn_kv, attn_q_offset,
                                     attn_q_heads, attn_kv_heads, attn_dim,
                                     attn_window, attn_vulkan) == 0,
                       "vulkan attention ran");
        fails += compare_op_outputs("attention", attn_ref, attn_vulkan,
                                    attn_q * attn_q_heads * attn_dim,
                                    1e-5f, 2e-5f);

        const float logits[] = {-2.0f, 0.5f, 7.0f, 1.0f, 7.0f, -1.0f};
        geist_token_t token = -1;
        fails += check(run_argmax("vulkan", logits,
                                  sizeof(logits) / sizeof(logits[0]),
                                  &token) == 0,
                       "vulkan argmax ran");
        fails += check(token == 2,
                       "vulkan argmax returns the lowest max-logit index");

        const float logits2[] = {4.0f, 9.0f, 3.0f, 2.0f, 1.0f, 0.0f};
        geist_token_t tokens[2] = {-1, -1};
        fails += check(run_argmax_reuse("vulkan", logits, logits2,
                                        sizeof(logits) / sizeof(logits[0]),
                                        tokens) == 0,
                       "vulkan argmax reuses backend resources");
        fails += check(tokens[0] == 2 && tokens[1] == 1,
                       "vulkan argmax reuse returns per-call maxima");

        constexpr size_t argmax_rows = 3;
        constexpr size_t argmax_n = 6;
        const float logits_batch[argmax_rows * argmax_n] = {
            1.0f,  2.0f,  7.0f,  7.0f, -1.0f,  0.0f,
            4.0f,  9.0f,  3.0f,  2.0f,  1.0f,  0.0f,
           -2.0f, -1.0f, -4.0f, -0.5f, -0.5f, -3.0f,
        };
        geist_token_t cpu_batch[argmax_rows] = {-1, -1, -1};
        geist_token_t vk_batch[argmax_rows] = {-1, -1, -1};
        fails += check(run_argmax_batch("cpu_scalar", logits_batch,
                                        argmax_rows, argmax_n,
                                        cpu_batch) == 0,
                       "cpu_scalar argmax_batch ran");
        fails += check(cpu_batch[0] == 2 &&
                       cpu_batch[1] == 1 &&
                       cpu_batch[2] == 3,
                       "cpu_scalar argmax_batch returns per-row maxima");
        fails += check(run_argmax_batch("vulkan", logits_batch,
                                        argmax_rows, argmax_n,
                                        vk_batch) == 0,
                       "vulkan argmax_batch ran");
        fails += check(vk_batch[0] == cpu_batch[0] &&
                       vk_batch[1] == cpu_batch[1] &&
                       vk_batch[2] == cpu_batch[2],
                       "vulkan argmax_batch matches cpu_scalar");
    }

    if (fails == 0) {
        if (have_neon && have_vulkan) {
            printf("PASS: add / mul / scale / gelu_tanh / gelu_tanh_mul / rmsnorm — "
                   "cpu_scalar ≡ cpu_neon; "
                   "vulkan add/mul/scale/gelu_tanh/gelu_tanh_mul/rmsnorm/matvec/rope/attention/argmax/argmax_batch matched\n");
        } else if (have_neon) {
            printf("PASS: add / mul / scale / gelu_tanh / gelu_tanh_mul / rmsnorm — "
                   "cpu_scalar ≡ cpu_neon\n");
        } else if (have_vulkan) {
            printf("PASS: add / mul / scale / gelu_tanh / gelu_tanh_mul / rmsnorm — "
                   "cpu_scalar; "
                   "vulkan add/mul/scale/gelu_tanh/gelu_tanh_mul/rmsnorm/matvec/rope/attention/argmax/argmax_batch matched\n");
        } else {
            printf("PASS: add / mul / scale / gelu_tanh / gelu_tanh_mul / rmsnorm — cpu_scalar "
                   "(cpu_neon not compiled in)\n");
        }
        return GEIST_TEST_PASS;
    }
    fprintf(stderr, "FAILED: %d check(s)\n", fails);
    return GEIST_TEST_FAIL;
}
