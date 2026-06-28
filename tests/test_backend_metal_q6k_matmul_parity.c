/*
 * test_backend_metal_q6k_matmul_parity — numerical parity gate for the Metal
 * Q6_K matmul (rows>1): the reduction kernels and the simdgroup-GEMM variant
 * (matmul_q6k_sg) prefill uses for the q6k down-projection.
 *
 * Deterministic q6k weight (scale=1) + fp16-exact f32 activation, run
 * vtbl->matmul_q6k, read back, compare to per-row CPU reference.
 * Helpers mirror tests/test_backend_ops_unit.c.
 */
#include <geist.h>
#include <geist_backend.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fail = 0;
static int check(bool cond, const char *what) {
    if (!cond) { fprintf(stderr, "FAIL: %s\n", what); g_fail++; }
    return cond ? 0 : 1;
}

static int8_t q6k_test_value(size_t row, size_t block, size_t half_idx,
                             size_t stream, size_t idx) {
    const size_t v = row * 5u + block * 7u + half_idx * 11u + stream * 3u + idx;
    return (int8_t) ((int) (v & 15u) - 8);
}

static void pack_q6k_test_matrix(size_t n_in, size_t n_out, uint8_t *dst) {
    const size_t bpr = n_in / 256u;
    for (size_t row = 0; row < n_out; row++) {
        for (size_t block = 0; block < bpr; block++) {
            uint8_t *b = dst + (row * bpr + block) * 210u;
            memset(b, 0, 210u);
            for (size_t i = 0; i < 16u; i++) b[192u + i] = 1u;
            b[208] = 0x00u; b[209] = 0x3cu; /* fp16(1.0) */
            for (size_t h = 0; h < 2u; h++) {
                uint8_t *ql = b + h * 64u, *qh = b + 128u + h * 32u;
                for (size_t idx = 0; idx < 32u; idx++) {
                    const uint8_t q0 = (uint8_t)(q6k_test_value(row, block, h, 0u, idx) + 32);
                    const uint8_t q1 = (uint8_t)(q6k_test_value(row, block, h, 1u, idx) + 32);
                    const uint8_t q2 = (uint8_t)(q6k_test_value(row, block, h, 2u, idx) + 32);
                    const uint8_t q3 = (uint8_t)(q6k_test_value(row, block, h, 3u, idx) + 32);
                    ql[idx] = (uint8_t)((q0 & 15u) | ((q2 & 15u) << 4u));
                    ql[idx + 32u] = (uint8_t)((q1 & 15u) | ((q3 & 15u) << 4u));
                    qh[idx] = (uint8_t)(((q0 >> 4u) & 3u) | (((q1 >> 4u) & 3u) << 2u) |
                                        (((q2 >> 4u) & 3u) << 4u) | (((q3 >> 4u) & 3u) << 6u));
                }
            }
        }
    }
}

static void ref_matmul_q6k(const float *x, size_t rows, size_t n_in,
                           size_t n_out, float *y) {
    const size_t bpr = n_in / 256u;
    for (size_t r = 0; r < rows; r++) {
        for (size_t o = 0; o < n_out; o++) {
            double acc = 0.0;
            for (size_t block = 0; block < bpr; block++)
                for (size_t h = 0; h < 2u; h++)
                    for (size_t s = 0; s < 4u; s++)
                        for (size_t idx = 0; idx < 32u; idx++) {
                            const size_t k = block * 256u + h * 128u + s * 32u + idx;
                            acc += (double) x[r * n_in + k] *
                                   (double) q6k_test_value(o, block, h, s, idx);
                        }
            y[r * n_out + o] = (float) acc;
        }
    }
}

static float exact_half(size_t a, size_t b) {
    return (float) ((int) ((a * 13u + b * 7u) % 9u) - 4) * 0.5f;
}

static struct geist_tensor t_f32_2d(struct geist_buffer *buf, size_t rows, size_t cols) {
    return (struct geist_tensor){.buffer = buf, .offset = 0, .dtype = GEIST_DTYPE_F32,
                                 .layout = GEIST_LAYOUT_DENSE, .ndim = 2,
                                 .shape = {(int64_t) rows, (int64_t) cols},
                                 .stride = {(int64_t) cols, 1}};
}
static struct geist_tensor t_q6k_2d(struct geist_buffer *buf, size_t rows, size_t cols) {
    return (struct geist_tensor){.buffer = buf, .offset = 0, .dtype = GEIST_DTYPE_Q6_K,
                                 .layout = GEIST_LAYOUT_BLOCK_QUANTIZED, .ndim = 2,
                                 .shape = {(int64_t) rows, (int64_t) cols}, .stride = {0, 0}};
}

static int run_matmul_q6k(const float *x_in, const uint8_t *w_in, size_t rows,
                          size_t n_in, size_t n_out, float *out_host) {
    struct geist_backend *be = nullptr;
    if (geist_backend_create("metal", nullptr, nullptr, &be) != GEIST_OK) return -1;
    int rc = -1;
    const struct geist_backend_vtbl *v = be->desc->vtbl;
    const size_t w_bytes = n_out * (n_in / 256u) * 210u;
    struct geist_buffer *bx = nullptr, *bw = nullptr, *by = nullptr;
    if (v->matmul_q6k == nullptr) goto done;
    enum geist_status s = v->buffer_create(be, rows * n_in * sizeof(float),
                                           GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_DEVICE, &bx);
    if (s == GEIST_OK) s = v->buffer_create(be, w_bytes, GEIST_BUFFER_WEIGHT, GEIST_MEMORY_DEVICE, &bw);
    if (s == GEIST_OK) s = v->buffer_create(be, rows * n_out * sizeof(float),
                                            GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_DEVICE, &by);
    if (s == GEIST_OK) s = v->buffer_upload(bx, rows * n_in * sizeof(float), (const uint8_t *) x_in);
    if (s == GEIST_OK) s = v->buffer_upload(bw, w_bytes, w_in);
    if (s == GEIST_OK) {
        struct geist_tensor x = t_f32_2d(bx, rows, n_in);
        struct geist_tensor w = t_q6k_2d(bw, n_out, n_in);
        struct geist_tensor y = t_f32_2d(by, rows, n_out);
        s = v->matmul_q6k(be, &x, &w, &y);
    }
    if (s == GEIST_OK) s = v->buffer_download(rows * n_out * sizeof(float), (uint8_t *) out_host, by);
    rc = (s == GEIST_OK) ? 0 : -1;
done:
    if (bx) v->buffer_destroy(be, bx);
    if (bw) v->buffer_destroy(be, bw);
    if (by) v->buffer_destroy(be, by);
    geist_backend_destroy(be);
    return rc;
}

static bool close_arr(const float *a, const float *b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        const float d = fabsf(a[i] - b[i]);
        if (d > 0.1f + 2e-3f * fabsf(b[i])) {
            fprintf(stderr, "  mismatch[%zu]: got %.3f want %.3f\n", i, a[i], b[i]);
            return false;
        }
    }
    return true;
}

static void one_shape(size_t rows, size_t n_in, size_t n_out) {
    char label[128];
    snprintf(label, sizeof label, "q6k matmul rows=%zu n_in=%zu n_out=%zu", rows, n_in, n_out);
    float *x = malloc(rows * n_in * sizeof(float));
    uint8_t *w = malloc(n_out * (n_in / 256u) * 210u);
    float *got = malloc(rows * n_out * sizeof(float));
    float *ref = malloc(rows * n_out * sizeof(float));
    for (size_t r = 0; r < rows; r++)
        for (size_t k = 0; k < n_in; k++) x[r * n_in + k] = exact_half(r, k);
    pack_q6k_test_matrix(n_in, n_out, w);
    ref_matmul_q6k(x, rows, n_in, n_out, ref);
    if (check(run_matmul_q6k(x, w, rows, n_in, n_out, got) == 0, label))
        check(close_arr(got, ref, rows * n_out), label);
    free(x); free(w); free(got); free(ref);
}

int main(void) {
    const size_t shapes[][3] = {
        {32, 256, 64}, {64, 512, 128}, {64, 2048, 256}, {40, 512, 192}, {50, 256, 100},
    };
    for (size_t i = 0; i < sizeof(shapes) / sizeof(shapes[0]); i++)
        one_shape(shapes[i][0], shapes[i][1], shapes[i][2]);
    if (g_fail == 0) { printf("PASS: all metal q6k matmul parity checks\n"); return 0; }
    fprintf(stderr, "%d FAILURES\n", g_fail);
    return 1;
}
