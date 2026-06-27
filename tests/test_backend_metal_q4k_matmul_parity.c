/*
 * test_backend_metal_q4k_matmul_parity — numerical parity gate for the Metal
 * Q4_K *matmul* (rows>1) path: the base/m8/m16 tiled kernels and the
 * simdgroup-matmul GEMM (mm_sg). This is the gate the prefill optimization
 * work requires: it did not exist before (the metal q4k unit test only checks
 * metal-absence; test_backend_ops_unit covers cpu/vulkan matvec, never a
 * metal rows>1 matmul).
 *
 * Method: pack a deterministic Q4_K weight (scale=1, min=0 so dequant ==
 * q4k_test_value), build an F32 activation whose values are exactly
 * representable in fp16 (so the mm_sg half-input path is bit-faithful), run
 * vtbl->matmul_q4k on the metal backend, read back via buffer_download, and
 * compare to a per-row CPU reference. Runs each conforming shape with mm_sg
 * OFF (m8/m16 path) and ON (simdgroup GEMM).
 *
 * Build (standalone, against a metal libgeist.a):
 *   cc -std=c23 -O2 -DGEIST_BACKEND_METAL=1 -Iinclude -Isrc/... \
 *      tests/test_backend_metal_q4k_matmul_parity.c lib/.../libgeist.a \
 *      -framework Accelerate -lomp -lm
 */
#include <geist.h>
#include <geist_backend.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fail = 0;
static int check(bool cond, const char *what) {
    if (!cond) {
        fprintf(stderr, "FAIL: %s\n", what);
        g_fail++;
    }
    return cond ? 0 : 1;
}

/* ---- deterministic Q4_K test matrix (copied from test_backend_ops_unit) ---- */
static uint8_t q4k_test_value(size_t row, size_t block, size_t sub, size_t idx) {
    return (uint8_t) ((row * 3u + block * 5u + sub * 7u + idx) & 15u);
}

static void pack_q4k_test_matrix(size_t n_in, size_t n_out, uint8_t *dst) {
    const size_t blocks_per_row = n_in / 256u;
    for (size_t row = 0; row < n_out; row++) {
        for (size_t block = 0; block < blocks_per_row; block++) {
            uint8_t *b = dst + (row * blocks_per_row + block) * 144u;
            memset(b, 0, 144u);
            b[1] = 0x3cu; /* fp16(1.0) scale */
            b[3] = 0x00u; /* fp16(0.0) min  */
            b[4] = b[5] = b[6] = b[7] = 1u;
            b[12] = b[13] = b[14] = b[15] = 1u;
            for (size_t pair = 0; pair < 4u; pair++) {
                for (size_t idx = 0; idx < 32u; idx++) {
                    const uint8_t lo = q4k_test_value(row, block, pair * 2u, idx);
                    const uint8_t hi = q4k_test_value(row, block, pair * 2u + 1u, idx);
                    b[16u + pair * 32u + idx] = (uint8_t) (lo | (hi << 4u));
                }
            }
        }
    }
}

/* Per-row reference: y[r,row] = sum_k x[r,k] * dequant(w[row,k]). */
static void ref_matmul_q4k(const float *x, size_t rows, size_t n_in,
                           size_t n_out, float *y) {
    const size_t bpr = n_in / 256u;
    for (size_t r = 0; r < rows; r++) {
        for (size_t row = 0; row < n_out; row++) {
            double acc = 0.0;
            for (size_t block = 0; block < bpr; block++) {
                for (size_t sub = 0; sub < 8u; sub++) {
                    for (size_t idx = 0; idx < 32u; idx++) {
                        const size_t k = block * 256u + sub * 32u + idx;
                        acc += (double) x[r * n_in + k] *
                               (double) q4k_test_value(row, block, sub, idx);
                    }
                }
            }
            y[r * n_out + row] = (float) acc;
        }
    }
}

/* fp16-exact activation: values in {-2,-1.5,...,2}. */
static void fill_x(float *x, size_t rows, size_t n_in) {
    for (size_t r = 0; r < rows; r++) {
        for (size_t k = 0; k < n_in; k++) {
            x[r * n_in + k] = (float) ((int) ((r * 13u + k * 7u) % 9u) - 4) * 0.5f;
        }
    }
}

static struct geist_tensor t_f32_2d(struct geist_buffer *buf, size_t rows,
                                    size_t cols) {
    return (struct geist_tensor){.buffer = buf, .offset = 0,
                                 .dtype = GEIST_DTYPE_F32,
                                 .layout = GEIST_LAYOUT_DENSE, .ndim = 2,
                                 .shape = {(int64_t) rows, (int64_t) cols},
                                 .stride = {(int64_t) cols, 1}};
}
static struct geist_tensor t_q4k_2d(struct geist_buffer *buf, size_t rows,
                                    size_t cols) {
    return (struct geist_tensor){.buffer = buf, .offset = 0,
                                 .dtype = GEIST_DTYPE_Q4_K,
                                 .layout = GEIST_LAYOUT_BLOCK_QUANTIZED, .ndim = 2,
                                 .shape = {(int64_t) rows, (int64_t) cols},
                                 .stride = {0, 0}};
}

/* Returns 0 and fills out_host[rows*n_out] on success, -1 otherwise. */
static int run_matmul_q4k(const float *x_in, const uint8_t *w_in, size_t rows,
                          size_t n_in, size_t n_out, float *out_host) {
    struct geist_backend *be = nullptr;
    if (geist_backend_create("metal", nullptr, nullptr, &be) != GEIST_OK) {
        return -1;
    }
    int rc = -1;
    const size_t w_bytes = n_out * (n_in / 256u) * 144u;
    struct geist_buffer *bx = nullptr, *bw = nullptr, *by = nullptr;
    const struct geist_backend_vtbl *v = be->desc->vtbl;
    if (v->matmul_q4k == nullptr) { goto done; }
    enum geist_status s = v->buffer_create(be, rows * n_in * sizeof(float),
                                           GEIST_BUFFER_ACTIVATION,
                                           GEIST_MEMORY_DEVICE, &bx);
    if (s == GEIST_OK)
        s = v->buffer_create(be, w_bytes, GEIST_BUFFER_WEIGHT,
                             GEIST_MEMORY_DEVICE, &bw);
    if (s == GEIST_OK)
        s = v->buffer_create(be, rows * n_out * sizeof(float),
                             GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_DEVICE, &by);
    if (s == GEIST_OK)
        s = v->buffer_upload(bx, rows * n_in * sizeof(float),
                             (const uint8_t *) x_in);
    if (s == GEIST_OK) s = v->buffer_upload(bw, w_bytes, w_in);
    if (s == GEIST_OK) {
        struct geist_tensor x = t_f32_2d(bx, rows, n_in);
        struct geist_tensor w = t_q4k_2d(bw, n_out, n_in);
        struct geist_tensor y = t_f32_2d(by, rows, n_out);
        s = v->matmul_q4k(be, &x, &w, &y);
    }
    if (s == GEIST_OK)
        s = v->buffer_download(rows * n_out * sizeof(float),
                               (uint8_t *) out_host, by);
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
        const float tol = 0.1f + 2e-3f * fabsf(b[i]);
        if (d > tol) {
            fprintf(stderr, "  mismatch[%zu]: got %.4f want %.4f (d=%.4f)\n",
                    i, a[i], b[i], d);
            return false;
        }
    }
    return true;
}

static void one_shape(size_t rows, size_t n_in, size_t n_out, bool mm_sg) {
    char label[128];
    snprintf(label, sizeof label, "q4k matmul rows=%zu n_in=%zu n_out=%zu mm_sg=%d",
             rows, n_in, n_out, (int) mm_sg);
    if (mm_sg) {
        setenv("GEIST_METAL_Q4K_MM_SG", "1", 1);
        setenv("GEIST_METAL_Q4K_MM_SG_UNSAFE", "1", 1);
    } else {
        unsetenv("GEIST_METAL_Q4K_MM_SG");
        unsetenv("GEIST_METAL_Q4K_MM_SG_UNSAFE");
    }
    float *x = malloc(rows * n_in * sizeof(float));
    uint8_t *w = malloc(n_out * (n_in / 256u) * 144u);
    float *got = malloc(rows * n_out * sizeof(float));
    float *ref = malloc(rows * n_out * sizeof(float));
    fill_x(x, rows, n_in);
    pack_q4k_test_matrix(n_in, n_out, w);
    ref_matmul_q4k(x, rows, n_in, n_out, ref);
    int r = run_matmul_q4k(x, w, rows, n_in, n_out, got);
    if (check(r == 0, label)) {
        check(close_arr(got, ref, rows * n_out), label);
    }
    free(x); free(w); free(got); free(ref);
}

int main(void) {
    /* conforming (mm_sg eligible: rows%32==0, n_out%64==0) — tested both paths */
    const size_t conforming[][3] = {
        {32, 256, 64}, {64, 512, 128}, {64, 2048, 1536}, {32, 1024, 256},
    };
    for (size_t i = 0; i < sizeof(conforming) / sizeof(conforming[0]); i++) {
        one_shape(conforming[i][0], conforming[i][1], conforming[i][2], false);
        one_shape(conforming[i][0], conforming[i][1], conforming[i][2], true);
    }
    /* non-conforming (mm_sg guard fails → m8/m16 fallback) */
    const size_t fallback[][3] = {
        {16, 512, 128}, {40, 512, 192}, {50, 256, 100}, {8, 768, 256},
    };
    for (size_t i = 0; i < sizeof(fallback) / sizeof(fallback[0]); i++) {
        one_shape(fallback[i][0], fallback[i][1], fallback[i][2], true);
    }
    if (g_fail == 0) {
        printf("PASS: all metal q4k matmul parity checks\n");
        return 0;
    }
    fprintf(stderr, "%d FAILURES\n", g_fail);
    return 1;
}
