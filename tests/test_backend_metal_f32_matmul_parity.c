/*
 * test_backend_metal_f32_matmul_parity — numerical parity gate for the Metal
 * F32 dense matmul (rows>1): the reduction kernel and the simdgroup-GEMM
 * variant (f32 mm_sg) that prefill uses for the PLE gate / f32 projections.
 *
 * Method: random f32 weight [n_out,n_in] + f32 activation [rows,n_in] with
 * fp16-exact values (so a half-staged GEMM is bit-faithful), run
 * vtbl->matmul_f32_dense on metal, read back via buffer_download, compare to a
 * CPU reference. Runs conforming shapes (mm_sg eligible) and fallback shapes.
 *
 * Build: like test_backend_metal_q4k_matmul_parity.c.
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

/* fp16-exact: values in {-2,-1.5,...,2} so a half-staged GEMM is exact. */
static float exact_half(size_t a, size_t b) {
    return (float) ((int) ((a * 13u + b * 7u) % 9u) - 4) * 0.5f;
}

static void ref_matmul_f32(const float *x, const float *w, size_t rows,
                           size_t n_in, size_t n_out, float *y) {
    for (size_t r = 0; r < rows; r++) {
        for (size_t o = 0; o < n_out; o++) {
            double acc = 0.0;
            for (size_t k = 0; k < n_in; k++) {
                acc += (double) x[r * n_in + k] * (double) w[o * n_in + k];
            }
            y[r * n_out + o] = (float) acc;
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

static int run_matmul_f32(const float *x_in, const float *w_in, size_t rows,
                          size_t n_in, size_t n_out, float *out_host) {
    struct geist_backend *be = nullptr;
    if (geist_backend_create("metal", nullptr, nullptr, &be) != GEIST_OK) {
        return -1;
    }
    int rc = -1;
    const struct geist_backend_vtbl *v = be->desc->vtbl;
    struct geist_buffer *bx = nullptr, *bw = nullptr, *by = nullptr;
    if (v->matmul_f32_dense == nullptr) { goto done; }
    enum geist_status s = v->buffer_create(be, rows * n_in * sizeof(float),
                                           GEIST_BUFFER_ACTIVATION,
                                           GEIST_MEMORY_DEVICE, &bx);
    if (s == GEIST_OK)
        s = v->buffer_create(be, n_out * n_in * sizeof(float),
                             GEIST_BUFFER_WEIGHT, GEIST_MEMORY_DEVICE, &bw);
    if (s == GEIST_OK)
        s = v->buffer_create(be, rows * n_out * sizeof(float),
                             GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_DEVICE, &by);
    if (s == GEIST_OK)
        s = v->buffer_upload(bx, rows * n_in * sizeof(float),
                             (const uint8_t *) x_in);
    if (s == GEIST_OK)
        s = v->buffer_upload(bw, n_out * n_in * sizeof(float),
                             (const uint8_t *) w_in);
    if (s == GEIST_OK) {
        struct geist_tensor x = t_f32_2d(bx, rows, n_in);
        struct geist_tensor w = t_f32_2d(bw, n_out, n_in);
        struct geist_tensor y = t_f32_2d(by, rows, n_out);
        s = v->matmul_f32_dense(be, &x, &w, &y);
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
            fprintf(stderr, "  mismatch[%zu]: got %.4f want %.4f\n", i, a[i], b[i]);
            return false;
        }
    }
    return true;
}

static void one_shape(size_t rows, size_t n_in, size_t n_out) {
    char label[128];
    snprintf(label, sizeof label, "f32 matmul rows=%zu n_in=%zu n_out=%zu",
             rows, n_in, n_out);
    float *x = malloc(rows * n_in * sizeof(float));
    float *w = malloc(n_out * n_in * sizeof(float));
    float *got = malloc(rows * n_out * sizeof(float));
    float *ref = malloc(rows * n_out * sizeof(float));
    for (size_t r = 0; r < rows; r++)
        for (size_t k = 0; k < n_in; k++) x[r * n_in + k] = exact_half(r, k);
    for (size_t o = 0; o < n_out; o++)
        for (size_t k = 0; k < n_in; k++) w[o * n_in + k] = exact_half(o + 1, k + 2);
    ref_matmul_f32(x, w, rows, n_in, n_out, ref);
    if (check(run_matmul_f32(x, w, rows, n_in, n_out, got) == 0, label)) {
        check(close_arr(got, ref, rows * n_out), label);
    }
    free(x); free(w); free(got); free(ref);
}

int main(void) {
    const size_t shapes[][3] = {
        {32, 256, 64}, {64, 512, 128}, {64, 2048, 256}, {32, 1024, 512},
        {16, 512, 128}, {40, 384, 192}, {50, 256, 100}, {8, 768, 64},
    };
    for (size_t i = 0; i < sizeof(shapes) / sizeof(shapes[0]); i++) {
        one_shape(shapes[i][0], shapes[i][1], shapes[i][2]);
    }
    if (g_fail == 0) { printf("PASS: all metal f32 matmul parity checks\n"); return 0; }
    fprintf(stderr, "%d FAILURES\n", g_fail);
    return 1;
}
