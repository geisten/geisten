/*
 * bench_sgemv — measure Apple BLAS sgemv throughput on Gemma-relevant
 * matrix sizes to verify the "decode is memory-bandwidth bound" claim.
 *
 * For each size, runs N calls and reports:
 *   wall_us/call   — average per-call time
 *   GB/s effective — weight-matrix bytes read per call / time
 *
 * If GB/s ≈ 50-80 GB/s we're at M1 unified-memory bandwidth ceiling.
 * If GB/s << 30 GB/s there's compute-side overhead we can reduce.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

typedef enum { CblasRowMajor = 101 } CBLAS_ORDER;
typedef enum { CblasNoTrans = 111, CblasTrans = 112 } CBLAS_TRANSPOSE;
extern void cblas_sgemv(CBLAS_ORDER,
                        CBLAS_TRANSPOSE,
                        int M,
                        int N,
                        float alpha,
                        const float* A,
                        int lda,
                        const float* X,
                        int incX,
                        float beta,
                        float* Y,
                        int incY);

static double now_us(void) {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return (double) tv.tv_sec * 1e6 + (double) tv.tv_usec;
}

static void bench(const char* name, int n_out, int n_in, int n_iter) {
    /* Allocate aligned-ish FP32 buffers. */
    float* W = (float*) aligned_alloc(64, (size_t) n_out * n_in * sizeof(float));
    float* x = (float*) aligned_alloc(64, (size_t) n_in * sizeof(float));
    float* y = (float*) aligned_alloc(64, (size_t) n_out * sizeof(float));
    /* Fill with non-zero data to defeat any zero-skip optimizations. */
    for (size_t i = 0; i < (size_t) n_out * n_in; i++)
        W[i] = (float) ((i & 0xff) - 128) * 0.01f;
    for (int i = 0; i < n_in; i++)
        x[i] = (float) ((i & 0x3f) - 32) * 0.01f;

    /* Warmup. */
    for (int i = 0; i < 3; i++) {
        cblas_sgemv(CblasRowMajor, CblasNoTrans, n_out, n_in, 1.0f, W, n_in, x, 1, 0.0f, y, 1);
    }

    double t0 = now_us();
    for (int it = 0; it < n_iter; it++) {
        cblas_sgemv(CblasRowMajor, CblasNoTrans, n_out, n_in, 1.0f, W, n_in, x, 1, 0.0f, y, 1);
    }
    double dt_us = (now_us() - t0) / n_iter;

    double bytes_per_call = (double) n_out * n_in * 4.0;
    double gbps = bytes_per_call / (dt_us * 1e3);
    double mb = bytes_per_call / (1024.0 * 1024.0);
    printf("  %-30s (%5d × %5d, %6.1f MB) | %6.0f us/call | %5.1f GB/s\n",
           name,
           n_out,
           n_in,
           mb,
           dt_us,
           gbps);

    free(W);
    free(x);
    free(y);
}

int main(void) {
    printf("Apple BLAS sgemv throughput on Gemma 4 E2B FP32-mirror sizes\n");
    printf("  size                              | wall          | bandwidth\n");

    /* Per-layer matmul sizes (one decode token = one call each). */
    bench("q_proj (8h × 256hd)", 4096, 1536, 200);
    bench("k_proj (1h × 256hd)", 256, 1536, 200);
    bench("v_proj (1h × 256hd)", 256, 1536, 200);
    bench("o_proj", 1536, 4096, 200);
    bench("gate (regular)", 6144, 1536, 200);
    bench("gate (kv-shared 2x)", 12288, 1536, 200);
    bench("up   (regular)", 6144, 1536, 200);
    bench("down (regular)", 1536, 6144, 200);
    bench("down (kv-shared 2x)", 1536, 12288, 100);
    bench("per_layer_gate", 256, 1536, 500);
    bench("per_layer_proj", 1536, 256, 500);

    /* Lm head — once per decode token. */
    bench("lm_head (262144 × 1536)", 262144, 1536, 5);

    /* Sanity: huge sgemv at 0.5 GB to see steady-state bandwidth. */
    bench("huge (32768 × 4096)", 32768, 4096, 10);

    return 0;
}
