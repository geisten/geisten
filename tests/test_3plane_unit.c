/*
 * test_3plane — verify linear_ptqtp_decode_3plane against a scalar reference.
 * Builds T1, T2, T3 trit-planes from a deterministic seed, encodes them in
 * the on-disk 3-plane format (1 byte per weight, idx = (T1+1)*9 + (T2+1)*3 +
 * (T3+1)), runs the kernel, and compares to a hand-rolled scalar dot product.
 */
#include "gguf_quant.h"
#include "ptqtp_kernel.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    const size_t n_in = 256;
    const size_t n_out = 8;
    const size_t group_size = 128;
    const size_t n_groups = n_in / group_size;

    /* Build random trits and alpha. */
    uint8_t* trits = (uint8_t*) aligned_alloc(64, n_out * n_in);
    float* alpha = (float*) aligned_alloc(64, n_out * n_groups * 3 * sizeof(float));
    int8_t* x_q8 = (int8_t*) aligned_alloc(64, n_in);

    /* Reference T1/T2/T3 in {-1, 0, +1}. */
    int8_t* T1 = (int8_t*) malloc(n_out * n_in);
    int8_t* T2 = (int8_t*) malloc(n_out * n_in);
    int8_t* T3 = (int8_t*) malloc(n_out * n_in);
    srand(42);
    for (size_t i = 0; i < n_out * n_in; i++) {
        T1[i] = (int8_t) ((rand() % 3) - 1);
        T2[i] = (int8_t) ((rand() % 3) - 1);
        T3[i] = (int8_t) ((rand() % 3) - 1);
        trits[i] = (uint8_t) ((T1[i] + 1) * 9 + (T2[i] + 1) * 3 + (T3[i] + 1));
    }
    for (size_t i = 0; i < n_out * n_groups * 3; i++) {
        alpha[i] = ((float) rand() / (float) RAND_MAX) * 0.1f;
    }
    for (size_t i = 0; i < n_in; i++)
        x_q8[i] = (int8_t) ((rand() % 256) - 128);
    const float scale_x = 0.005f;

    /* Reference: scalar y[n] = scale_x * sum_g (a1 sum_k T1*x + a2 sum T2*x + a3 sum T3*x). */
    float* y_ref = (float*) malloc(n_out * sizeof(float));
    for (size_t n = 0; n < n_out; n++) {
        float acc = 0.0f;
        for (size_t g = 0; g < n_groups; g++) {
            int32_t a1_acc = 0, a2_acc = 0, a3_acc = 0;
            for (size_t k = 0; k < group_size; k++) {
                const size_t off = n * n_in + g * group_size + k;
                a1_acc += (int32_t) T1[off] * x_q8[g * group_size + k];
                a2_acc += (int32_t) T2[off] * x_q8[g * group_size + k];
                a3_acc += (int32_t) T3[off] * x_q8[g * group_size + k];
            }
            acc += alpha[n * n_groups * 3 + g * 3 + 0] * (float) a1_acc +
                   alpha[n * n_groups * 3 + g * 3 + 1] * (float) a2_acc +
                   alpha[n * n_groups * 3 + g * 3 + 2] * (float) a3_acc;
        }
        y_ref[n] = scale_x * acc;
    }

    /* Kernel. */
    float* y_kernel = (float*) malloc(n_out * sizeof(float));
    ptqtp_gemv_3plane_fp32alpha(n_in, n_out, group_size, x_q8, scale_x, trits, alpha, y_kernel);

    /* Compare. */
    int fails = 0;
    for (size_t n = 0; n < n_out; n++) {
        const float diff = y_kernel[n] - y_ref[n];
        const float rel = fabsf(diff) / (fabsf(y_ref[n]) + 1e-6f);
        if (rel > 1e-4f) {
            printf("  row %zu: kernel %+.6f  ref %+.6f  diff %+.6f  rel %.4e\n",
                   n,
                   y_kernel[n],
                   y_ref[n],
                   diff,
                   rel);
            fails++;
        }
    }
    if (fails == 0) {
        printf("PASS: 3-plane kernel matches scalar reference (%zu rows)\n", n_out);
    } else {
        printf("FAIL: %d/%zu rows mismatched\n", fails, n_out);
    }

    free(trits);
    free(alpha);
    free(x_q8);
    free(T1);
    free(T2);
    free(T3);
    free(y_ref);
    free(y_kernel);
    return fails == 0 ? 0 : 1;
}
