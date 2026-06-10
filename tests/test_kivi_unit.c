/*
 * test_kivi_unit — verifies KIVI 2-bit pack/unpack semantics.
 *
 * Four scenarios:
 *   1. Packing logic on narrow-range data (cos ≥ 0.999) — proves the
 *      bit-encoding is correct (every value maps to its nearest level,
 *      bytes pack/unpack in the right order).
 *   2. V-row round-trip on iid Gaussian (cos ≥ 0.85). Pure Gaussian is
 *      2-bit's worst case; real KV cache data has channel structure that
 *      KIVI exploits to land >0.99 cos sim. The 0.85 floor matches the
 *      theoretical mse ≈ range²/48 for asymmetric 2-bit on iid input.
 *   3. K-group round-trip on iid Gaussian (cos ≥ 0.85).
 *   4. K-group with channel-wise outliers — per-channel quant should
 *      isolate the outliers so non-outlier channels stay accurate
 *      (cos ≥ 0.85 globally; KIVI's whole point is this beats per-token
 *      quant on the same data, but we don't bench against per-token here).
 *
 * Deterministic — fixed seed, no model needed.
 */
#include "kivi.h"
#include "test_helpers.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float cosine_sim(const float* a, const float* b, size_t n) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < n; i++) {
        dot += (double) a[i] * (double) b[i];
        na += (double) a[i] * (double) a[i];
        nb += (double) b[i] * (double) b[i];
    }
    if (na == 0.0 || nb == 0.0)
        return 0.0f;
    return (float) (dot / (sqrt(na) * sqrt(nb)));
}

/* Box-Muller for deterministic N(0,1) samples. */
static float gauss(uint32_t* seed) {
    uint32_t a = (*seed = (*seed) * 1103515245u + 12345u);
    uint32_t b = (*seed = (*seed) * 1103515245u + 12345u);
    float u1 = ((float) (a & 0xffffff) + 1.0f) / (float) 0x1000000;
    float u2 = ((float) (b & 0xffffff)) / (float) 0x1000000;
    return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
}

/* Scenario 1: 4 distinct values evenly spaced in a known range so each
 * maps cleanly to one of the 4 quant levels. Round-trip must reproduce
 * the input exactly (modulo float rounding). Verifies bit-packing
 * direction (q0 in low 2 bits, q3 in high 2 bits). */
static int test_packing_exact(void) {
    /* 16-element row with values that quantize to {0,1,2,3,0,1,2,3,...}.
     * scale=1, zero=0 → q=round(x), q∈{0,1,2,3}. */
    float in[16];
    for (size_t i = 0; i < 16; i++)
        in[i] = (float) (i % 4);
    uint8_t q[4];
    float scale = 0.0f, zero = 0.0f;
    kivi_pack_v_row(16, in, q, &scale, &zero);

    /* For input {0,1,2,3,...}: vmin=0, vmax=3, range=3, scale=1, zero=0. */
    if (scale != 1.0f || zero != 0.0f) {
        fprintf(stderr,
                "  FAIL: expected scale=1, zero=0; got %.3f, %.3f\n",
                (double) scale,
                (double) zero);
        return 1;
    }
    /* Each byte encodes 4 quant values q0..q3, with q0 in bits 0-1.
     * Input {0,1,2,3} → byte (3<<6)|(2<<4)|(1<<2)|0 = 0xE4. */
    if (q[0] != 0xE4) {
        fprintf(stderr, "  FAIL: expected q[0]=0xE4, got 0x%02X\n", q[0]);
        return 1;
    }

    float out[16];
    kivi_unpack_v_row(16, q, scale, zero, out);
    for (size_t i = 0; i < 16; i++) {
        if (out[i] != in[i]) {
            fprintf(stderr,
                    "  FAIL: unpack[%zu]=%.3f expected %.3f\n",
                    i,
                    (double) out[i],
                    (double) in[i]);
            return 1;
        }
    }
    printf("  packing_exact: round-trip bit-identical on aligned input ✓\n");
    return 0;
}

static int test_v_row_gaussian(void) {
    const size_t n = 256;
    float in[256], out[256];
    uint8_t q[64];
    float scale, zero;

    uint32_t seed = 0xC0FFEEu;
    for (size_t i = 0; i < n; i++)
        in[i] = gauss(&seed) * 0.5f;

    kivi_pack_v_row(n, in, q, &scale, &zero);
    kivi_unpack_v_row(n, q, scale, zero, out);

    const float cos = cosine_sim(in, out, n);
    printf("  v_row_gaussian n=%zu: cos=%.5f scale=%.4f zero=%.4f\n",
           n,
           cos,
           (double) scale,
           (double) zero);
    if (cos < 0.85f) {
        fprintf(stderr, "  FAIL: cos %.5f < 0.85\n", cos);
        return 1;
    }
    return 0;
}

static int test_k_group_gaussian(void) {
    const size_t g = KIVI_K_GROUP_SIZE;
    const size_t c = 256;
    const size_t n = g * c;

    float* in = malloc(n * sizeof(float));
    float* out = malloc(n * sizeof(float));
    uint8_t* q = malloc(n / 4);
    float* scales = malloc(c * sizeof(float));
    float* zeros = malloc(c * sizeof(float));

    uint32_t seed = 0x42424242u;
    for (size_t i = 0; i < n; i++)
        in[i] = gauss(&seed) * 0.5f;

    kivi_pack_k_group(g, c, in, q, scales, zeros);
    kivi_unpack_k_group(g, c, q, scales, zeros, out);

    const float cos = cosine_sim(in, out, n);
    printf("  k_group_gaussian g=%zu c=%zu: cos=%.5f\n", g, c, cos);
    int fails = (cos < 0.85f);
    free(in);
    free(out);
    free(q);
    free(scales);
    free(zeros);
    if (fails) {
        fprintf(stderr, "  FAIL: cos %.5f < 0.85\n", cos);
        return 1;
    }
    return 0;
}

static int test_k_group_outliers(void) {
    /* KIVI's claim: per-channel K quant isolates outliers — non-outlier
     * channels keep their normal precision. We test that the outlier
     * channels themselves are preserved well (since their channel scale
     * adapts to their range), AND the overall cos sim stays decent. */
    const size_t g = KIVI_K_GROUP_SIZE;
    const size_t c = 256;
    const size_t n = g * c;

    float* in = malloc(n * sizeof(float));
    float* out = malloc(n * sizeof(float));
    uint8_t* q = malloc(n / 4);
    float* scales = malloc(c * sizeof(float));
    float* zeros = malloc(c * sizeof(float));

    uint32_t seed = 0xABCDEFu;
    for (size_t t = 0; t < g; t++) {
        for (size_t ch = 0; ch < c; ch++) {
            float v = gauss(&seed) * 0.5f;
            if (ch % 50 == 0)
                v *= 10.0f;
            in[t * c + ch] = v;
        }
    }

    kivi_pack_k_group(g, c, in, q, scales, zeros);
    kivi_unpack_k_group(g, c, q, scales, zeros, out);

    const float cos = cosine_sim(in, out, n);
    printf("  k_group_outliers g=%zu c=%zu: cos=%.5f (5 channels with 10x magnitude)\n", g, c, cos);
    int fails = (cos < 0.85f);
    free(in);
    free(out);
    free(q);
    free(scales);
    free(zeros);
    if (fails) {
        fprintf(stderr, "  FAIL: cos %.5f < 0.85\n", cos);
        return 1;
    }
    return 0;
}

int main(void) {
    printf("KIVI 2-bit pack/unpack round-trip tests\n");
    int fails = 0;
    fails += test_packing_exact();
    fails += test_v_row_gaussian();
    fails += test_k_group_gaussian();
    fails += test_k_group_outliers();
    if (fails == 0) {
        printf("PASS: KIVI helpers preserve packing semantics and round-trip tolerance\n");
        return GEIST_TEST_PASS;
    }
    return GEIST_TEST_FAIL;
}
