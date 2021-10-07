/**
 * # geisten - The library for creating efficient binary neural networks
 *
 * author: Germar Schlegel
 * date: 01.09.2021
 *
 * This c header only library provides an efficient network architecture
 * and a set of two hyper-parameters in order to build very
 * small, low latency models that can be easily matched to the
 * design requirements for mobile and embedded vision applications.
 *
 * The 1-bit convolutional neural network (1-bit CNN, also known as binary neu-
 * ral network) [7,30], of which both weights and activations are binary, has
 * been recognized as one of the most promising neural network compression
 * methods for deploying models onto the resource-limited devices. It enjoys 32×
 * memory compression ratio, and up to 58×practical computational reduction on
 * CPU, as demonstrated in [30]. Moreover, with its pure logical computation
 * (i.e., XNOR operations between binary weights and binary activations), 1-bit
 * CNN is both highly energy-efficient for embedded devices [8,40], and
 * possesses the potential of being directly deployed on next generation
 * memristor-based hardware.
 *
 * The library is based on following papers:
 * [ReActNet: Towards Precise Binary Neural Network with Generalized Activation
 * Functions](https://arxiv.org/pdf/2003.03488.pdf)
 *
 * ## TODO
 * - Create loss function
 * - Create softmax function
 * - Optimize weights
 * -
 */

#include <err.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifdef WITH_THREADS
#include <omp.h>
#define MP_LOOP() PRAGMA(omp for simd)
#else
#define MP_LOOP()
#endif

#define PRAGMA(X) _Pragma(#X)
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

#define ARRAY_LENGTH(_arr) (sizeof(_arr) / sizeof((_arr)[0]))
#define BIT_SIZE(_type) (sizeof(_type) * CHAR_BIT)
#define BIT_SIZE_ULL BIT_SIZE(unsigned long long)
#define SIZE_BITS BIT_SIZE(size_t)
#define BIT_ARRAY_LEN(_n, _bits) (((_n)-1 + (_bits)) / (_bits))
#define BIT_ARRAY_SIZE(_arr, _bits) (BIT_ARRAY_LEN(ARRAY_LENGTH(_arr), (_bits)))

#define BIT_TEST(var, pos) (((var) & (1 << (pos))) != 0)
#define BIT_TEST_ARRAY(_arr, _pos) \
    (BIT_TEST(_arr[(_pos) / BIT_SIZE(_arr[0])], (_pos) % BIT_SIZE(_arr[0])))
#define BIT_SET(_arr, _pos) \
    _arr[(_pos) / BIT_SIZE(_arr[0])] |= (1LLU << ((_pos) % BIT_SIZE(_arr[0])))

#define BIT_AS_SIGN(var, pos) (BIT_TEST(var, pos) - !BIT_TEST(var, pos))

// deprecated
#define BINARIZE(_b, _i, _t, _v) (_b) |= ((_v) > (_t)) * (1LLU << (_i))

/**
 * The special operator __has_builtin (operand) is used to test whether
 * the symbol named by its operand is recognized as a built-in function by GCC
 * in the current language and conformance mode. The __has_builtin operator by
 * itself, without any operand or parentheses, acts as a predefined macro so
 * that support for it can be tested in portable code.
 */
#if defined __has_builtin
#if __has_builtin(__builtin_popcountll)
#define POPCOUNT(_x) __builtin_popcountll((_x))
#endif
#endif
#ifndef __builtin_popcountll
static unsigned long long popcount_soft(unsigned x) {
    unsigned long long c = 0;
    for (; x != 0; x &= x - 1) c++;
    return c;
}
#define POPCOUNT(_x) popcount_soft(_x)
#endif

/**
 * rprelu_fp16() - RPReLU function
 *
 * [Background](https://arxiv.org/pdf/2003.03488.pdf)
 */
static int rprelu(int x, int beta, int gamma, int zeta) {
    return (x - gamma) * ((x > gamma) + (x <= gamma) * beta) + zeta;
}

static int rprelu_derived(int x, int beta, int gamma) {
    return (x > gamma) + (x <= gamma) * beta;
}

/**
 * bnn() - Bitwise matrix multiplication
 * a and b are both binary values thus  a∗b can be implemented with bitwise
 * operations. In each bit, the result of basic multiplication a × b is one of
 * three values {−1,0,1}.
 * (Details)[https://arxiv.org/pdf/1909.11366.pdf]
 * @param a
 * @param b
 * @return
 */
static int16_t bmm(unsigned long long a, unsigned long long b) {
    unsigned long long pos = a & b;
    unsigned long long neg = a & (~b);
    return POPCOUNT(pos) - POPCOUNT(neg);
}

/**
 * ## binarize() - Binarizes 8 bit fix pont elements of array `x`
 * Binarizes all elements of array `x` and writes the result to the bit array
 * `b`. The conversion is as follows:
 *
 *     if x[i] > threshold then set bit=1 else set bit=0
 *
 * ### Parameter
 * - `x` The fix point array of size BIT_SIZE(unsigned long long)
 * - `threshold` The conversion threshold
 * Returns the bit array (of size BIT_SIZE(unsigned long long))
 */
static unsigned long long binarize(uint32_t size, const int8_t a[size],
                                   uint8_t level) {
    unsigned long long bin = 0;
    for (uint32_t i = 0; i < size; i++) {
        BINARIZE(bin, i, level, a[i]);
    }
    return bin;
}

static int error(uint32_t m, const int16_t a[m], const int16_t y[m]) {
    int result = 0;
    for (uint32_t i = 0; i < m; i++) {
        result += pow(y[i] - a[i], 2);
    }
    return result / m;
}

static void delta(uint32_t m, const int16_t a[m], const int16_t y[m],
                  int16_t d[m]) {
    for (uint32_t i = 0; i < m; i++) {
        d[i] = (int16_t)(y[i] - a[i]);
    }
}

/**
 *
 * @param n
 * @param wb
 * @param alpha
 * @param x
 * Return the sum of the element wise multiplication.
 */
static int forward(
    uint32_t n,
    const unsigned long long wb[(n / BIT_SIZE(unsigned long long)) + 1],
    int8_t alpha, const int8_t x[n]) {
    unsigned long long xb;
    int y;
    uint32_t i;
    for (i = 0, y = 0; i < (n / BIT_SIZE(wb[0]) + 1); i++) {
        xb = binarize(MIN(n, (1 + i) * BIT_SIZE(wb[0])),
                      &x[i * BIT_SIZE(wb[0])], alpha);
        y += bmm(xb, wb[i]);
    }
    return y;
}

static void backward(
    uint32_t m, uint32_t n,
    const unsigned long long wb[(n / BIT_SIZE(unsigned long long)) + 1],
    const int8_t y, int x[m]) {
    for (uint32_t i = 0; i < m; i++) {
        x[i] += (2 * BIT_TEST_ARRAY(wb, i) - 1) * y;
    }
}

static int update_activation(uint32_t m, const int16_t delta[m], int rate,
                             int alpha) {
    int delta_alpha = 0;
    for (uint32_t i; i < m; i++) {
        delta_alpha += delta[m];
    }
    return alpha - rate * delta_alpha / (int)m;
}

static void update_rprelu(uint32_t m, const int16_t delta[m], int rate,
                          int *beta, int *gamma, int *zeta) {
    int delta_beta  = 0;
    int delta_gamma = 0;
    int delta_zeta  = 0;
    for (uint32_t i; i < m; i++) {
        delta_beta += (delta[i] <= *gamma) * (delta[i] - *gamma);
        delta_gamma += -(delta[i] <= *gamma) * (*beta) * -(delta[i] > *gamma);
        delta_zeta += delta[i];
    }
    *beta -= rate * delta_beta / (int)m;
    *gamma -= rate * delta_gamma / (int)m;
    *zeta -= rate * delta_zeta / (int)m;
}

/**
 * TODO Fix Point calculus
 * TODO richtige berechnung col/rows
 * @param rate
 * @param delta
 * @param m
 * @param x
 * @param alpha
 * @param w
 */
static void update_weights(
    uint32_t m, const int16_t x[m], int rate, int delta, int alpha,
    unsigned long long w[m / BIT_SIZE(unsigned long long) + 1]) {
    int factor = rate * delta;
    unsigned long long bit;
    int result;
    for (uint32_t j = 0; j < m; j++) {
        bit    = BIT_TEST_ARRAY(w, j);
        result = (int)(bit - !bit) * alpha - factor * x[j];
        BINARIZE(w[j / BIT_SIZE(w[0])], (j % BIT_SIZE(w[0])), 0, result);
    }
}