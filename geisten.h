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

#define ARRAY_LENGTH(_arr) (sizeof(_arr) / sizeof((_arr)[0]))
#define BIT_SIZE(_type) (sizeof(_type) * CHAR_BIT)
#define BIT_ARRAY_LEN(_n, _bits) (((_n)-1 + (_bits)) / (_bits))
#define BIT_ARRAY_SIZE(_arr, _bits) (BIT_ARRAY_LEN(ARRAY_LENGTH(_arr), (_bits)))

#define BIT_TEST(var, pos) (((var) & (1 << (pos))) != 0)
#define BIT_TEST_ARRAY(_arr, _pos) \
    (BIT_TEST(_arr[(_pos) / BIT_SIZE(_arr[0])], (_pos) % BIT_SIZE(_arr[0])))

#define BIT2SIGN(_b, _i) (2 * BIT_TEST_ARRAY(_b, _i) - 1)

// deprecated
#define BINARIZE(_b, _i, _t, _v) \
    (_b) = ((_v) > (_t)) ? (_b) | (1LLU << (_i)) : (_b) & ~(1LLU << (_i))

#define BINARIZE_POS(_b, _i, _t, _v)                                         \
    BINARIZE((_b)[(_i) / BIT_SIZE((_b)[0])], (_i) % BIT_SIZE((_b)[0]), (_t), \
             (_v))

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
 * - `x` The function variable
 * - `beta` The slope constant, if x <= gamma
 * - `gamma` The level constant
 * - `zeta` The offset constant
 *
 * [Background](https://arxiv.org/pdf/2003.03488.pdf)
 * _Returns the function result_
 */
static int rprelu(int x, int beta, int gamma, int zeta) {
    return (x - gamma) * ((x > gamma) + (x <= gamma) * beta) + zeta;
}

/**
 * ## rprelu_derived() - The derived RPReLU function
 * - `x` The function variable
 * - `beta` The slope constant, if x <= gamma
 * - `gamma` The level constant
 *
 * _Returns the derived function value_
 */
static int rprelu_derived(int x, int beta, int gamma) {
    return (x > gamma) + (x <= gamma) * beta;
}

/**
 * ## binarize() - Binarizes 8 bit fix pont elements of array `x`
 * - `x` The fix point array of size BIT_SIZE(unsigned long long)
 * - `threshold` The conversion threshold
 * 
 * Binarizes all elements of array `x` and writes the result to the bit array
 * `b`. The conversion is as follows:
 *
 *     if x[i] > threshold then set bit=1 else set bit=0
 * 
 * Returns the bit array (of size BIT_SIZE(unsigned long long))
 */
static void binarize(
    uint32_t size, const int8_t a[size], uint8_t level,
    unsigned long long result[(size / BIT_SIZE(unsigned long long)) + 1]) {
    for (uint32_t i = 0; i < size; i++) {
        BINARIZE_POS(result, i, level, a[i]);
    }
}

/**
 * ## forward() - Linear forward transformation
 * - `m` The number of input cells (weight matrix rows)
 * - `wb` The `j`th column of the binary m x n weight matrix
 * - `x` The activation binary array
 *
 * ### Bitwise matrix multiplication
 *
 * `xb` and `wb[i]` are both binary values thus  `xb ∗ wb[i]` can be implemented with bitwise
 * operations. In each bit, the result of basic multiplication `xb × wb[i]` is one of
 * three values {−1,0,1}.
 * (Details)[https://arxiv.org/pdf/1909.11366.pdf]
 *
 * Return the sum of the element wise multiplication.
 */
static int forward(uint32_t m, const unsigned long long wb[m],
                   const unsigned long long x[m]) {
    int y;
    uint32_t i;
    for (i = 0, y = 0; i < m; i++) {
        y += (int)POPCOUNT((x[i] & wb[i])) - (int)POPCOUNT((x[i] & (~wb[i])));
    }
    return y;
}

static unsigned long long mixin(unsigned long long bits, int n01) {
    return bits;
}

/**
 * ## backward() - Backward propagation
 * - `m` The number of input cells (weight matrix rows)
 * - `wb` The `j`th column of the binary m x n weight matrix
 * - `y` The output cell at position `j`
 * - `x` The input vector (of size `m`)
 *
 * The resulting x vector must also be multiplied by the weight factor to obtain
 * the correct value.
 */
static void backward(
    uint32_t m,
    const unsigned long long wb[(m / BIT_SIZE(unsigned long long)) + 1],
    const int y, int x[m]) {
    for (uint32_t i = 0; i < m; i++) {
        x[i] += BIT2SIGN(wb, i) * y;
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

static void update_rprelu(uint32_t m, const int delta[m], int rate, int *beta,
                          int *gamma, int *zeta) {
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
 * ## update_weights() - Update the weight matrix
 * - `m` The number of input cells (weight matrix rows)
 * - `x` The input vector (of size `m`) 
 * - `delta` The output cell value (delta y) at position `j`
 * - `w` The binary m x n weight matrix to be updated
 * 
 * The function updates a column of a binary matrix.
 * The matrix is binarized around the 0 value. 
 * If the value is greater than 0, then the updated binary 
 * matrix is set to 1; otherwise set to 0 
 * The learning rate can be controlled via the delta value 
 * (e.g.: delta_learn = delta * rate)
 * 
 */
static int update_weights_i(
    uint32_t m, const int x[m], int delta, int alpha,
    unsigned long long w[(m / BIT_SIZE(unsigned long long)) + 1]) {
    int result = 0;
    for (uint32_t i = 0; i < m; i++) {
        result = BIT2SIGN(w, i) * alpha - x[i] * delta;
        BINARIZE(w[i / BIT_SIZE(w[0])], i % BIT_SIZE(w[0]), 0, result);
    }
    return result;
}

/**
 * ## update_weights() - Update the weight matrix
 * - `m` The number of input cells (weight matrix rows)
 * - `x` The **8 bit** input vector (of size `m`)
 * - `delta` The output cell value (delta y) at position `j`
 * - `w` The binary m x n weight matrix to be updated
 *
 * The function updates a column of a binary matrix.
 * The matrix is binarized around the 0 value.
 * If the value is greater than 0, then the updated binary
 * matrix is set to 1; otherwise set to 0
 * The learning rate can be controlled via the delta value
 * (e.g.: delta_learn = delta * rate)
 *
 */
static int update_weights_i8(
    uint32_t m, const int8_t x[m], int delta, int alpha,
    unsigned long long w[(m / BIT_SIZE(unsigned long long)) + 1]) {
    int result = 0;
    for (uint32_t i = 0; i < m; i++) {
        result = BIT2SIGN(w, i) * alpha - x[i] * delta;
        BINARIZE(w[i / BIT_SIZE(w[0])], i % BIT_SIZE(w[0]), 0, result);
    }
    return result;
}

#define update_weights(_m, _x, _delta, _alpha, _w) \
    _Generic((_x),\
             int8_t *: update_weights_i8,\
               int *: update_weights_i, \
               default: update_weights_i)((_m), (_x), (_delta), (_alpha), (_w))
