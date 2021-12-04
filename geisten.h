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

#pragma once

#include <err.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define ARRAY_LENGTH(_arr) (sizeof(_arr) / sizeof((_arr)[0]))
#define NBITS(_type) (sizeof(_type) * CHAR_BIT)
#define BIT_ARRAY_LEN(_n, _bits) (((_n)-1 + (_bits)) / (_bits))
#define BIT_ARRAY_SIZE(_arr, _bits) (BIT_ARRAY_LEN(ARRAY_LENGTH(_arr), (_bits)))

// deprecated
#define BINARIZE(_b, _i, _t, _v) \
    (_b) = ((_v) > (_t)) ? (_b) | (1LLU << (_i)) : (_b) & ~(1LLU << (_i))

#define BINARIZE_POS(_b, _i, _t, _v) \
    BINARIZE((_b)[(_i) / NBITS((_b)[0])], (_i) % NBITS((_b)[0]), (_t), (_v))

/**
 * The special operator __has_builtin (operand) is used to test whether
 * the symbol named by its operand is recognized as a built-in function by GCC
 * in the current language and conformance mode. The __has_builtin operator by
 * itself, without any operand or parentheses, acts as a predefined macro so
 * that support for it can be tested in portable code.
 */

#ifndef __builtin_popcountll
static unsigned long long popcount_soft(unsigned x) {
    unsigned long long c = 0;
    for (; x != 0; x &= x - 1) c++;
    return c;
}
#define builtin_popcountll(_x) popcount_soft((_x))
#endif

/**
 * relu() - ReLU function
 * - `x` The function variable
 *
 * [Background](https://arxiv.org/pdf/2003.03488.pdf)
 * _Returns the function result_
 */
static int relu(int x) { return x * (x > 0); }

/**
 * ## binarize() - Binarizes 8 bit fix pont elements of array `x`
 * - `x` The fix point array of size NBITS(unsigned long long)
 * - `threshold` The conversion threshold
 * 
 * Binarizes all elements of array `x` and writes the result to the bit array
 * `b`. The conversion is as follows:
 *
 *     if x[i] > threshold then set bit=1 else set bit=0
 * 
 * Returns the bit array (of size NBITS(unsigned long long))
 */
static void binarize(
    uint32_t size, const int8_t a[size], uint8_t level,
    unsigned long long result[(size / NBITS(unsigned long long)) + 1]) {
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
        y += (int)builtin_popcountll((x[i] & wb[i])) -
             (int)builtin_popcountll((x[i] & (~wb[i])));
    }
    return y;
}

/**
 * rate() - Calculates the adaptation rate
 * - `expected` The expected target value
 * - `given` The given return value
 *
 * Returns the calculated rate in [-1, 1].
 */
static double rate(int expected, int given, int total) {
    return (double)(expected - given) / (total);
}

/**
 * ## entropy() - calculates the new weights bit array
 * - `w` The weights bit array
 * - `r` The rate to adapt the bit array `w`
 *
 * Returns the adapted bit array.
 */
static unsigned long long entropy(unsigned long long w, double r) {
    for (unsigned i = 0; i < (random() % (long)(NBITS(w) * r)); i++) {
        unsigned pos = random() % NBITS(w);
        w = (w & ~(1ULL << pos)) | (((unsigned long long)r > 0) << pos);
    }
    return w;
}
