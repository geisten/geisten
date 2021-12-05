/**
 * # geisten - The library for creating efficient binary neural networks
 *
 *     author: Germar Schlegel
 *     date: 01.09.2021
 *
 * This c header only library provides an efficient neural network architecture
 * in order to build very small, low latency models that can be easily matched to the
 * design requirements for mobile and embedded vision applications.
 *
 * The 1-bit convolutional neural network (1-bit CNN, also known as binary neu-
 * ral network), of which both weights and activations are binary, has
 * been recognized as one of the most promising neural network compression
 * methods for deploying models onto the resource-limited devices. Moreover,
 * with its pure logical computation (i.e., XNOR operations between binary weights
 * and binary activations), 1-bit CNN is both highly energy-efficient for
 * embedded devices, and possesses the potential of being directly deployed on
 * next generation memristor-based hardware.
 */

#pragma once

#include <stdint.h>
#include <stdlib.h>
#include <limits.h>

/**
 * ## NBITS() - Returns the number of bits of the type ´type´.
 * - ´type´ The c type
 * Macro to calculate the number of bits.
 */
#define NBITS(_type) (sizeof(_type) * CHAR_BIT)

/*
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
 * ## relu() - ReLU function
 * - `x` The function variable
 */
static int relu(int x) { return x * (x > 0); }

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
