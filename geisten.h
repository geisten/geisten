/**
 * # geisten - The library for creating efficient binary neural networks
 *
 *
 *        ╚══╗ ║ ╔════╝
 *           ╠═╩═╩╗
 *           ║ ▒▒ ║
 *           ╠═╦═╦╝
 *        ╔══╝ ╚═╚════╗
 *
 *     author: Germar Schlegel
 *     date: 01.09.2021
 *
 * This c header only library provides an efficient binary, neural network architecture in order to build very small, low
 * latency models that can be easily matched to the design requirements for mobile and embedded vision applications.
 *
 * The 1-bit convolutional neural network (1-bit CNN, also known as binary neural network), of which both weights and
 * activations are binary, has been recognized as one of the most promising neural network compression methods for
 * deploying models onto resource-limited devices. Moreover, with its pure logical computation (i.e., XNOR operations
 * between binary weights and binary activations), 1-bit CNN is both highly energy-efficient for embedded devices, and
 * possesses the potential of being directly deployed on next generation memristor-based hardware.
 */

#pragma once

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

/**
 * ## Functions
 */

/**
 * ### NBITS() - Returns the number of bits of the type ´type´.
 * - `type` The c type
 *
 * Macro to calculate the number of bits.
 * A word is a fixed-size data element that is processed as a unit by the
 * processor's instruction set or hardware. The number of bits in a
 * word (the word size, word width, or word length) is an important
 * characteristic of a particular processor design or computer architecture.
 */
#define NBITS(_type) (sizeof(_type) * CHAR_BIT)

/**
 * ### WORDS_INDEX() - Returns the index of the words array at bit position i
 */
#define WORDS_INDEX(_w, _i) ((_i) / NBITS((_w)[0]))
#define WORDS_POS(_w, _i) ((_i) % NBITS((_w)[0]))

/**
 * ### foreach() - The foreach loop macro.
 * - `_a` The index of the current array element
 * - `_b` The array
 */
#define foreach(_a, _b) \
    for (size_t _a = 0; _a < (sizeof(_b) / sizeof((_b)[0])); _a++)

/**
 * ### foreach_to() - The foreach loop macro to iterate over range `[0, _b]`.
 * - `_a` The index of the current array element
 * - `_b` The upper boundary value (length(array))
 */
#define foreach_to(_a, _b) for (size_t _a = 0; _a < (_b); _a++)

/**
 * ### WORD_ONE() - standard word with first bit set.
 * - `_b` The word type
 *
 * Returns the standard value `1` of type 'b'
 */
#define WORD_ONE(_b)                                     \
    _Generic((_b), /* The 1 word type*/                  \
             unsigned int : 1U, unsigned long int : 1LU, \
             unsigned long long int : 1LLU, default : 1U)

/**
 * ### binarize() - Binarize the value `v` at position `i` and write the result in `w`.
 * - `_w` The binary word
 * - `_i` The position (index) of the bit within the word `_w`
 * - `_t` The conversion threshold
 * - `_v` The value to be binarized
 *
 * ```
 * if _v > _t then set bit=1 else set bit=0
 * ```
 */
#define binarize(_w, _i, _t, _v)                     \
    (((_v) > (_t)) ? ((_w) | (WORD_ONE(_w) << (_i))) \
                   : ((_w) & ~(WORD_ONE(_w) << (_i))))

/**
 * ### binarize_at_pos() - Binarize the value `v` at position `i` and write the result into array `w`.
 * - `_w` The word array
 * - `_i` The position (index) of the bit within the word `_w`
 * - `_t` The conversion threshold
 * - `_v` The value to be binarized
 *
 * Binarizes all elements of array `x` and writes the result to the bit array
 * `_w`. The conversion is as follows:
 *
 * ```
 * if _x[i] > _t then set bit=1 else set bit=0 in element (_w)[WORDS_INDEX((_w), (_i))] of array `_w`
 * ```
 */
#define binarize_at_pos(_w, _i, _x, _t)                                      \
    (_w)[WORDS_INDEX((_w), (_i))] =                                          \
        binarize((_w)[WORDS_INDEX((_w), (_i))], WORDS_POS((_w), (_i)), (_t), \
                 (_x)[(_i)]);

/*
 * The special operator __has_builtin (operand) is used to test whether
 * the symbol named by its operand is recognized as a built-in function by GCC
 * in the current language and conformance mode. The __has_builtin operator by
 * itself, without any operand or parentheses, acts as a predefined macro so
 * that support for it can be tested in portable code.
 */
#ifndef __builtin_popcountll
#define popcount_soft(_x)                      \
    ({                                         \
        unsigned c = 0;                        \
        for (; (_x) != 0; (_x) &= (_x)-1) c++; \
        c;                                     \
    })

static int popcountll_soft(unsigned long long x) { return popcount_soft(x); }

static int popcountl_soft(unsigned long x) { return popcount_soft(x); }

static int popcounti_soft(unsigned x) { return popcount_soft(x); }

#define popcountll popcountll_soft
#define popcountl popcountl_soft
#define popcounti popcounti_soft
#elif
#define popcountll _builtin_popcountll
#define popcountl _builtin_popcountl
#define popcounti _builtin_popcount
#endif

#define popcount(_x)                                     \
    _Generic((_x), /* Count the number of active bits */ \
             unsigned int                                \
             : popcounti, unsigned long int              \
             : popcountl, unsigned long long int         \
             : popcountll, default                       \
             : popcounti)((_x))

/**
 * ### relu() - ReLU function
 * - `x` The function variable
 */
static int relu(int x) { return x * (x > 0); }

/**
 * ### forward() - Linear forward transformation
 * - `w` The binary weights word
 * - `x` The activation binaries word
 *
 * ### Bitwise matrix multiplication
 *
 * `_x` and `_w` are both binary values thus  `_x ∗ _w` can be implemented with bitwise
 * operations. In each bit, the result of basic multiplication `_x × _w` is one of
 * three values {−1,0,1}.
 * [Details](https://arxiv.org/pdf/1909.11366.pdf)
 *
 * Return the sum of the element wise multiplication.
 */
#define forward(_w, _x) (popcount((_x) & (_w)) - popcount((_x) & ~(_w)))
