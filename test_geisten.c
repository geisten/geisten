//
// Created by germar on 31.07.21.
//
#include <time.h>

#include "geisten.h"
#include "test.h"

TEST_INIT();

#define ARRAY_LENGTH(_arr) (sizeof((_arr)) / sizeof(((_arr)[0])))
#define BIT_ARRAY_LEN(_n, _bits) (((_n)-1 + (_bits)) / (_bits))
#define BIT_ARRAY_SIZE(_arr, _bits) (BIT_ARRAY_LEN(ARRAY_LENGTH(_arr), (_bits)))

/**
 * ## binarize() - Binarize 8 bit fix point elements of array `x`
 * - `size` The length of the arrwy `x`
 * - `x` The fix point array
 * - `threshold` The conversion threshold
 * - `result` The bit result
 *
 * Binarize all elements of array `x` and writes the result to the bit array
 * `result`. The conversion is as follows:
 *
 * ```
 * if x[i] > threshold then set bit=1 else set bit=0
 * ```
 */
static void binarize_i8(
    uint32_t size, const int8_t x[size], const uint8_t threshold[size],
    unsigned long long result[(size / NBITS(unsigned long long)) + 1]) {
    foreach_to(i, size) { binarize_at_pos(result, i, x, threshold[i]); }
}

#define BINARIZE(_input, _a, _words) \
    foreach (i, _input) { binarize_at_pos((_words), i, (_input), (_a)[i]); }

#define FORWARD(_words, _w, _output)                        \
    foreach_to(j, ARRAY_LENGTH(_output)) {                  \
        foreach (i, (_words)) {                             \
            (_output)[j] = linear((_w)[j][i], (_words)[i]); \
        }                                                   \
    }

#define ACTIVATE(_array, _func, _output) \
    foreach (i, (_array)) { (_output)[i] = (_func)((_array)[i]); }

// static int test_bit(long long A, int k) { return ((A & (1 << k)) != 0); }

static bool vec_is_equal(uint32_t n, const int a[n], const int b[n],
                         int epsilon) {
    foreach_to(i, n) {
        if (abs(a[i] - b[i]) >= epsilon) return false;
    }
    return true;
}

static void test_relu() {
    int result = relu(40);
    test(result == 40 && "relu function must return 40");
    result = relu(-340);
    test(result == 0 && "relu function must return 0");

    int x[]          = {5, -99, 0, 100, 1000, -9999999, 2147483647};
    int y_expected[] = {5, 0, 0, 100, 1000, 0, 2147483647};
    int y[ARRAY_LENGTH(x)];
    ACTIVATE(x, relu, y);
    test(vec_is_equal(ARRAY_LENGTH(y), y, y_expected, 1) &&
         "compute relu function correct on array elements");
}

static void test_binarization_det() {
    size_t weights_demo[65];
    weights_demo[0]  = 1;
    int8_t weights[] = {-5, 127, -128, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0,  0,   0,    0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0,  0,   0,    0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0,  0,   0,    0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    unsigned long long weights_b[1];
    binarize_i8(ARRAY_LENGTH(weights), weights, 1, weights_b);
    test(weights_b[0] == 34 &&
         "Positions in bit array must binarized as expected");
    test(BIT_ARRAY_SIZE(weights_demo, sizeof(weights_b[0]) * CHAR_BIT) == 2 &&
         "Array size must be 2");
}

static void test_forward() {
    int8_t input[]               = {5, -2, 0, 3, -1};
    unsigned long long wb[][(ARRAY_LENGTH(input) / NBITS(unsigned long long) +
                             1)] = {{19}, {28}, {31}, {29}};
    const uint32_t OUTPUT_SIZE   = ARRAY_LENGTH(wb);
    int8_t alpha[]               = {2, 2, 2, 2, 2};
    int y[OUTPUT_SIZE];
    unsigned long long
        input_bits[(ARRAY_LENGTH(input) / NBITS(unsigned long long) + 1)] = {0};

    BINARIZE(input, alpha, input_bits);
    FORWARD(input_bits, wb, y);
    ACTIVATE(y, relu, y);

    int y_expected[] = {0, 0, 2, 2};
    test(input_bits[0] == 9 && "convert input into binary form");
    test(vec_is_equal(OUTPUT_SIZE, y, y_expected, 1) &&
         "transform to output vector");
}

int main() {
    srandom(time(NULL));
    test_relu();
    test_binarization_det();
    test_forward();
    return TEST_RESULT;
}
