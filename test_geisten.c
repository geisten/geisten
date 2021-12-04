//
// Created by germar on 31.07.21.
//
#include "geisten.h"
#include "test.h"

TEST_INIT();

// static int test_bit(long long A, int k) { return ((A & (1 << k)) != 0); }

static bool vec_is_equal(uint32_t n, const int a[n], const int b[n],
                         int epsilon) {
    for (uint32_t i = 0; i < n; i++) {
        if (abs(a[i] - b[i]) >= epsilon) return false;
    }
    return true;
}

static bool vec_is_equal_i8(uint32_t n, const int8_t a[n], const int8_t b[n],
                            int epsilon) {
    for (uint32_t i = 0; i < n; i++) {
        if (abs(a[i] - b[i]) >= epsilon) return false;
    }
    return true;
}

static void vec_write_i8(FILE *fp, uint32_t size, const int8_t arr[static size],
                         const char str[]) {
    fprintf(fp, "%s: [", str);
    for (uint64_t i = 0; i < size; i++) {
        fprintf(fp, "%d, ", arr[i]);
    }
    fprintf(fp, "]\n");
}

static double delta(uint32_t n, const int8_t a[n], const int8_t b[n],
                    int8_t error[n]) {
    double result = 0.0;
    for (uint32_t i = 0; i < n; i++) {
        error[i] = (int8_t)(b[i] - a[i]);
        result += pow(error[i], 2);
    }
    return result / n;
}

static void vec_write(FILE *fp, uint32_t size, const int arr[static size],
                      const char str[]) {
    fprintf(fp, "%s: [", str);
    for (uint64_t i = 0; i < size; i++) {
        fprintf(fp, "%d, ", arr[i]);
    }
    fprintf(fp, "]\n");
}

static void test_binarization_det() {
    size_t weights_demo[65];
    weights_demo[0]  = 1;
    int8_t weights[] = {-5, 127, -128, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0,  0,   0,    0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0,  0,   0,    0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0,  0,   0,    0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    unsigned long long weights_b[1];
    binarize(ARRAY_LENGTH(weights), weights, 0, weights_b);
    test(weights_b[0] == 34 && "Positions in bit array must be set correctly");
    test(BIT_ARRAY_SIZE(weights_demo, sizeof(weights_b[0]) * CHAR_BIT) == 2 &&
         "Array size must be 2");
}

static void test_forward() {
    int8_t input[]               = {5, -2, 0, 3, -1};
    unsigned long long wb[][(ARRAY_LENGTH(input) / NBITS(unsigned long long) +
                             1)] = {{19}, {28}, {31}, {29}};
    const uint32_t OUTPUT_SIZE   = ARRAY_LENGTH(wb);
    int8_t alpha                 = 2;
    int y[OUTPUT_SIZE];
    unsigned long long
        input_bits[(ARRAY_LENGTH(input) / NBITS(unsigned long long) + 1)] = {0};
    binarize(ARRAY_LENGTH(input), input, alpha, input_bits);
    for (uint32_t j = 0; j < OUTPUT_SIZE; j++) {
        y[j] = forward(ARRAY_LENGTH(input_bits), wb[j], input_bits);
    }
    int y_expected[] = {0, 0, 2, 2};
    test(input_bits[0] == 9 && "convert input into binary form");
    test(vec_is_equal(OUTPUT_SIZE, y, y_expected, 1) &&
         "transform to output vector");
}

static void test_entropy() {
    unsigned long long w    = 0;
    unsigned long long wres = entropy(w, 1.0);
    unsigned count_w    = builtin_popcountll(w);
    unsigned count_wres = builtin_popcountll(wres);
    printf("new w count: %u\n", count_wres);
    test(count_wres >= count_w && "test if entropy is increasing");

    wres = entropy(w, -1.0);
    count_wres = builtin_popcountll(wres);
    printf("new w count: %u\n", count_wres);
    test(count_wres == 0 &&
         "test if empty bit field is still empty when rate < 0 (-1.0)");

    w       = 99484776326;
    count_w = builtin_popcountll(w);
    wres    = entropy(w, -1.0);
    count_wres = builtin_popcountll(wres);
    printf("old w count: %u, new w count: %u\n", count_w, count_wres);
    test(count_wres <= count_w &&
         "test if number of active bits decrease if rate < 0 (-1.0)");
}

int main() {
    srandom(time(NULL));
    test_binarization_det();
    test_forward();
    test_entropy();
    return TEST_RESULT;
}
