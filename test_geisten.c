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
    int8_t input[] = {5, -2, 0, 3, -1};
    unsigned long long
        wb[][(ARRAY_LENGTH(input) / BIT_SIZE(unsigned long long) + 1)] = {
            {19}, {28}, {31}, {29}};
    const uint32_t OUTPUT_SIZE = ARRAY_LENGTH(wb);
    int8_t alpha               = 2;
    int y[OUTPUT_SIZE];
    unsigned long long input_bits[(
        ARRAY_LENGTH(input) / BIT_SIZE(unsigned long long) + 1)] = {0};
    binarize(ARRAY_LENGTH(input), input, alpha, input_bits);
    for (uint32_t j = 0; j < OUTPUT_SIZE; j++) {
        y[j] = forward(ARRAY_LENGTH(input_bits), wb[j], input_bits);
    }
    int y_expected[] = {0, 0, 2, 2};
    test(input_bits[0] == 9 && "convert input into binary form");
    test(vec_is_equal(OUTPUT_SIZE, y, y_expected, 1) &&
         "transform to output vector");
}

static void test_rprelu() {
    int test_0 = rprelu(0, 1, 2, 3);
    test(test_0 == 1 && "rprelu function must handle x=0");
    test_0 = rprelu(0, 0, 2, 3);
    test(test_0 == 3 && "rprelu function must handle x == 0 with beta == 0");
    int test_minus1 = rprelu(-1, 2, 2, 3);
    test(test_minus1 == -3 && "rprelu function with x == -1 and gamma > 0");
    int test_1 = rprelu(3, 3, 2, 3);
    test(test_1 == 4 && "rprelu function with x == 3 and gamma == 2");
}

static void test_backward() {
    int8_t output[]      = {1, 0, -2, 2};
    int delta[5]         = {0};
    int delta_expected[] = {
        1, -3, -1, -1, 1,
    };
    unsigned long long
        wb[][(ARRAY_LENGTH(delta) / BIT_SIZE(unsigned long long) + 1)] = {
            {19}, {28}, {31}, {29}};  //wb[m][n]
    for (uint32_t j = 0; j < ARRAY_LENGTH(output); j++) {
        backward(ARRAY_LENGTH(delta), wb[j], output[j], delta);
    }
    test(vec_is_equal(ARRAY_LENGTH(delta), delta, delta_expected, 1) &&
         "calculate the backward delta vector");
}

static void test_update_weights() {
    int delta[]                         = {20, -5, 8, 3};
    int x[]                             = {13, 9, 127, 6, 3};
    unsigned long long wb[][1]          = {{9}, {17}, {21}, {29}};
    unsigned long long wb_expected[][1] = {{0}, {21}, {16}, {25}};
    /*
    Expected w (decimal)
        C1      C2      C3      C4
1	-157    168	    -1	    64
2	-283	-58	    -175    -130
3	-2643	532	    -913	-278
4	-17	    -73	    -151	85
5	-163	118	    79	    94*/
    update_weights(ARRAY_LENGTH(x), x, delta[0], 103, wb[0]);
    test(wb[0][0] == wb_expected[0][0] &&
         "calculate the updated weights first column");
    update_weights(ARRAY_LENGTH(x), x, delta[1], 103, wb[1]);
    test(wb[1][0] == wb_expected[1][0] &&
         "calculate the updated weights second column");
    update_weights(ARRAY_LENGTH(x), x, delta[2], 103, wb[2]);
    test(wb[2][0] == wb_expected[2][0] &&
         "calculate the updated weights 3rd column");
    update_weights(ARRAY_LENGTH(x), x, delta[3], 103, wb[3]);
    test(wb[3][0] == wb_expected[3][0] &&
         "calculate the updated weights 4th column");
}

/*
 * Test the backpropagation of a two layer network.
 * First Layer : 16x8
 * Second Layer: 8x6  
 */
static void test_train() {
    int input[16]                    = {0};
    unsigned long long wb_input[][1] = {{34},    {53247}, {53223}, {36839},
                                        {65535}, {4095},  {4071},  {20071}};
    int hidden[]                = {1, 456, 0, -2345, 3456, -55445, -775, 443};
    int hidden_delta_expected[] = {-125, 125, 135, 123, -79, -59, 123, -71};
    int hidden_delta_derived_expected[]    = {23, 1, 1, 1, 23, 1, 1, 23};
    int hidden_delta[ARRAY_LENGTH(hidden)] = {0};
    int hidden_delta_derived[ARRAY_LENGTH(hidden)] = {0};
    unsigned long long wb_hidden[][1] = {{34}, {17}, {78}, {206}, {254}, {5}};
    unsigned long long wb_hidden_expected[][1] = {{104}, {147}, {108},
                                                  {150}, {108}, {109}};
    int8_t target[]                            = {5, 0, 127, -128, -5, 8};
    int8_t output[]                            = {4, 9, 30, -123, -34, 2};
    int8_t error[ARRAY_LENGTH(target)];
    int8_t d_expected[] = {1, -9, 97, -5, 29, 6};

    delta(ARRAY_LENGTH(target), output, target, error);
    test(vec_is_equal_i8(ARRAY_LENGTH(error), error, d_expected, 1) &&
         "calculate the delta between target and output");

    //calculate the error of the hidden layer
    for (uint32_t j = 0; j < ARRAY_LENGTH(error); j++) {
        backward(ARRAY_LENGTH(hidden), wb_hidden[j], error[j], hidden_delta);
    }
    test(vec_is_equal(ARRAY_LENGTH(hidden), hidden_delta, hidden_delta_expected,
                      1) &&
         "calculate the hidden delta vector");
    for (uint32_t i = 0; i < ARRAY_LENGTH(hidden_delta); i++) {
        hidden_delta_derived[i] = rprelu_derived(hidden_delta[i], 23, -60);
    }
    test(vec_is_equal(ARRAY_LENGTH(hidden), hidden_delta_derived,
                      hidden_delta_derived_expected, 1) &&
         "calculate the derived RPReLU vector");
    //update the weights of the 2nd layer
    for (uint32_t j = 0; j < ARRAY_LENGTH(error); j++) {
        update_weights(ARRAY_LENGTH(hidden), hidden, error[j], 103,
                       wb_hidden[j]);
        test(wb_hidden[j][0] == wb_hidden_expected[j][0] &&
             "calculate the updated weight matrix");
    }

    //update the weights of the first layer
    for (uint32_t j = 0; j < ARRAY_LENGTH(hidden); j++) {
        update_weights(ARRAY_LENGTH(input), input, hidden_delta[j], 103,
                       wb_input[j]);
    }
}

int main() {
    // srandom(time(NULL));
    test_binarization_det();
    test_rprelu();
    test_forward();
    test_backward();
    test_update_weights();
    test_train();
    return TEST_RESULT;
}
