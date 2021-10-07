#pragma once

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

extern struct test_status {
    int tests_failed;
    int tests;
} test_status;

#define TEST_INIT() struct test_status test_status = {0, 0}

#define test(_expr)                                                            \
    do {                                                                       \
        if (!(_expr)) {                                                        \
            printf("FAILED (Line %d): %s\n", __LINE__, #_expr);                \
            ++test_status.tests_failed;                                        \
        } else {                                                               \
            printf("    ok: %s >> %s\n", __func__, #_expr);                    \
        }                                                                      \
        ++test_status.tests;                                                   \
    } while (0)


static inline int test_result() {
    if (test_status.tests_failed == 0) {
        printf("All %d tests ok\n", test_status.tests);
    } else {
        printf("Result: %d / %d tests failed \n", test_status.tests_failed,
               test_status.tests);
    }
    return test_status.tests_failed ? EXIT_FAILURE : EXIT_SUCCESS;
}

#define TEST_RESULT test_result()
