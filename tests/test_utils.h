/*
 * test_utils — shared helpers for test/bench drivers (C23).
 *
 * Header-only: include from test_*.c / bench_*.c. Each TU gets its own
 * inline copy; deliberately tiny so the duplication doesn't matter.
 */
#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

[[nodiscard]] static inline double now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return (double) tv.tv_sec * 1000.0 + (double) tv.tv_usec / 1000.0;
}

/* Read a binary file as int32_t array. Returns nullptr on any I/O error
 * (file missing, ftell fail, malloc fail, short read). On success
 * *n_out = number of int32_t elements. Caller frees with free(). */
[[nodiscard]] static inline int32_t* read_int32_bin(const char* path, size_t* n_out) {
    FILE* f = fopen(path, "rb");
    if (!f)
        return nullptr;
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return nullptr;
    }
    long sz = ftell(f);
    if (sz < 0 || sz % 4 != 0) {
        fclose(f);
        return nullptr;
    }
    rewind(f);
    int32_t* p = (int32_t*) malloc((size_t) sz);
    if (!p) {
        fclose(f);
        return nullptr;
    }
    size_t got = fread(p, 1, (size_t) sz, f);
    fclose(f);
    if (got != (size_t) sz) {
        free(p);
        return nullptr;
    }
    *n_out = (size_t) sz / sizeof(int32_t);
    return p;
}

static inline void write_bin(const char* path, const void* data, size_t n_bytes) {
    FILE* f = fopen(path, "wb");
    if (!f)
        return;
    if (fwrite(data, 1, n_bytes, f) != n_bytes)
        fprintf(stderr, "write_bin: short write to %s\n", path);
    fclose(f);
}

#endif
