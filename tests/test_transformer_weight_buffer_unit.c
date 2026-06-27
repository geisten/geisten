/*
 * test_transformer_weight_buffer_unit - weight loader host-buffer policy.
 *
 * Verifies the architecture-side load helper used by GGUF tensor loading:
 * CPU backends keep the mmap/arena alias fast path, while device-only
 * backends can reject aliasing and still receive identical bytes by upload.
 */
#include "test_helpers.h"

#define GEIST_INTERNAL_ARCH_LAYER
#include "src/archs/transformer/weight_load/internal.h"

#include <geist.h>
#include <geist_backend.h>

#include <stdio.h>
#include <string.h>

static int check(bool cond, const char *what) {
    if (!cond) {
        fprintf(stderr, "FAIL: %s\n", what);
        return 1;
    }
    return 0;
}

static int check_bytes_equal(const uint8_t *a, const uint8_t *b,
                             size_t n, const char *what) {
    if (memcmp(a, b, n) == 0) {
        return 0;
    }
    fprintf(stderr, "FAIL: %s\n", what);
    return 1;
}

static int exercise_policy(struct geist_backend *be, bool expect_alias) {
    int fails = 0;
    uint8_t src[96];
    uint8_t dst[96];
    for (size_t i = 0; i < sizeof src; i++) {
        src[i] = (uint8_t) (i * 13u + 7u);
        dst[i] = 0;
    }

    struct geist_buffer *buf = nullptr;
    enum geist_status s = weight_load_buffer_from_host(
        be, src, sizeof src, GEIST_BUFFER_WEIGHT, true, &buf);
    fails += check(s == GEIST_OK, "prefer-alias helper returns OK");
    fails += check(buf != nullptr, "prefer-alias helper returns buffer");
    if (buf == nullptr) {
        return fails;
    }

    void *mapped = be->desc->vtbl->buffer_map(buf);
    if (expect_alias) {
        fails += check(mapped == src, "CPU helper preserves host alias");
        if (mapped != nullptr) {
            fails += check_bytes_equal(mapped, src, sizeof src,
                                       "mapped alias bytes match");
        }
    } else {
        fails += check(mapped == nullptr, "device helper avoids host map");
    }
    be->desc->vtbl->buffer_unmap(buf);

    s = be->desc->vtbl->buffer_download(sizeof dst, dst, buf);
    fails += check(s == GEIST_OK, "helper buffer downloads");
    fails += check_bytes_equal(dst, src, sizeof src,
                               "helper upload/fallback preserves bytes");
    be->desc->vtbl->buffer_destroy(be, buf);

    buf = nullptr;
    s = weight_load_buffer_from_host(
        be, src, sizeof src, GEIST_BUFFER_WEIGHT, false, &buf);
    fails += check(s == GEIST_OK, "force-upload helper returns OK");
    fails += check(buf != nullptr, "force-upload helper returns buffer");
    if (buf != nullptr) {
        memset(dst, 0, sizeof dst);
        s = be->desc->vtbl->buffer_download(sizeof dst, dst, buf);
        fails += check(s == GEIST_OK, "force-upload buffer downloads");
        fails += check_bytes_equal(dst, src, sizeof src,
                                   "force-upload preserves bytes");
        be->desc->vtbl->buffer_destroy(be, buf);
    }

    return fails;
}

int main(void) {
    int fails = 0;

    struct geist_backend *cpu = nullptr;
    enum geist_status s = geist_backend_create("cpu_scalar", nullptr, nullptr, &cpu);
    fails += check(s == GEIST_OK, "cpu_scalar create OK");
    fails += check(cpu != nullptr, "cpu_scalar handle non-null");
    if (cpu != nullptr) {
        fails += exercise_policy(cpu, true);
        geist_backend_destroy(cpu);
    }

    struct geist_backend *vk = nullptr;
    s = geist_backend_create("vulkan", nullptr, nullptr, &vk);
#if defined(GEIST_BACKEND_VULKAN) && GEIST_BACKEND_VULKAN
    if (s == GEIST_OK) {
        fails += exercise_policy(vk, false);
        geist_backend_destroy(vk);
    } else {
        fails += check(s == GEIST_E_BACKEND,
                       "vulkan unavailable cleanly for weight buffer policy");
    }
#else
    fails += check(s == GEIST_E_NOT_FOUND, "vulkan not compiled in");
#endif

    if (fails == 0) {
        printf("PASS: transformer weight buffer alias/upload policy\n");
        return GEIST_TEST_PASS;
    }
    fprintf(stderr, "FAILED: %d check(s)\n", fails);
    return GEIST_TEST_FAIL;
}
