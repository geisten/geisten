/*
 * test_transformer_kv_store_unit - transformer KV append backend policy.
 *
 * Verifies that the FP32 KV append path uses backend buffer_copy so Vulkan
 * device buffers do not need host mapping in the attention block.
 */
#include "test_helpers.h"

#define GEIST_INTERNAL_ARCH_LAYER
#include "src/archs/transformer/forward/internal.h"

#include <geist.h>
#include <geist_backend.h>

#include <math.h>
#include <stdio.h>
#include <string.h>

static int check(bool cond, const char *what) {
    if (!cond) {
        fprintf(stderr, "FAIL: %s\n", what);
        return 1;
    }
    return 0;
}

static int check_close(size_t n,
                       const float got[static n],
                       const float want[static n],
                       const char *what) {
    const ptrdiff_t bad = geist_fp32_close_array(want, got, n, 0.0f, 0.0f);
    if (bad < 0) {
        return 0;
    }
    fprintf(stderr, "FAIL %s: idx %td got %.7f expected %.7f\n",
            what, bad, (double) got[bad], (double) want[bad]);
    return 1;
}

static int exercise_fp32_append(struct geist_backend *be,
                                enum geist_memory_flags memory_hint) {
    int fails = 0;
    const struct geist_backend_vtbl *v = be->desc->vtbl;
    constexpr size_t seq = 2;
    constexpr size_t q_position = 2;
    constexpr size_t kv_out = 4;
    constexpr size_t cache_rows = 6;
    constexpr size_t scratch_n = seq * kv_out;
    constexpr size_t cache_n = cache_rows * kv_out;

    const float k_src[scratch_n] = {
        1.0f, 1.1f, 1.2f, 1.3f,
        2.0f, 2.1f, 2.2f, 2.3f,
    };
    const float v_src[scratch_n] = {
        -1.0f, -1.1f, -1.2f, -1.3f,
        -2.0f, -2.1f, -2.2f, -2.3f,
    };
    float k_cache_init[cache_n];
    float v_cache_init[cache_n];
    float k_expected[cache_n];
    float v_expected[cache_n];
    for (size_t i = 0; i < cache_n; i++) {
        k_cache_init[i] = 100.0f + (float) i;
        v_cache_init[i] = -100.0f - (float) i;
        k_expected[i] = k_cache_init[i];
        v_expected[i] = v_cache_init[i];
    }
    memcpy(k_expected + q_position * kv_out, k_src, sizeof(k_src));
    memcpy(v_expected + q_position * kv_out, v_src, sizeof(v_src));

    struct geist_buffer *scratch_k = nullptr;
    struct geist_buffer *scratch_v = nullptr;
    struct geist_buffer *k_cache = nullptr;
    struct geist_buffer *v_cache = nullptr;
    enum geist_status s = v->buffer_create(
        be, sizeof(k_src), GEIST_BUFFER_ACTIVATION, memory_hint, &scratch_k);
    if (s == GEIST_OK) {
        s = v->buffer_create(
            be, sizeof(v_src), GEIST_BUFFER_ACTIVATION, memory_hint, &scratch_v);
    }
    if (s == GEIST_OK) {
        s = v->buffer_create(
            be, sizeof(k_cache_init), GEIST_BUFFER_KV_CACHE, memory_hint, &k_cache);
    }
    if (s == GEIST_OK) {
        s = v->buffer_create(
            be, sizeof(v_cache_init), GEIST_BUFFER_KV_CACHE, memory_hint, &v_cache);
    }
    if (s == GEIST_OK) {
        s = v->buffer_upload(scratch_k, sizeof(k_src),
                             (const uint8_t *) k_src);
    }
    if (s == GEIST_OK) {
        s = v->buffer_upload(scratch_v, sizeof(v_src),
                             (const uint8_t *) v_src);
    }
    if (s == GEIST_OK) {
        s = v->buffer_upload(k_cache, sizeof(k_cache_init),
                             (const uint8_t *) k_cache_init);
    }
    if (s == GEIST_OK) {
        s = v->buffer_upload(v_cache, sizeof(v_cache_init),
                             (const uint8_t *) v_cache_init);
    }

    struct transformer_arch_session sess = {
        .scratch_k = scratch_k,
        .scratch_v = scratch_v,
    };
    struct transformer_arch_state st = {
        .backend = be,
        .sess = &sess,
        .n_kv_heads = 1,
    };
    struct transformer_layer_forward_ctx ctx = {
        .st = &st,
        .be = be,
        .v = v,
        .q_position = q_position,
        .seq = seq,
        .kv_int8_enabled = false,
        .kv_kivi_enabled = false,
        .hd = kv_out,
        .kv_out = kv_out,
        .k_cache_buf = k_cache,
        .v_cache_buf = v_cache,
    };
    if (s == GEIST_OK) {
        s = transformer_kv_store_append(&ctx);
    }
    fails += check(s == GEIST_OK, "FP32 KV append returns OK");

    float k_got[cache_n];
    float v_got[cache_n];
    if (s == GEIST_OK) {
        s = v->buffer_download(sizeof(k_got), (uint8_t *) k_got, k_cache);
        fails += check(s == GEIST_OK, "K cache downloads after append");
    }
    if (s == GEIST_OK) {
        s = v->buffer_download(sizeof(v_got), (uint8_t *) v_got, v_cache);
        fails += check(s == GEIST_OK, "V cache downloads after append");
    }
    if (s == GEIST_OK) {
        fails += check_close(cache_n, k_got, k_expected,
                             "K cache has appended rows only");
        fails += check_close(cache_n, v_got, v_expected,
                             "V cache has appended rows only");
    }

    if (scratch_k != nullptr) {
        v->buffer_destroy(be, scratch_k);
    }
    if (scratch_v != nullptr) {
        v->buffer_destroy(be, scratch_v);
    }
    if (k_cache != nullptr) {
        v->buffer_destroy(be, k_cache);
    }
    if (v_cache != nullptr) {
        v->buffer_destroy(be, v_cache);
    }
    return fails;
}

static int exercise_fp32_empty_append(struct geist_backend *be,
                                      enum geist_memory_flags memory_hint) {
    int fails = 0;
    const struct geist_backend_vtbl *v = be->desc->vtbl;
    constexpr size_t q_position = 1;
    constexpr size_t kv_out = 4;
    constexpr size_t cache_rows = 3;
    constexpr size_t cache_n = cache_rows * kv_out;

    float cache_init[cache_n];
    for (size_t i = 0; i < cache_n; i++) {
        cache_init[i] = 10.0f + (float) i;
    }

    struct geist_buffer *scratch_k = nullptr;
    struct geist_buffer *scratch_v = nullptr;
    struct geist_buffer *k_cache = nullptr;
    struct geist_buffer *v_cache = nullptr;
    enum geist_status s = v->buffer_create(
        be, 1, GEIST_BUFFER_ACTIVATION, memory_hint, &scratch_k);
    if (s == GEIST_OK) {
        s = v->buffer_create(
            be, 1, GEIST_BUFFER_ACTIVATION, memory_hint, &scratch_v);
    }
    if (s == GEIST_OK) {
        s = v->buffer_create(
            be, sizeof(cache_init), GEIST_BUFFER_KV_CACHE, memory_hint, &k_cache);
    }
    if (s == GEIST_OK) {
        s = v->buffer_create(
            be, sizeof(cache_init), GEIST_BUFFER_KV_CACHE, memory_hint, &v_cache);
    }
    if (s == GEIST_OK) {
        s = v->buffer_upload(k_cache, sizeof(cache_init),
                             (const uint8_t *) cache_init);
    }
    if (s == GEIST_OK) {
        s = v->buffer_upload(v_cache, sizeof(cache_init),
                             (const uint8_t *) cache_init);
    }

    struct transformer_arch_session sess = {
        .scratch_k = scratch_k,
        .scratch_v = scratch_v,
    };
    struct transformer_arch_state st = {
        .backend = be,
        .sess = &sess,
        .n_kv_heads = 1,
    };
    struct transformer_layer_forward_ctx ctx = {
        .st = &st,
        .be = be,
        .v = v,
        .q_position = q_position,
        .seq = 0,
        .kv_int8_enabled = false,
        .kv_kivi_enabled = false,
        .hd = kv_out,
        .kv_out = kv_out,
        .k_cache_buf = k_cache,
        .v_cache_buf = v_cache,
    };
    if (s == GEIST_OK) {
        s = transformer_kv_store_append(&ctx);
    }
    fails += check(s == GEIST_OK, "empty FP32 KV append returns OK");

    float k_got[cache_n];
    float v_got[cache_n];
    if (s == GEIST_OK) {
        s = v->buffer_download(sizeof(k_got), (uint8_t *) k_got, k_cache);
        fails += check(s == GEIST_OK, "K cache downloads after empty append");
    }
    if (s == GEIST_OK) {
        s = v->buffer_download(sizeof(v_got), (uint8_t *) v_got, v_cache);
        fails += check(s == GEIST_OK, "V cache downloads after empty append");
    }
    if (s == GEIST_OK) {
        fails += check_close(cache_n, k_got, cache_init,
                             "K cache unchanged for empty append");
        fails += check_close(cache_n, v_got, cache_init,
                             "V cache unchanged for empty append");
    }

    if (scratch_k != nullptr) {
        v->buffer_destroy(be, scratch_k);
    }
    if (scratch_v != nullptr) {
        v->buffer_destroy(be, scratch_v);
    }
    if (k_cache != nullptr) {
        v->buffer_destroy(be, k_cache);
    }
    if (v_cache != nullptr) {
        v->buffer_destroy(be, v_cache);
    }
    return fails;
}

int main(void) {
    int fails = 0;

    struct geist_backend *cpu = nullptr;
    enum geist_status s = geist_backend_create("cpu_scalar", nullptr, nullptr, &cpu);
    fails += check(s == GEIST_OK, "cpu_scalar create OK");
    if (cpu != nullptr) {
        fails += exercise_fp32_append(cpu, GEIST_MEMORY_AUTO);
        fails += exercise_fp32_empty_append(cpu, GEIST_MEMORY_AUTO);
        geist_backend_destroy(cpu);
    }

    struct geist_backend *vk = nullptr;
    s = geist_backend_create("vulkan", nullptr, nullptr, &vk);
#if defined(GEIST_BACKEND_VULKAN) && GEIST_BACKEND_VULKAN
    if (s == GEIST_OK) {
        fails += exercise_fp32_append(vk, GEIST_MEMORY_DEVICE);
        fails += exercise_fp32_empty_append(vk, GEIST_MEMORY_DEVICE);
        geist_backend_destroy(vk);
    } else {
        fails += check(s == GEIST_E_BACKEND,
                       "vulkan unavailable cleanly for KV append test");
    }
#else
    fails += check(s == GEIST_E_NOT_FOUND, "vulkan not compiled in");
#endif

    if (fails == 0) {
        printf("PASS: transformer FP32 KV append uses backend copy\n");
        return GEIST_TEST_PASS;
    }
    fprintf(stderr, "FAILED: %d check(s)\n", fails);
    return GEIST_TEST_FAIL;
}
