/*
 * test_backend_vulkan_registry_unit - build-gated Vulkan backend skeleton.
 *
 * Verifies the first Vulkan increment:
 *   - CPU-only builds do not expose a "vulkan" backend.
 *   - Vulkan-selected builds can create/destroy the skeleton backend when a
 *     loader is present, or report clean runtime unavailability otherwise.
 *   - A created backend advertises the first native Level-2 compute op.
 */
#include "test_helpers.h"

#include <geist.h>
#include <geist_util.h>
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

#if defined(GEIST_BACKEND_VULKAN) && GEIST_BACKEND_VULKAN
static int check_bytes_equal(const uint8_t *a, const uint8_t *b,
                             size_t n, const char *what) {
    if (memcmp(a, b, n) == 0) {
        return 0;
    }
    fprintf(stderr, "FAIL: %s\n", what);
    for (size_t i = 0; i < n; i++) {
        if (a[i] != b[i]) {
            fprintf(stderr, "  first mismatch at %zu: got %u expected %u\n",
                    i, (unsigned) a[i], (unsigned) b[i]);
            break;
        }
    }
    return 1;
}

static int exercise_buffers(struct geist_backend *be) {
    int fails = 0;
    const size_t n = 64;
    uint8_t src[n];
    uint8_t dst[n];
    for (size_t i = 0; i < n; i++) {
        src[i] = (uint8_t) (i * 5 + 11);
        dst[i] = 0;
    }

    struct geist_buffer *device_buf = nullptr;
    enum geist_status s = be->desc->vtbl->buffer_create(
        be, n, GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_DEVICE, &device_buf);
    fails += check(s == GEIST_OK, "device buffer_create OK");
    fails += check(device_buf != nullptr, "device buffer handle non-null");
    if (device_buf != nullptr) {
        s = be->desc->vtbl->buffer_upload(device_buf, n, src);
        fails += check(s == GEIST_OK, "device buffer_upload OK");
        s = be->desc->vtbl->buffer_download(n, dst, device_buf);
        fails += check(s == GEIST_OK, "device buffer_download OK");
        fails += check_bytes_equal(dst, src, n, "device download matches upload");
        fails += check(be->desc->vtbl->buffer_map(device_buf) == nullptr,
                       "device buffer_map returns null");

        struct geist_buffer *copy_buf = nullptr;
        s = be->desc->vtbl->buffer_create(
            be, n, GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_DEVICE, &copy_buf);
        fails += check(s == GEIST_OK, "device copy target buffer_create OK");
        fails += check(copy_buf != nullptr, "device copy target handle non-null");
        if (copy_buf != nullptr) {
            memset(dst, 0, n);
            s = be->desc->vtbl->buffer_upload(copy_buf, n, dst);
            fails += check(s == GEIST_OK, "device copy target zero upload OK");
            s = be->desc->vtbl->buffer_copy(copy_buf, 12, device_buf, 5, 21);
            fails += check(s == GEIST_OK, "device buffer_copy offset range OK");
            s = be->desc->vtbl->buffer_download(n, dst, copy_buf);
            fails += check(s == GEIST_OK, "device copy target download OK");
            fails += check(memcmp(dst + 12, src + 5, 21) == 0,
                           "device buffer_copy copied requested bytes");
            fails += check(dst[11] == 0 && dst[33] == 0,
                           "device buffer_copy leaves surrounding bytes untouched");
            s = be->desc->vtbl->buffer_copy(copy_buf, n - 4, device_buf, 0, 8);
            fails += check(s == GEIST_E_INVALID_ARG,
                           "device buffer_copy rejects out-of-range copy");
            be->desc->vtbl->buffer_destroy(be, copy_buf);
        }
        be->desc->vtbl->buffer_unmap(device_buf);
        be->desc->vtbl->buffer_destroy(be, device_buf);
    }

    struct geist_buffer *staging_buf = nullptr;
    s = be->desc->vtbl->buffer_create(
        be, n, GEIST_BUFFER_STAGING, GEIST_MEMORY_HOST_VISIBLE, &staging_buf);
    fails += check(s == GEIST_OK, "staging buffer_create OK");
    fails += check(staging_buf != nullptr, "staging buffer handle non-null");
    if (staging_buf != nullptr) {
        s = be->desc->vtbl->buffer_upload(staging_buf, n, src);
        fails += check(s == GEIST_OK, "staging buffer_upload OK");
        void *host = be->desc->vtbl->buffer_map(staging_buf);
        fails += check(host != nullptr, "staging buffer_map returns host pointer");
        if (host != nullptr) {
            fails += check_bytes_equal((const uint8_t *) host, src, n,
                                       "mapped staging bytes match upload");
            ((uint8_t *) host)[7] ^= 0x5au;
        }
        be->desc->vtbl->buffer_unmap(staging_buf);
        uint8_t expected[n];
        memcpy(expected, src, n);
        expected[7] ^= 0x5au;
        s = be->desc->vtbl->buffer_download(n, dst, staging_buf);
        fails += check(s == GEIST_OK, "staging buffer_download OK");
        fails += check_bytes_equal(dst, expected, n,
                                   "staging download reflects mapped write");
        be->desc->vtbl->buffer_destroy(be, staging_buf);
    }

    struct geist_buffer *bad = (struct geist_buffer *) 0x1;
    s = be->desc->vtbl->buffer_create(
        be, 0, GEIST_BUFFER_STAGING, GEIST_MEMORY_HOST_VISIBLE, &bad);
    fails += check(s == GEIST_E_INVALID_ARG, "zero-byte buffer rejected");
    fails += check(bad == nullptr, "zero-byte buffer clears out param");

    return fails;
}
#endif

int main(void) {
    int fails = 0;

    struct geist_backend *be = nullptr;
    enum geist_status s = geist_backend_create("vulkan", nullptr, nullptr, &be);

#if defined(GEIST_BACKEND_VULKAN) && GEIST_BACKEND_VULKAN
    if (s == GEIST_OK) {
        fails += check(be != nullptr, "vulkan backend handle is non-null");
        if (be == nullptr) {
            return GEIST_TEST_FAIL;
        }

        fails += check(strcmp(geist_backend_name(be), "vulkan") == 0,
                       "backend name is vulkan");
        fails += check(geist_backend_errcode(be) == GEIST_OK,
                       "errcode is OK after create");
        fails += check(be->desc->vtbl->query_accel_caps != nullptr,
                       "vulkan exposes accel caps query");
        if (be->desc->vtbl->query_accel_caps != nullptr) {
            struct geist_backend_accel_caps small = {0};
            enum geist_status qs = be->desc->vtbl->query_accel_caps(be, &small);
            fails += check(qs == GEIST_E_INVALID_ARG,
                           "accel caps rejects undersized struct");

            struct geist_backend_accel_caps caps = {
                .struct_size = sizeof(caps),
            };
            qs = be->desc->vtbl->query_accel_caps(be, &caps);
            fails += check(qs == GEIST_OK, "accel caps query OK");
            fails += check(caps.compute_queue, "accel caps has compute queue");
            fails += check(caps.device_resident_buffers,
                           "accel caps has device-resident buffers");
            fails += check(caps.pipeline_cache, "accel caps has pipeline cache");
            fails += check(caps.device_name[0] != '\0',
                           "accel caps device name non-empty");
        }

        struct geist_op_support_query linear_q = {
            .op = GEIST_OP_LINEAR,
            .input_count = 2,
            .inputs = {
                {.dtype = GEIST_DTYPE_F32, .layout = GEIST_LAYOUT_DENSE},
                {.dtype = GEIST_DTYPE_F32, .layout = GEIST_LAYOUT_DENSE},
            },
            .output_count = 1,
            .outputs = {{.dtype = GEIST_DTYPE_F32, .layout = GEIST_LAYOUT_DENSE}},
        };
        fails += check(geist_backend_supports_op(be, &linear_q) ==
                           GEIST_SUPPORT_NATIVE,
                       "vulkan advertises native F32 dense linear matvec");
        fails += check(be->desc->vtbl->matvec_f32_dense != nullptr,
                       "vulkan exposes F32 dense matvec entrypoint");
        linear_q.inputs[1] = (struct geist_tensor_format){
            .dtype = GEIST_DTYPE_Q4_K,
            .layout = GEIST_LAYOUT_BLOCK_QUANTIZED,
        };
        fails += check(geist_backend_supports_op(be, &linear_q) ==
                           GEIST_SUPPORT_NATIVE,
                       "vulkan advertises native Q4_K decode linear matvec");
        fails += check(be->desc->vtbl->matvec_q4k != nullptr,
                       "vulkan exposes Q4_K matvec entrypoint");
        linear_q.inputs[1] = (struct geist_tensor_format){
            .dtype = GEIST_DTYPE_Q6_K,
            .layout = GEIST_LAYOUT_BLOCK_QUANTIZED,
        };
        fails += check(geist_backend_supports_op(be, &linear_q) ==
                           GEIST_SUPPORT_NATIVE,
                       "vulkan advertises native Q6_K decode linear matvec");
        fails += check(be->desc->vtbl->matvec_q6k != nullptr,
                       "vulkan exposes Q6_K matvec entrypoint");
        fails += check(be->desc->vtbl->argmax_f32 != nullptr,
                       "vulkan exposes F32 greedy argmax entrypoint");
        fails += check(be->desc->vtbl->rope_apply != nullptr,
                       "vulkan exposes F32 RoPE entrypoint");
        fails += check(be->desc->vtbl->attention != nullptr,
                       "vulkan exposes F32 attention entrypoint");
        fails += check(be->desc->vtbl->gelu_tanh_mul != nullptr,
                       "vulkan exposes F32 GEGLU middle entrypoint");
        fails += check(be->desc->vtbl->attention_block != nullptr,
                       "vulkan exposes attention block entrypoint");
        fails += check(be->desc->vtbl->attention_query_block != nullptr,
                       "vulkan exposes attention query block entrypoint");
        fails += check(be->desc->vtbl->greedy_head != nullptr,
                       "vulkan exposes greedy head entrypoint");
        fails += check(be->desc->vtbl->command_sequence_begin != nullptr,
                       "vulkan exposes command sequence begin");
        fails += check(be->desc->vtbl->command_sequence_end != nullptr,
                       "vulkan exposes command sequence end");

        struct geist_op_support_query attention_q = {
            .op = GEIST_OP_ATTENTION,
            .input_count = 3,
            .inputs = {
                {.dtype = GEIST_DTYPE_F32, .layout = GEIST_LAYOUT_DENSE},
                {.dtype = GEIST_DTYPE_F32, .layout = GEIST_LAYOUT_DENSE},
                {.dtype = GEIST_DTYPE_F32, .layout = GEIST_LAYOUT_DENSE},
            },
            .output_count = 1,
            .outputs = {{.dtype = GEIST_DTYPE_F32, .layout = GEIST_LAYOUT_DENSE}},
        };
        fails += check(geist_backend_supports_op(be, &attention_q) ==
                           GEIST_SUPPORT_NATIVE,
                       "vulkan advertises native F32 attention");

        struct geist_op_support_query add_q = {
            .op = GEIST_OP_RESIDUAL_ADD,
            .input_count = 2,
            .inputs = {
                {.dtype = GEIST_DTYPE_F32, .layout = GEIST_LAYOUT_DENSE},
                {.dtype = GEIST_DTYPE_F32, .layout = GEIST_LAYOUT_DENSE},
            },
            .output_count = 1,
            .outputs = {{.dtype = GEIST_DTYPE_F32, .layout = GEIST_LAYOUT_DENSE}},
        };
        fails += check(geist_backend_supports_op(be, &add_q) ==
                           GEIST_SUPPORT_NATIVE,
                       "vulkan advertises native F32 residual add");

        struct geist_op_support_query rmsnorm_q = {
            .op = GEIST_OP_RMSNORM,
            .input_count = 2,
            .inputs = {
                {.dtype = GEIST_DTYPE_F32, .layout = GEIST_LAYOUT_DENSE},
                {.dtype = GEIST_DTYPE_F32, .layout = GEIST_LAYOUT_DENSE},
            },
            .output_count = 1,
            .outputs = {{.dtype = GEIST_DTYPE_F32, .layout = GEIST_LAYOUT_DENSE}},
        };
        fails += check(geist_backend_supports_op(be, &rmsnorm_q) ==
                           GEIST_SUPPORT_NATIVE,
                       "vulkan advertises native F32 rmsnorm");

        fails += exercise_buffers(be);

        geist_backend_destroy(be);
    } else {
        fails += check(s == GEIST_E_BACKEND,
                       "vulkan backend reports clean runtime unavailability");
        fails += check(be == nullptr, "failed vulkan create leaves handle null");
        fails += check(strstr(geist_last_create_error(), "vulkan") != nullptr,
                       "create-time error mentions vulkan");
    }
#else
    fails += check(s == GEIST_E_NOT_FOUND, "vulkan backend is not compiled in");
    fails += check(be == nullptr, "failed create leaves handle null");
    fails += check(strstr(geist_last_create_error(), "vulkan") != nullptr,
                   "create-time error mentions vulkan");
#endif

    if (fails == 0) {
        printf("PASS: vulkan backend registry gating\n");
        return GEIST_TEST_PASS;
    }
    fprintf(stderr, "FAILED: %d check(s)\n", fails);
    return GEIST_TEST_FAIL;
}
