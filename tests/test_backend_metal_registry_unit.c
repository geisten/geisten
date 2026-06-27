/*
 * test_backend_metal_registry_unit - build-gated Metal backend skeleton.
 *
 * Verifies the first native-Apple backend increment:
 *   - CPU-only builds do not expose "metal".
 *   - Metal-selected builds can create/destroy the backend when Metal is
 *     present, or report clean runtime unavailability otherwise.
 *   - A created backend exposes accelerator caps and the narrow native ops
 *     used by the Metal transformer bring-up path.
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

int main(void) {
    int fails = 0;

    struct geist_backend *be = nullptr;
    enum geist_status s = geist_backend_create("metal", nullptr, nullptr, &be);

#if defined(GEIST_BACKEND_METAL) && GEIST_BACKEND_METAL
    if (s == GEIST_E_UNSUPPORTED) {
        printf("SKIP: metal runtime unavailable: %s\n",
               geist_last_create_error());
        return GEIST_TEST_SKIP;
    }
    fails += check(s == GEIST_OK, "metal backend create OK");
    fails += check(be != nullptr, "metal backend handle is non-null");
    if (be == nullptr) {
        return GEIST_TEST_FAIL;
    }

    fails += check(strcmp(geist_backend_name(be), "metal") == 0,
                   "backend name is metal");
    fails += check(be->desc->vtbl->query_accel_caps != nullptr,
                   "metal exposes accel caps query");
    fails += check(be->desc->vtbl->supports_op != nullptr,
                   "metal exposes supports_op hook");
    fails += check(be->desc->vtbl->buffer_create != nullptr,
                   "metal exposes buffer_create");
    fails += check(be->desc->vtbl->buffer_destroy != nullptr,
                   "metal exposes buffer_destroy");
    fails += check(be->desc->vtbl->buffer_upload != nullptr,
                   "metal exposes buffer_upload");
    fails += check(be->desc->vtbl->buffer_download != nullptr,
                   "metal exposes buffer_download");
    fails += check(be->desc->vtbl->buffer_copy != nullptr,
                   "metal exposes buffer_copy");
    fails += check(be->desc->vtbl->buffer_map != nullptr,
                   "metal exposes buffer_map");
    fails += check(be->desc->vtbl->buffer_unmap != nullptr,
                   "metal exposes buffer_unmap");
    fails += check(be->desc->vtbl->buffer_create_aliased == nullptr,
                   "metal does not alias external host storage");
    fails += check(be->desc->vtbl->prepare_weight_layout != nullptr,
                   "metal exposes weight layout preparation");
    fails += check(be->desc->vtbl->prepare_weight_layout_from_host != nullptr,
                   "metal exposes host weight layout preparation");
    fails += check(be->desc->vtbl->embedding_lookup != nullptr,
                   "metal exposes embedding lookup");
    fails += check(be->desc->vtbl->embedding_lookup_scaled != nullptr,
                   "metal exposes scaled embedding lookup");
    fails += check(be->desc->vtbl->matvec_q4k != nullptr,
                   "metal exposes Q4_K matvec");
    fails += check(be->desc->vtbl->matmul_q4k != nullptr,
                   "metal exposes Q4_K matmul");
    fails += check(be->desc->vtbl->matvec_q6k != nullptr,
                   "metal exposes Q6_K matvec");
    fails += check(be->desc->vtbl->matmul_q6k != nullptr,
                   "metal exposes Q6_K matmul");
    fails += check(be->desc->vtbl->ffn_geglu_block != nullptr,
                   "metal exposes GEGLU FFN block");
    fails += check(be->desc->vtbl->greedy_head != nullptr,
                   "metal exposes greedy head");
    fails += check(be->desc->vtbl->greedy_head_batch != nullptr,
                   "metal exposes batched greedy head");
    fails += check(be->desc->vtbl->command_sequence_begin != nullptr,
                   "metal exposes command sequence begin");
    fails += check(be->desc->vtbl->command_sequence_end != nullptr,
                   "metal exposes command sequence end");
    fails += check(be->desc->vtbl->command_sequence_read_token != nullptr,
                   "metal exposes command sequence token read");
    fails += check(be->desc->vtbl->command_sequence_read_tokens != nullptr,
                   "metal exposes command sequence token batch read");
    fails += check(
        be->desc->vtbl->command_sequence_replay_decode_greedy_step != nullptr,
        "metal exposes decode greedy command sequence replay");
    fails += check(be->desc->vtbl->attention_block != nullptr,
                   "metal exposes attention block");
    fails += check(be->desc->vtbl->attention_query_block != nullptr,
                   "metal exposes attention query block");

    struct geist_backend_accel_caps small = {0};
    enum geist_status qs = be->desc->vtbl->query_accel_caps(be, &small);
    fails += check(qs == GEIST_E_INVALID_ARG,
                   "metal caps rejects undersized struct");

    struct geist_backend_accel_caps caps = {
        .struct_size = sizeof(caps),
    };
    qs = be->desc->vtbl->query_accel_caps(be, &caps);
    fails += check(qs == GEIST_OK, "metal caps query OK");
    fails += check(caps.device_resident_buffers,
                   "metal caps has device-resident buffers");
    fails += check(caps.compute_queue, "metal caps has compute queue");
    fails += check(caps.pipeline_cache, "metal caps has pipeline cache");
    fails += check(caps.device_name[0] != '\0',
                   "metal caps device name non-empty");

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
                       GEIST_SUPPORT_NONE,
                   "metal does not advertise F32 dense linear yet");

    struct geist_op_support_query q4_q = {
        .op = GEIST_OP_LINEAR,
        .input_count = 2,
        .inputs = {
            {.dtype = GEIST_DTYPE_F32, .layout = GEIST_LAYOUT_DENSE},
            {.dtype = GEIST_DTYPE_Q4_K,
             .layout = GEIST_LAYOUT_BLOCK_QUANTIZED},
        },
        .output_count = 1,
        .outputs = {{.dtype = GEIST_DTYPE_F32, .layout = GEIST_LAYOUT_DENSE}},
    };
    fails += check(geist_backend_supports_op(be, &q4_q) ==
                       GEIST_SUPPORT_NATIVE,
                   "metal advertises native Q4_K decode linear");

    geist_backend_destroy(be);
#else
    fails += check(s == GEIST_E_NOT_FOUND,
                   "metal backend is absent from non-metal builds");
    fails += check(be == nullptr,
                   "failed metal create leaves output null");
#endif

    if (fails == 0) {
        printf("PASS: backend metal registry\n");
        return GEIST_TEST_PASS;
    }
    fprintf(stderr, "FAILED: %d check(s)\n", fails);
    return GEIST_TEST_FAIL;
}
