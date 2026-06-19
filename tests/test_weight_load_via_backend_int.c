/*
 * test_weight_load_via_backend_int — verifies the pattern by which the
 * B-4e production-swap will load Gemma 4 weights from a GGUF file into
 * backend-owned buffers.
 *
 * For each weight tensor in blk.0 (one full transformer layer's worth):
 *   1. gguf_get_tensor(ctx, name) → pointer + size into mmap'd GGUF
 *   2. backend->buffer_create(bytes, role=WEIGHT) → backend-owned buffer
 *   3. backend->buffer_upload(buf, src, bytes) → copy raw block bytes
 *   4. backend->buffer_download(dst, buf, bytes) → readback
 *   5. memcmp(dst, src, bytes) == 0
 *
 * This proves the buffer pipeline preserves quantized weight bytes exactly,
 * which is the foundation for routing lm.c's per-layer linear() calls
 * through backend->vtbl->linear with the same Q3_K / Q4_K bytes.
 *
 * SKIPs cleanly if no GGUF is available.
 */
#include "test_helpers.h"

#include <geist.h>
#include <geist_backend.h>

#include "gguf_reader.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The 11 per-layer weights for a Gemma 4 transformer block. */
static const char *LAYER_TENSORS[] = {
        "blk.0.attn_norm.weight",
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_v.weight",
        "blk.0.attn_q_norm.weight",
        "blk.0.attn_k_norm.weight",
        "blk.0.attn_output.weight",
        "blk.0.ffn_norm.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_up.weight",
        "blk.0.ffn_down.weight",
        nullptr,
};

/* Global (model-level) tensors. */
static const char *GLOBAL_TENSORS[] = {
        "token_embd.weight",
        "output_norm.weight",
        nullptr,
};

static int load_and_verify(struct geist_backend *be,
                           struct gguf_ctx      *ctx,
                           const char           *name,
                           size_t               *out_bytes_loaded) {
    const struct gguf_tensor_t *t = gguf_get_tensor(ctx, name);
    if (t == nullptr) {
        fprintf(stderr, "  tensor '%s' not found\n", name);
        return 1;
    }

    struct geist_buffer *buf = nullptr;
    enum geist_status    s   = be->desc->vtbl->buffer_create(
            be, t->nbytes, GEIST_BUFFER_WEIGHT, GEIST_MEMORY_AUTO, &buf);
    if (s != GEIST_OK || buf == nullptr) {
        fprintf(stderr, "  buffer_create failed for '%s': %s\n", name, geist_status_to_string(s));
        return 1;
    }

    s = be->desc->vtbl->buffer_upload(buf, t->nbytes, (const uint8_t *) t->data);
    if (s != GEIST_OK) {
        fprintf(stderr, "  buffer_upload failed for '%s': %s\n", name, geist_status_to_string(s));
        be->desc->vtbl->buffer_destroy(be, buf);
        return 1;
    }

    /* Read back and verify bit-identical. */
    uint8_t *readback = malloc(t->nbytes);
    if (readback == nullptr) {
        be->desc->vtbl->buffer_destroy(be, buf);
        return 1;
    }
    s = be->desc->vtbl->buffer_download(t->nbytes, readback, buf);
    if (s != GEIST_OK) {
        fprintf(stderr, "  buffer_download failed for '%s'\n", name);
        free(readback);
        be->desc->vtbl->buffer_destroy(be, buf);
        return 1;
    }
    int mismatch = memcmp(readback, t->data, t->nbytes);
    free(readback);

    /* Verify buffer_map gives a host pointer (CPU shortcut). */
    void *mapped = be->desc->vtbl->buffer_map(buf);
    int   map_ok = (mapped != nullptr && memcmp(mapped, t->data, t->nbytes) == 0);
    be->desc->vtbl->buffer_unmap(buf);
    be->desc->vtbl->buffer_destroy(be, buf);

    if (mismatch != 0) {
        fprintf(stderr, "  '%s' bytes mismatch after upload+download\n", name);
        return 1;
    }
    if (!map_ok) {
        fprintf(stderr, "  '%s' buffer_map didn't expose host bytes\n", name);
        return 1;
    }
    *out_bytes_loaded += t->nbytes;
    return 0;
}

int main(void) {
    GEIST_REQUIRE_GGUF(model_path);

    const char      *err = nullptr;
    struct gguf_ctx *ctx = gguf_open(model_path, &err);
    if (ctx == nullptr) {
        fprintf(stderr, "gguf_open: %s\n", err != nullptr ? err : "(no detail)");
        return GEIST_TEST_ERROR;
    }

    struct geist_backend *be = nullptr;
    enum geist_status     s  = geist_backend_create("cpu_neon", nullptr, nullptr, &be);
    if (s != GEIST_OK) {
        s = geist_backend_create("cpu_scalar", nullptr, nullptr, &be);
    }
    if (s != GEIST_OK) {
        fprintf(stderr, "backend create failed: %s\n", geist_last_create_error());
        gguf_close(ctx);
        return GEIST_TEST_ERROR;
    }

    /* Per-layer (blk.0). */
    int    fails        = 0;
    size_t bytes_loaded = 0;
    int    n_per_layer  = 0;
    printf("loading blk.0 layer weights via %s buffer ops...\n", geist_backend_name(be));
    for (size_t i = 0; LAYER_TENSORS[i] != nullptr; i++) {
        if (load_and_verify(be, ctx, LAYER_TENSORS[i], &bytes_loaded) != 0) {
            fails++;
        } else {
            n_per_layer++;
        }
    }

    /* Global tensors. */
    int n_global = 0;
    printf("loading global tensors via %s buffer ops...\n", geist_backend_name(be));
    for (size_t i = 0; GLOBAL_TENSORS[i] != nullptr; i++) {
        if (load_and_verify(be, ctx, GLOBAL_TENSORS[i], &bytes_loaded) != 0) {
            fails++;
        } else {
            n_global++;
        }
    }

    geist_backend_destroy(be);
    gguf_close(ctx);

    if (fails == 0) {
        printf("PASS: loaded %d per-layer + %d global tensors via backend "
               "(%.2f MB total)\n",
               n_per_layer,
               n_global,
               (double) bytes_loaded / (1024 * 1024));
        printf("  pattern works — Phase B-4e production-swap can scale this to 35 layers\n");
        return GEIST_TEST_PASS;
    }
    return GEIST_TEST_FAIL;
}
