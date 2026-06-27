/*
 * src/archs/transformer/weight_load/tensor_views.c — GGUF tensor →
 * backend buffer staging and arena capacity computation.
 *
 * Layer: ARCHITECTURE.
 *
 * Extracted from weight_load.c during R5 of the C23/AGENT.md cleanup.
 * Contains:
 *
 *   compute_weight_arena_capacity — sum tensor bytes for arena sizing
 *   load_tensor_to_buffer         — bump-alloc + memcpy, or mmap-alias
 *
 * make_view_2d / make_view_1d / arena_alloc live as static inline in
 * internal.h.
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "internal.h"

#include "gguf_reader.h"
#include "heap.h"

#include <geist.h>
#include <geist_backend.h>

#include <stddef.h>
#include <stdint.h>
#include <string.h>

[[nodiscard]] enum geist_status weight_load_buffer_from_host(
    struct geist_backend *be,
    void *host_ptr,
    size_t n_bytes,
    enum geist_buffer_role role,
    bool prefer_alias,
    struct geist_buffer **out_buf) {

    if (be == nullptr || be->desc == nullptr || be->desc->vtbl == nullptr ||
        host_ptr == nullptr || n_bytes == 0 || out_buf == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    *out_buf = nullptr;

    const struct geist_backend_vtbl *v = be->desc->vtbl;
    if (prefer_alias && v->buffer_create_aliased != nullptr) {
        enum geist_status s = v->buffer_create_aliased(
            be, host_ptr, n_bytes, role, out_buf);
        if (s == GEIST_OK) {
            return GEIST_OK;
        }
        if (s != GEIST_E_UNSUPPORTED) {
            return s;
        }
        *out_buf = nullptr;
    }

    enum geist_status s = v->buffer_create(
        be, n_bytes, role, GEIST_MEMORY_AUTO, out_buf);
    if (s != GEIST_OK) {
        return s;
    }
    s = v->buffer_upload(*out_buf, n_bytes, (const uint8_t *) host_ptr);
    if (s != GEIST_OK) {
        v->buffer_destroy(be, *out_buf);
        *out_buf = nullptr;
        return s;
    }
    return GEIST_OK;
}

[[nodiscard]] enum geist_status compute_weight_arena_capacity(
    struct gguf_ctx *gguf, size_t *out_bytes) {

    size_t total = 0;
    const size_t n = gguf_tensor_count(gguf);
    for (size_t i = 0; i < n; i++) {
        const struct gguf_tensor_t *t = gguf_tensor_at(gguf, i);
        if (t == nullptr) continue;
        const size_t aligned = (t->nbytes + 63u) & ~((size_t) 63u);
        total += aligned;
    }
    /* Headroom for derived buffers: per_layer_model_proj FP32 (2× the
     * F16 source, ~28 MB extra on Gemma 4 E2B). Round up to 64 MB to
     * absorb any other small dequant'd globals. */
    total += 64ULL * 1024 * 1024;
    *out_bytes = total;
    return GEIST_OK;
}

[[nodiscard]] enum geist_status load_tensor_to_buffer(
    struct transformer_arch_state *st, struct gguf_ctx *gguf, const char *name,
    size_t expected_elems, const struct gguf_tensor_t **out_t,
    struct geist_buffer **out_buf) {

    struct geist_backend *be = st->backend;

    *out_buf = nullptr;
    *out_t   = nullptr;

    const struct gguf_tensor_t *t = gguf_get_tensor(gguf, name);
    if (t == nullptr) {
        geist_backend_set_error(be, GEIST_E_NOT_FOUND,
                                "transformer: tensor '%s' not found in GGUF", name);
        return GEIST_E_NOT_FOUND;
    }
    size_t actual = gguf_tensor_elem_count(t);
    if (actual != expected_elems) {
        geist_backend_set_error(be, GEIST_E_FORMAT,
                                "transformer: '%s' has %zu elements, expected %zu",
                                name, actual, expected_elems);
        return GEIST_E_FORMAT;
    }

    /* Two storage modes, picked at state-create time:
     *
     *   β mode (default, post-P1.1.f): weight bytes are copied from
     *   the GGUF mmap into a backend-owned arena via bump-allocation;
     *   gguf_close runs after all loads. Backend has full ownership.
     *   Cost: 2.8 GB upfront disk read + memcpy on Pi 5 IQ2_M.
     *
     *   mmap-alias mode (GEIST_WEIGHT_MMAP=1): weight bytes are NOT
     *   copied; we wrap the mmap pointer in an aliased buffer (the
     *   P0.3 path). gguf_ctx is retained for state lifetime; kernels
     *   read directly from mmap pages. Disk reads happen on demand
     *   during attention. Pi 5 IQ2_M cold-load ~1.7 s.
     *
     * CPU backends usually expose a GEIST_MEMORY_ALIASED buffer to the
     * kernel layer. Device-only backends fall back to buffer_create +
     * buffer_upload, keeping the same tensor metadata while removing the
     * host-pointer requirement. */
    struct geist_buffer *buf = nullptr;
    enum geist_status    s;
    void *raw_ptr;
    if (st->weight_arena != nullptr) {
        /* β: bump-allocate + memcpy. */
        raw_ptr = arena_alloc(st, t->nbytes, 64);
        if (raw_ptr == nullptr) {
            geist_backend_set_error(be, GEIST_E_OOM,
                                    "transformer: weight arena exhausted at '%s' "
                                    "(used %zu, capacity %zu, need %zu)",
                                    name, st->weight_arena_used,
                                    st->weight_arena_capacity, t->nbytes);
            return GEIST_E_OOM;
        }
        memcpy(raw_ptr, t->data, t->nbytes);
    } else {
        /* mmap-alias: zero-copy; gguf mmap retained by caller. */
        raw_ptr = (void *) t->data;
    }
    s = weight_load_buffer_from_host(be, raw_ptr, t->nbytes,
                                     GEIST_BUFFER_WEIGHT, true, &buf);
    if (s != GEIST_OK) {
        return s;
    }

    *out_t   = t;
    *out_buf = buf;
    return GEIST_OK;
}
