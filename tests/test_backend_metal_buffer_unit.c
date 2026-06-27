/*
 * test_backend_metal_buffer_unit - native Metal buffer layer contract.
 *
 * Exercises the first device-resident Metal increment:
 *   - default activation buffers are private/device-only
 *   - staging/host-visible buffers are mappable
 *   - upload/download and backend-to-backend copies preserve bytes
 *   - overlapping same-buffer copies behave like memmove
 *   - invalid arguments and bounds checks fail explicitly
 */
#include "test_helpers.h"

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

static int check_status(enum geist_status got,
                        enum geist_status want,
                        const char *what) {
    if (got != want) {
        fprintf(stderr, "FAIL: %s: got %s, want %s\n",
                what, geist_status_to_string(got),
                geist_status_to_string(want));
        return 1;
    }
    return 0;
}

static int check_bytes(const uint8_t *got,
                       const uint8_t *want,
                       size_t n,
                       const char *what) {
    for (size_t i = 0; i < n; i++) {
        if (got[i] != want[i]) {
            fprintf(stderr,
                    "FAIL: %s[%zu]: got 0x%02x, want 0x%02x\n",
                    what, i, got[i], want[i]);
            return 1;
        }
    }
    return 0;
}

static int create_metal_or_skip(struct geist_backend **out) {
    *out = nullptr;
    enum geist_status s = geist_backend_create("metal", nullptr, nullptr, out);
#if defined(GEIST_BACKEND_METAL) && GEIST_BACKEND_METAL
    if (s == GEIST_E_UNSUPPORTED) {
        printf("SKIP: metal runtime unavailable: %s\n",
               geist_last_create_error());
        return GEIST_TEST_SKIP;
    }
    if (s != GEIST_OK) {
        fprintf(stderr, "FAIL: metal create failed: %s: %s\n",
                geist_status_to_string(s), geist_last_create_error());
        return GEIST_TEST_FAIL;
    }
    return GEIST_TEST_PASS;
#else
    int fails = 0;
    fails += check_status(s, GEIST_E_NOT_FOUND,
                          "metal is absent from non-metal builds");
    fails += check(*out == nullptr,
                   "failed metal create leaves output null");
    return fails == 0 ? GEIST_TEST_SKIP : GEIST_TEST_FAIL;
#endif
}

static int test_invalid_create(struct geist_backend *be) {
    int fails = 0;
    const struct geist_backend_vtbl *v = be->desc->vtbl;
    struct geist_buffer *buf = (struct geist_buffer *) (uintptr_t) 1u;
    enum geist_status s = v->buffer_create(
        be, 16, GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_AUTO, nullptr);
    fails += check_status(s, GEIST_E_INVALID_ARG,
                          "buffer_create rejects null out");

    buf = (struct geist_buffer *) (uintptr_t) 1u;
    s = v->buffer_create(be, 0, GEIST_BUFFER_ACTIVATION,
                         GEIST_MEMORY_AUTO, &buf);
    fails += check_status(s, GEIST_E_INVALID_ARG,
                          "buffer_create rejects zero bytes");
    fails += check(buf == nullptr,
                   "failed zero-byte create clears output");
    return fails;
}

static int test_private_upload_download(struct geist_backend *be) {
    int fails = 0;
    const struct geist_backend_vtbl *v = be->desc->vtbl;
    static const uint8_t src[17] = {
        0x00, 0x01, 0x02, 0x10, 0x20, 0x30, 0x40, 0x7f, 0x80,
        0x81, 0xa0, 0xb1, 0xc2, 0xd3, 0xe4, 0xf5, 0xff,
    };
    uint8_t got[sizeof(src)] = {0};
    uint8_t one = 0;

    struct geist_buffer *buf = nullptr;
    enum geist_status s = v->buffer_create(
        be, sizeof(src), GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_AUTO, &buf);
    fails += check_status(s, GEIST_OK, "private buffer_create OK");
    if (s != GEIST_OK || buf == nullptr) {
        return fails + 1;
    }

    fails += check(v->buffer_map(buf) == nullptr,
                   "default activation buffer is device-only");
    fails += check_status(v->buffer_upload(buf, 0, &one), GEIST_OK,
                          "zero-byte upload OK");
    fails += check_status(v->buffer_download(0, &one, buf), GEIST_OK,
                          "zero-byte download OK");
    fails += check_status(v->buffer_upload(buf, sizeof(src), src), GEIST_OK,
                          "private upload OK");
    fails += check_status(v->buffer_download(sizeof(got), got, buf), GEIST_OK,
                          "private download OK");
    fails += check_bytes(got, src, sizeof(src),
                         "private upload/download roundtrip");
    fails += check_status(v->buffer_upload(buf, sizeof(src) + 1, src),
                          GEIST_E_INVALID_ARG,
                          "upload rejects over-capacity write");
    fails += check_status(v->buffer_download(sizeof(got) + 1, got, buf),
                          GEIST_E_INVALID_ARG,
                          "download rejects over-capacity read");
    fails += check_status(v->buffer_upload(nullptr, sizeof(src), src),
                          GEIST_E_INVALID_ARG,
                          "upload rejects null buffer");
    fails += check_status(v->buffer_download(sizeof(got), got, nullptr),
                          GEIST_E_INVALID_ARG,
                          "download rejects null buffer");

    v->buffer_destroy(be, buf);
    return fails;
}

static int test_host_visible_map(struct geist_backend *be) {
    int fails = 0;
    const struct geist_backend_vtbl *v = be->desc->vtbl;
    static const uint8_t src[8] = {9, 8, 7, 6, 5, 4, 3, 2};
    static const uint8_t via_map[8] = {1, 3, 5, 7, 9, 11, 13, 15};
    uint8_t got[sizeof(src)] = {0};

    struct geist_buffer *buf = nullptr;
    enum geist_status s = v->buffer_create(
        be, sizeof(src), GEIST_BUFFER_STAGING, GEIST_MEMORY_AUTO, &buf);
    fails += check_status(s, GEIST_OK, "staging buffer_create OK");
    if (s != GEIST_OK || buf == nullptr) {
        return fails + 1;
    }

    void *mapped = v->buffer_map(buf);
    fails += check(mapped != nullptr, "staging buffer is mappable");
    fails += check_status(v->buffer_upload(buf, sizeof(src), src), GEIST_OK,
                          "staging upload OK");
    fails += check_status(v->buffer_download(sizeof(got), got, buf), GEIST_OK,
                          "staging download OK");
    fails += check_bytes(got, src, sizeof(src),
                         "staging upload/download roundtrip");

    if (mapped != nullptr) {
        memcpy(mapped, via_map, sizeof(via_map));
        v->buffer_unmap(buf);
        memset(got, 0, sizeof(got));
        fails += check_status(v->buffer_download(sizeof(got), got, buf),
                              GEIST_OK, "mapped host write download OK");
        fails += check_bytes(got, via_map, sizeof(via_map),
                             "mapped write is visible");
    }

    v->buffer_destroy(be, buf);
    return fails;
}

static int test_copy_and_overlap(struct geist_backend *be) {
    int fails = 0;
    const struct geist_backend_vtbl *v = be->desc->vtbl;
    static const uint8_t src[16] = {
        0, 1, 2, 3, 4, 5, 6, 7,
        8, 9, 10, 11, 12, 13, 14, 15,
    };
    static const uint8_t partial_expect[16] = {
        0xee, 0xee, 0xee, 0xee, 3, 4, 5, 6,
        7, 8, 9, 10, 0xee, 0xee, 0xee, 0xee,
    };
    static const uint8_t overlap_forward_expect[16] = {
        0, 1, 2, 3, 0, 1, 2, 3,
        4, 5, 6, 7, 12, 13, 14, 15,
    };
    static const uint8_t overlap_backward_expect[16] = {
        4, 5, 6, 7, 8, 9, 10, 11,
        8, 9, 10, 11, 12, 13, 14, 15,
    };
    uint8_t got[sizeof(src)] = {0};
    uint8_t fill[sizeof(src)];
    memset(fill, 0xee, sizeof(fill));

    struct geist_buffer *a = nullptr;
    struct geist_buffer *b = nullptr;
    enum geist_status s = v->buffer_create(
        be, sizeof(src), GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_AUTO, &a);
    if (s == GEIST_OK) {
        s = v->buffer_create(be, sizeof(src), GEIST_BUFFER_ACTIVATION,
                             GEIST_MEMORY_AUTO, &b);
    }
    fails += check_status(s, GEIST_OK, "copy buffers create OK");
    if (s != GEIST_OK || a == nullptr || b == nullptr) {
        if (a != nullptr) { v->buffer_destroy(be, a); }
        if (b != nullptr) { v->buffer_destroy(be, b); }
        return fails + 1;
    }

    fails += check_status(v->buffer_upload(a, sizeof(src), src), GEIST_OK,
                          "copy source upload OK");
    fails += check_status(v->buffer_upload(b, sizeof(fill), fill), GEIST_OK,
                          "copy destination fill OK");
    fails += check_status(v->buffer_copy(b, 4, a, 3, 8), GEIST_OK,
                          "partial device copy OK");
    fails += check_status(v->buffer_download(sizeof(got), got, b), GEIST_OK,
                          "partial copy download OK");
    fails += check_bytes(got, partial_expect, sizeof(got),
                         "partial device copy bytes");

    uint8_t one = 0;
    fails += check_status(v->buffer_copy(b, 0, a, 0, 0), GEIST_OK,
                          "zero-byte copy OK");
    fails += check_status(v->buffer_copy(b, sizeof(src), a, 0, 1),
                          GEIST_E_INVALID_ARG,
                          "copy rejects destination overflow");
    fails += check_status(v->buffer_copy(b, 0, a, sizeof(src), 1),
                          GEIST_E_INVALID_ARG,
                          "copy rejects source overflow");
    fails += check_status(v->buffer_copy(nullptr, 0, a, 0, 1),
                          GEIST_E_INVALID_ARG,
                          "copy rejects null destination");
    fails += check_status(v->buffer_upload(a, 0, &one), GEIST_OK,
                          "source survives failed copy checks");

    fails += check_status(v->buffer_upload(a, sizeof(src), src), GEIST_OK,
                          "overlap forward reset OK");
    fails += check_status(v->buffer_copy(a, 4, a, 0, 8), GEIST_OK,
                          "overlap forward copy OK");
    fails += check_status(v->buffer_download(sizeof(got), got, a), GEIST_OK,
                          "overlap forward download OK");
    fails += check_bytes(got, overlap_forward_expect, sizeof(got),
                         "overlap forward memmove bytes");

    fails += check_status(v->buffer_upload(a, sizeof(src), src), GEIST_OK,
                          "overlap backward reset OK");
    fails += check_status(v->buffer_copy(a, 0, a, 4, 8), GEIST_OK,
                          "overlap backward copy OK");
    fails += check_status(v->buffer_download(sizeof(got), got, a), GEIST_OK,
                          "overlap backward download OK");
    fails += check_bytes(got, overlap_backward_expect, sizeof(got),
                         "overlap backward memmove bytes");

    v->buffer_destroy(be, a);
    v->buffer_destroy(be, b);
    return fails;
}

static int test_cross_backend_copy_rejected(struct geist_backend *be) {
    int fails = 0;
    const struct geist_backend_vtbl *v = be->desc->vtbl;
    struct geist_backend *other = nullptr;
    enum geist_status s = geist_backend_create("metal", nullptr, nullptr,
                                               &other);
    fails += check_status(s, GEIST_OK, "second metal backend create OK");
    if (s != GEIST_OK || other == nullptr) {
        return fails + 1;
    }

    struct geist_buffer *a = nullptr;
    struct geist_buffer *b = nullptr;
    s = v->buffer_create(be, 4, GEIST_BUFFER_ACTIVATION,
                         GEIST_MEMORY_AUTO, &a);
    if (s == GEIST_OK) {
        s = other->desc->vtbl->buffer_create(
            other, 4, GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_AUTO, &b);
    }
    fails += check_status(s, GEIST_OK, "cross-backend buffers create OK");
    if (s == GEIST_OK) {
        fails += check_status(v->buffer_copy(a, 0, b, 0, 1),
                              GEIST_E_INVALID_ARG,
                              "copy rejects buffers from different backends");
    }

    if (a != nullptr) { v->buffer_destroy(be, a); }
    if (b != nullptr) { other->desc->vtbl->buffer_destroy(other, b); }
    geist_backend_destroy(other);
    return fails;
}

int main(void) {
    struct geist_backend *be = nullptr;
    int create_result = create_metal_or_skip(&be);
    if (create_result != GEIST_TEST_PASS) {
        return create_result;
    }

    int fails = 0;
    const struct geist_backend_vtbl *v = be->desc->vtbl;
    fails += check(v->buffer_create != nullptr, "buffer_create present");
    fails += check(v->buffer_destroy != nullptr, "buffer_destroy present");
    fails += check(v->buffer_upload != nullptr, "buffer_upload present");
    fails += check(v->buffer_download != nullptr, "buffer_download present");
    fails += check(v->buffer_copy != nullptr, "buffer_copy present");
    fails += check(v->buffer_map != nullptr, "buffer_map present");
    fails += check(v->buffer_unmap != nullptr, "buffer_unmap present");
    fails += check(v->buffer_create_aliased == nullptr,
                   "aliased host buffers unsupported");

    if (fails == 0) { fails += test_invalid_create(be); }
    if (fails == 0) { fails += test_private_upload_download(be); }
    if (fails == 0) { fails += test_host_visible_map(be); }
    if (fails == 0) { fails += test_copy_and_overlap(be); }
    if (fails == 0) { fails += test_cross_backend_copy_rejected(be); }

    geist_backend_destroy(be);
    if (fails == 0) {
        printf("PASS: backend metal buffer\n");
        return GEIST_TEST_PASS;
    }
    fprintf(stderr, "FAILED: %d check(s)\n", fails);
    return GEIST_TEST_FAIL;
}
