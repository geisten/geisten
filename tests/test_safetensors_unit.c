/*
 * test_safetensors — bit-exact verification harness for safetensors_reader.
 *
 * Usage:
 *   test_safetensors <model.safetensors> [tensor_name ...]
 *
 * If no tensor names are given, a fixed default set covering all major
 * Gemma 4 module types is used. For each requested tensor, prints a
 * single-line fingerprint:
 *
 *   <name> <dtype> <shape> <nbytes> <first_32_hex> <last_32_hex>
 *
 * Run safetensors_oracle.py with the same arguments to produce the
 * Python reference, then diff the two outputs.
 */
#include "safetensors_reader.h"
#include "test_helpers.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *DEFAULT_NAMES[] = {
        "model.language_model.embed_tokens.weight",
        "model.language_model.embed_tokens_per_layer.weight",
        "model.language_model.layers.0.self_attn.q_proj.weight",
        "model.language_model.layers.4.self_attn.q_proj.weight",
        "model.language_model.norm.weight",
};
static const size_t DEFAULT_COUNT = sizeof(DEFAULT_NAMES) / sizeof(DEFAULT_NAMES[0]);

static void print_hex(const uint8_t *p, size_t n) {
    for (size_t i = 0; i < n; i++) {
        printf("%02x", p[i]);
    }
}

static void print_fingerprint(const struct st_tensor_t *t) {
    printf("%s %s [", t->name, st_dtype_name(t->dtype));
    for (size_t i = 0; i < t->rank; i++) {
        if (i > 0)
            printf(",");
        printf("%zu", t->shape[i]);
    }
    printf("] nbytes=%zu first32=", t->nbytes);

    const uint8_t *p    = (const uint8_t *) t->data;
    size_t         head = t->nbytes < 32 ? t->nbytes : 32;
    print_hex(p, head);

    printf(" last32=");
    if (t->nbytes <= 32) {
        // Already covered; emit empty marker for stable diff.
        printf("(<=head)");
    } else {
        print_hex(p + (t->nbytes - 32), 32);
    }
    printf("\n");
}

int main(int argc, char **argv) {
    GEIST_REQUIRE_ARGS(argc, 2, "<model.safetensors> [tensor_name ...]");

    const char    *errmsg = nullptr;
    struct st_ctx *ctx    = st_open(argv[1], &errmsg);
    if (!ctx) {
        fprintf(stderr, "st_open failed: %s\n", errmsg ? errmsg : "(no message)");
        return 1;
    }

    fprintf(stderr, "# loaded %zu tensors from %s\n", st_count(ctx), argv[1]);

    const char *const *names;
    size_t             names_count;
    if (argc > 2) {
        names       = (const char *const *) (argv + 2);
        names_count = (size_t) (argc - 2);
    } else {
        names       = DEFAULT_NAMES;
        names_count = DEFAULT_COUNT;
    }

    int rc = 0;
    for (size_t i = 0; i < names_count; i++) {
        const struct st_tensor_t *t = st_get(ctx, names[i]);
        if (!t) {
            fprintf(stderr, "MISSING: %s\n", names[i]);
            rc = 1;
            continue;
        }
        print_fingerprint(t);
    }

    st_close(ctx);
    return rc;
}
