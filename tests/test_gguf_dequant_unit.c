/*
 * test_gguf_dequant — dequantizes a tensor from a GGUF file and writes
 * raw FP32 binary. validate_gguf_dequant.py compares this against gguf-py's
 * dequantization for bit-close parity.
 */
#include "gguf_reader.h"
#include "quant.h"
#include "test_helpers.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    GEIST_REQUIRE_ARGS(argc, 4, "<model.gguf> <tensor_name> <out.bin>");

    const char      *err = nullptr;
    struct gguf_ctx *ctx = gguf_open(argv[1], &err);
    if (!ctx) {
        fprintf(stderr, "gguf_open: %s\n", err);
        return 1;
    }

    const struct gguf_tensor_t *t = gguf_get_tensor(ctx, argv[2]);
    if (!t) {
        fprintf(stderr, "missing tensor: %s\n", argv[2]);
        gguf_close(ctx);
        return 1;
    }

    size_t elems = gguf_tensor_elem_count(t);
    fprintf(stderr, "tensor %s: %s, dims=[", t->name, gguf_dtype_name(t->dtype));
    for (int d = 0; d < t->n_dims; d++) {
        if (d > 0)
            fprintf(stderr, ",");
        fprintf(stderr, "%llu", (unsigned long long) t->dims[d]);
    }
    fprintf(stderr, "], %zu elems, %zu bytes\n", elems, t->nbytes);

    float *out = (float *) malloc(elems * sizeof(float));
    if (!out) {
        gguf_close(ctx);
        return 1;
    }

    switch (t->dtype) {
    case GGUF_TYPE_Q8_0:
        dequant_q8_0_row(t->data, out, elems);
        break;
    case GGUF_TYPE_Q4_K:
        dequant_q4_K_row(t->data, out, elems);
        break;
    case GGUF_TYPE_Q6_K:
        dequant_q6_K_row(t->data, out, elems);
        break;
    case GGUF_TYPE_F32:
        memcpy(out, t->data, elems * sizeof(float));
        break;
    case GGUF_TYPE_F16: {
        const uint16_t *h = (const uint16_t *) t->data;
        for (size_t i = 0; i < elems; i++)
            out[i] = fp16_to_fp32(h[i]);
        break;
    }
    default:
        fprintf(stderr, "unsupported dtype %s for dequant\n", gguf_dtype_name(t->dtype));
        free(out);
        gguf_close(ctx);
        return 1;
    }

    FILE *fo = fopen(argv[3], "wb");
    if (!fo) {
        perror("fopen out");
        free(out);
        gguf_close(ctx);
        return 1;
    }
    xfwrite(out, sizeof(float), elems, fo);
    fclose(fo);
    fprintf(stderr, "wrote %zu floats to %s\n", elems, argv[3]);

    free(out);
    gguf_close(ctx);
    return 0;
}
