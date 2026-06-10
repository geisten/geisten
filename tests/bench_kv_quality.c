/*
 * bench_kv_quality — KV-quant quality A/B via single-pass top-1 prediction.
 *
 * Tokenizes a fixed paragraph, then calls verify_forward on the WHOLE
 * sequence in one batched pass. verify_forward produces the model's
 * argmax at each position simultaneously, so we get N-1 predictions for
 * the cost of one forward pass.
 *
 * Reports top-1 accuracy: fraction of positions where the model's argmax
 * matches the actual next token in the text. A monotonic scalar quality
 * signal sensitive to KV-cache quantization noise.
 *
 * Run with different GEIST_KV_* env vars to compare modes:
 *   GEIST_KV_INT8=0  ./bench_kv_quality   # FP32 KV (reference)
 *   GEIST_KV_INT8=1  ./bench_kv_quality   # INT8 KV
 *   GEIST_KV_KIVI=1  ./bench_kv_quality   # KIVI (future)
 *
 * Uses arch_ops->verify_forward via geist_model_internal_arch_meta — a
 * test-only accessor that bypasses the public session API to keep the
 * harness independent of the live spec_step plumbing. SKIPs if no GGUF
 * or tokenizer is reachable.
 */
#define GEIST_INTERNAL_ENGINE_LAYER
#define GEIST_INTERNAL_ARCH_LAYER

#include "test_helpers.h"

#include "src/engine/model.h"
#include "src/engine/sp_bpe_tokenizer.h"
#include "src/archs/transformer/arch.h"

#include <geist.h>
#include <geist_backend.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ~400 tokens so the KIVI drained 2-bit path is actually exercised
 * (KIVI_K_GROUP_SIZE=128: a drain triggers around token 128 and again
 * around 256, giving us ≥256 tokens of *drained 2-bit* attention reads
 * mixed with the residual). */
static const char* DEFAULT_TEXT =
        "Raspberry Pi is a series of small single-board computers developed "
        "in the United Kingdom by the Raspberry Pi Foundation. The original "
        "model became more popular than anticipated, selling outside its "
        "target market for uses such as robotics. It is widely used in many "
        "areas, such as for weather monitoring, because of its low cost, "
        "modularity, and open design. After the release of the second board "
        "type, the Raspberry Pi Foundation set up a new entity, Raspberry "
        "Pi Trading, to handle commercial operations while the original "
        "Foundation focused on educational outreach. The boards run a "
        "Debian-based operating system called Raspberry Pi OS as the "
        "official supported distribution. The Pi 5 was released in 2023 "
        "with a Broadcom BCM2712 system-on-chip featuring a quad-core "
        "Arm Cortex-A76 processor clocked at 2.4 gigahertz, paired with "
        "an 800 megahertz VideoCore VII graphics processor. The board "
        "includes options for 4 or 8 gigabytes of LPDDR4X memory, two "
        "micro-HDMI ports supporting dual 4K displays, two USB 3.0 ports, "
        "two USB 2.0 ports, gigabit Ethernet, and Wi-Fi 5 plus Bluetooth "
        "5.0 wireless connectivity. A new dedicated power button replaces "
        "the previous practice of pulling the power cable to shut down. "
        "An internal real-time clock with battery backup keeps the system "
        "time accurate across reboots and unpowered intervals. The board "
        "also exposes a PCI Express interface via a flat ribbon connector "
        "for high-speed peripherals such as NVMe storage.";

int main(int argc, char** argv) {
    const char* model_path = argc > 1 ? argv[1] : geist_test_find_gguf();
    GEIST_SKIP_IF(model_path == nullptr, "no GGUF model found — pass path or set GEIST_GGUF_PATH");

    const char* text = (argc > 2) ? argv[2] : DEFAULT_TEXT;

    struct geist_backend* be = nullptr;
    enum geist_status s = geist_backend_create("cpu_neon", nullptr, nullptr, &be);
    if (s != GEIST_OK) {
        s = geist_backend_create("cpu_scalar", nullptr, nullptr, &be);
    }
    if (s != GEIST_OK) {
        fprintf(stderr, "backend create: %s\n", geist_last_create_error());
        return GEIST_TEST_ERROR;
    }

    struct geist_model* model = nullptr;
    s = geist_model_load(model_path, be, &model);
    if (s != GEIST_OK) {
        fprintf(stderr,
                "model_load(%s): %s — %s\n",
                model_path,
                geist_status_to_string(s),
                geist_last_create_error());
        geist_backend_destroy(be);
        return GEIST_TEST_FAIL;
    }

    struct sp_bpe_tokenizer* tok = geist_model_internal_tokenizer(model);
    GEIST_SKIP_IF(tok == nullptr, "no tokenizer.bin reachable — set GEIST_TOKENIZER_PATH");

    uint32_t* ids = nullptr;
    size_t n_ids = 0;
    if (!sp_bpe_tokenizer_encode(tok, text, &ids, &n_ids)) {
        fprintf(stderr, "tokenizer encode failed\n");
        geist_model_destroy(model);
        geist_backend_destroy(be);
        return GEIST_TEST_FAIL;
    }
    if (n_ids < 2) {
        fprintf(stderr, "text too short (n_ids=%zu)\n", n_ids);
        free(ids);
        geist_model_destroy(model);
        geist_backend_destroy(be);
        return GEIST_TEST_FAIL;
    }

    /* verify_forward runs the WHOLE input through the layer stack in one
     * batched pass (using prefill chunks of m_max internally) and writes
     * argmax at each position. We don't need a session — go straight to
     * arch_ops + arch_meta. */
    void* arch_meta = geist_model_internal_arch_meta(model);
    const struct geist_arch_ops_decoder* ops = &geist_arch_transformer;

    if (ops->state_reset == nullptr || ops->verify_forward == nullptr ||
        ops->kv_truncate == nullptr) {
        fprintf(stderr,
                "arch lacks state_reset / verify_forward / kv_truncate — "
                "expected from transformer decoder\n");
        free(ids);
        geist_model_destroy(model);
        geist_backend_destroy(be);
        return GEIST_TEST_FAIL;
    }

    ops->state_reset(arch_meta);
    ops->kv_truncate(arch_meta, 0); /* belt-and-braces: prefix_length may be nonzero */

    const size_t k = n_ids - 1; /* we predict positions 1..N-1 from inputs 0..N-2 */
    geist_token_t* preds = (geist_token_t*) calloc(k, sizeof *preds);
    if (preds == nullptr) {
        fprintf(stderr, "alloc preds (%zu) failed\n", k);
        free(ids);
        geist_model_destroy(model);
        geist_backend_destroy(be);
        return GEIST_TEST_FAIL;
    }
    /* Chunk because verify_forward caps at m_max (default 64). Each chunk
     * extends the KV cache; predictions are appended in order.
     *
     * verify_forward does NOT drain KIVI residuals (it's the tentative-
     * write path), so we follow each chunk with kv_truncate(kv_len) —
     * which is a no-op on kv_len but forces a drain if the residual
     * grew past R. This makes the harness exercise the actual 2-bit
     * drained-cache attention path for KIVI mode, matching what an
     * accept-only decode_step stream would do. For non-KIVI modes the
     * truncate is harmless. */
    const size_t M_MAX = 64;
    size_t kv_len_acc = 0;
    for (size_t off = 0; off < k; off += M_MAX) {
        const size_t chunk = (k - off > M_MAX) ? M_MAX : (k - off);
        enum geist_status vs = ops->verify_forward(
                arch_meta, chunk, (const geist_token_t*) ids + off, preds + off);
        if (vs != GEIST_OK) {
            fprintf(stderr,
                    "verify_forward(chunk@%zu, %zu): %s\n",
                    off,
                    chunk,
                    geist_status_to_string(vs));
            free(preds);
            free(ids);
            geist_model_destroy(model);
            geist_backend_destroy(be);
            return GEIST_TEST_FAIL;
        }
        kv_len_acc += chunk;
        ops->kv_truncate(arch_meta, kv_len_acc);
    }

    /* preds[i] is the model's argmax given prefix ids[0..i]; the target
     * for position i+1 in the text is ids[i+1]. */
    size_t n_correct = 0;
    for (size_t i = 0; i < k - 1; i++) {
        if ((uint32_t) preds[i] == ids[i + 1])
            n_correct++;
    }
    const size_t n_eval = k - 1;
    const double acc = n_eval > 0 ? (double) n_correct / (double) n_eval : 0.0;

    const char* mode_kivi = getenv("GEIST_KV_KIVI");
    const char* mode_int8 = getenv("GEIST_KV_INT8");
    const char* mode_label = (mode_kivi != nullptr && mode_kivi[0] == '1')   ? "KIVI"
                             : (mode_int8 != nullptr && mode_int8[0] == '1') ? "INT8"
                             : (mode_int8 != nullptr && mode_int8[0] == '0')
                                     ? "FP32"
                                     : "FP32 (Apple-default)";

    printf("model:    %s\n", model_path);
    printf("backend:  %s\n", geist_backend_name(be));
    printf("kv_mode:  %s\n", mode_label);
    printf("n_tokens: %zu (n_eval=%zu)\n", n_ids, n_eval);
    printf("top-1 acc: %zu / %zu = %.4f\n", n_correct, n_eval, acc);

    free(preds);
    free(ids);
    geist_model_destroy(model);
    geist_backend_destroy(be);
    return GEIST_TEST_PASS;
}
