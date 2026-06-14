/*
 * test_video_attach_int — end-to-end video pipeline via geist_session.
 *
 *   N frames (uint8 RGB HWC, synthetic) → vision_siglip per-frame
 *   → 70 soft tokens/frame → 2240 total (N=32) → prefill_image → decode
 *
 * Generates 32 synthetic frames with a time-varying phase shift so each
 * frame has slightly different visual content (avoids trivial tower
 * paths that might short-circuit on identical inputs).
 *
 * SKIPs cleanly if model or vision_tower.safetensors is missing.
 */
#include "test_helpers.h"

#include <geist.h>
#include <geist_util.h>
#include <geist_backend.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N_FRAMES 32
#define FRAME_H 192
#define FRAME_W 192

static void synth_frame(int frame_idx, uint8_t* out) {
    /* Three-channel gradient with a per-frame phase shift in each axis,
     * matching the synthetic image generator's flavor: smooth content
     * so bicubic actually interpolates, varied between frames so per-
     * frame tower outputs differ. */
    const float phase = (float) frame_idx / (float) N_FRAMES;
    for (int y = 0; y < FRAME_H; y++) {
        for (int x = 0; x < FRAME_W; x++) {
            float fx = (float) x / (FRAME_W - 1);
            float fy = (float) y / (FRAME_H - 1);
            float r = 255.0f * (0.5f + 0.5f * sinf(6.28f * (fx + phase)));
            float g = 255.0f * (0.5f + 0.5f * sinf(6.28f * (fy + 1.7f * phase)));
            float b = 255.0f * (0.5f + 0.5f * sinf(6.28f * (fx + fy + 2.3f * phase)));
            uint8_t* p = out + (y * FRAME_W + x) * 3;
            p[0] = (uint8_t) (r < 0 ? 0 : (r > 255 ? 255 : r));
            p[1] = (uint8_t) (g < 0 ? 0 : (g > 255 ? 255 : g));
            p[2] = (uint8_t) (b < 0 ? 0 : (b > 255 ? 255 : b));
        }
    }
}

int main(int argc, char** argv) {
    (void) argc;
    (void) argv;
    GEIST_REQUIRE_GGUF(model_path);

    const size_t frame_bytes = (size_t) FRAME_H * FRAME_W * 3;
    uint8_t* frames = malloc((size_t) N_FRAMES * frame_bytes);
    if (frames == nullptr) {
        fprintf(stderr, "alloc failed\n");
        return GEIST_TEST_ERROR;
    }
    for (int f = 0; f < N_FRAMES; f++) {
        synth_frame(f, frames + (size_t) f * frame_bytes);
    }
    printf("synthesized %d × %dx%d RGB frames (%.1f MB)\n",
           N_FRAMES,
           FRAME_H,
           FRAME_W,
           (double) (N_FRAMES * frame_bytes) / (1024.0 * 1024.0));

    struct geist_backend* be = nullptr;
    enum geist_status s = geist_backend_create("cpu_neon", nullptr, nullptr, &be);
    if (s != GEIST_OK)
        s = geist_backend_create("cpu_scalar", nullptr, nullptr, &be);
    if (s != GEIST_OK) {
        fprintf(stderr, "backend create failed: %s\n", geist_last_create_error());
        free(frames);
        return GEIST_TEST_ERROR;
    }

    struct geist_model* model = nullptr;
    s = geist_model_load(model_path, be, &model);
    if (s != GEIST_OK) {
        fprintf(stderr, "model_load failed: %s\n", geist_last_create_error());
        geist_backend_destroy(be);
        free(frames);
        return GEIST_TEST_FAIL;
    }

    struct geist_session_opts opts = {.max_seq_len = 4096};
    struct geist_session* sess = nullptr;
    s = geist_session_create(model, be, &opts, &sess);
    if (s != GEIST_OK) {
        fprintf(stderr, "session_create failed\n");
        geist_model_destroy(model);
        geist_backend_destroy(be);
        free(frames);
        return GEIST_TEST_FAIL;
    }

    s = geist_session_attach_video(sess, N_FRAMES, FRAME_H, FRAME_W, frames);
    free(frames);
    if (s == GEIST_E_NOT_FOUND) {
        printf("SKIP: vision_tower.safetensors not found\n");
        printf("  errmsg: %s\n", geist_session_errmsg(sess));
        geist_session_destroy(sess);
        geist_model_destroy(model);
        geist_backend_destroy(be);
        return GEIST_TEST_SKIP;
    }
    if (s != GEIST_OK) {
        fprintf(stderr,
                "attach_video failed: %s — %s\n",
                geist_status_to_string(s),
                geist_session_errmsg(sess));
        geist_session_destroy(sess);
        geist_model_destroy(model);
        geist_backend_destroy(be);
        return GEIST_TEST_FAIL;
    }
    printf("video attached: %d × 70 = %d soft tokens injected\n", N_FRAMES, N_FRAMES * 70);

    /* Decode 6 tokens just to confirm the KV cache isn't poisoned. As
     * with the image attach-only test, no chat-template wrap means the
     * model emits <pad> — what we're checking is that decode_step
     * doesn't error and the session is alive. */
    int fails = 0;
    geist_token_t tok;
    printf("decoded:");
    for (int i = 0; i < 6; i++) {
        s = geist_session_decode_step(sess, &tok);
        if (s != GEIST_OK) {
            fprintf(stderr, "\ndecode_step[%d] failed: %s\n", i, geist_status_to_string(s));
            fails++;
            break;
        }
        const char* t = geist_session_token_to_str(sess, tok);
        printf(" %d(%s)", tok, t ? t : "?");
    }
    printf("\n");

    geist_session_destroy(sess);
    geist_model_destroy(model);
    geist_backend_destroy(be);
    return fails == 0 ? GEIST_TEST_PASS : GEIST_TEST_FAIL;
}
