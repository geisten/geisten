/*
 * test_audio_attach_int — end-to-end audio pipeline via geist_session
 * public API.
 *
 *   PCM (int16, 16 kHz) → mel → Conformer encoder → soft tokens
 *                              → decoder prefill_audio → decode
 *
 * SKIPs cleanly if any of the four required files is missing:
 *   - GGUF model (GEIST_GGUF_PATH or default search)
 *   - audio_tower.safetensors (auto-found near model)
 *   - mel_constants.bin (auto-found)
 *   - input wav file (path passed as argv[1], optional; defaults to
 *     audio_bench/de_hello.wav)
 *
 * Phase B-5 smoke test for the audio_conformer encoder arch.
 */
#include "test_helpers.h"

#include <geist.h>
#include <geist_util.h>
#include <geist_backend.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Minimal 16-bit PCM WAV reader: skip header, return mono int16 samples
 * at native sample rate. Caller frees. */
static int16_t* read_wav_pcm(const char* path, size_t* n_samples_out, int* sample_rate_out) {
    FILE* f = fopen(path, "rb");
    if (f == nullptr) {
        return nullptr;
    }
    /* Read RIFF header to find data chunk. Validate just enough to be safe. */
    unsigned char hdr[44];
    if (fread(hdr, 1, 44, f) != 44 || memcmp(hdr, "RIFF", 4) != 0 ||
        memcmp(hdr + 8, "WAVE", 4) != 0) {
        fclose(f);
        return nullptr;
    }
    /* Audio format = PCM (1), channels at offset 22, sample rate at 24,
     * bits-per-sample at 34. */
    unsigned short channels = (unsigned short) (hdr[22] | (hdr[23] << 8));
    unsigned int rate =
            (unsigned int) (hdr[24] | (hdr[25] << 8) | (hdr[26] << 16) | (hdr[27] << 24));
    unsigned short bps = (unsigned short) (hdr[34] | (hdr[35] << 8));
    if (channels != 1 || bps != 16) {
        fclose(f);
        return nullptr;
    }
    /* Read remaining samples — assumes "data" chunk starts at offset 36
     * (typical RIFF layout for canonical WAV). */
    fseek(f, 0, SEEK_END);
    long file_bytes = ftell(f);
    long data_bytes = file_bytes - 44;
    fseek(f, 44, SEEK_SET);
    size_t n = (size_t) data_bytes / 2;
    int16_t* pcm = malloc(n * sizeof(int16_t));
    if (pcm == nullptr) {
        fclose(f);
        return nullptr;
    }
    size_t got = fread(pcm, sizeof(int16_t), n, f);
    fclose(f);
    if (got != n) {
        free(pcm);
        return nullptr;
    }
    *n_samples_out = n;
    *sample_rate_out = (int) rate;
    return pcm;
}

int main(int argc, char** argv) {
    GEIST_REQUIRE_GGUF(model_path);

    const char* wav_path = argc > 1 ? argv[1] : "audio_bench/de_hello.wav";
    size_t n_samples;
    int sample_rate;
    int16_t* pcm = read_wav_pcm(wav_path, &n_samples, &sample_rate);
    if (pcm == nullptr) {
        GEIST_SKIP("could not read input wav (set first arg to a valid 16-bit "
                   "mono 16 kHz WAV file, default: audio_bench/de_hello.wav)");
    }
    if (sample_rate != 16000) {
        free(pcm);
        GEIST_SKIP("wav sample rate is not 16 kHz");
    }
    printf("loaded %zu PCM samples @ 16 kHz (%.2f s)\n", n_samples, (double) n_samples / 16000.0);

    struct geist_backend* be = nullptr;
    enum geist_status s = geist_backend_create("cpu_neon", nullptr, nullptr, &be);
    if (s != GEIST_OK) {
        s = geist_backend_create("cpu_scalar", nullptr, nullptr, &be);
    }
    if (s != GEIST_OK) {
        fprintf(stderr, "backend create failed: %s\n", geist_last_create_error());
        free(pcm);
        return GEIST_TEST_ERROR;
    }

    struct geist_model* model = nullptr;
    s = geist_model_load(model_path, be, &model);
    if (s != GEIST_OK) {
        fprintf(stderr, "model_load failed: %s\n", geist_last_create_error());
        geist_backend_destroy(be);
        free(pcm);
        return GEIST_TEST_FAIL;
    }

    struct geist_session_opts opts = {.max_seq_len = 1024};
    struct geist_session* sess = nullptr;
    s = geist_session_create(model, be, &opts, &sess);
    if (s != GEIST_OK) {
        fprintf(stderr, "session_create failed\n");
        geist_model_destroy(model);
        geist_backend_destroy(be);
        free(pcm);
        return GEIST_TEST_FAIL;
    }

    s = geist_session_attach_audio(sess, n_samples, pcm, 16000);
    free(pcm);
    if (s == GEIST_E_NOT_FOUND) {
        printf("SKIP: audio_tower.safetensors or mel_constants.bin not found\n");
        printf("  errmsg: %s\n", geist_session_errmsg(sess));
        geist_session_destroy(sess);
        geist_model_destroy(model);
        geist_backend_destroy(be);
        return GEIST_TEST_SKIP;
    }
    if (s != GEIST_OK) {
        fprintf(stderr,
                "attach_audio failed: %s — %s\n",
                geist_status_to_string(s),
                geist_session_errmsg(sess));
        geist_session_destroy(sess);
        geist_model_destroy(model);
        geist_backend_destroy(be);
        return GEIST_TEST_FAIL;
    }
    printf("audio attached: soft tokens injected into KV cache\n");

    /* Decode a handful of tokens after the audio context. */
    int fails = 0;
    geist_token_t tok;
    printf("decoded:");
    for (int i = 0; i < 6; i++) {
        s = geist_session_decode_step(sess, &tok);
        if (s != GEIST_OK) {
            fprintf(stderr, "\ndecode_step[%d] failed\n", i);
            fails++;
            break;
        }
        const char* text = geist_session_token_to_str(sess, tok);
        printf(" %d(%s)", tok, text != nullptr ? text : "?");
    }
    printf("\n");

    geist_session_destroy(sess);
    geist_model_destroy(model);
    geist_backend_destroy(be);
    return fails == 0 ? GEIST_TEST_PASS : GEIST_TEST_FAIL;
}
