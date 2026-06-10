/*
 * test_mel_pipeline — runs mel_pipeline on a WAV file and dumps fp32 mel
 * frames for comparison against the HF reference (dumps/audio/<stem>.npz mel).
 *
 * Usage: test_mel_pipeline <wav> <constants.bin> <out_mel.bin>
 *
 * Implements the same semicausal padding as Gemma4AudioFeatureExtractor:
 * prepend MEL_FRAME_LENGTH/2 = 160 zero samples so the first analysis window
 * is centered at t=0. After that, pure hop=160 sliding.
 */
#include "mel_pipeline.h"
#include "test_helpers.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HOP_LENGTH 160

/* Read 16-bit mono PCM from a WAV file. Returns malloc'd float array in
 * [-1, 1]; *n_out filled with sample count. Caller frees. */
static float* read_wav_mono16k(const char* path, size_t* n_out) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "open %s\n", path);
        return nullptr;
    }
    char hdr[12];
    fread(hdr, 1, 12, f);
    if (memcmp(hdr, "RIFF", 4) || memcmp(hdr + 8, "WAVE", 4)) {
        fprintf(stderr, "not a WAV: %s\n", path);
        fclose(f);
        return nullptr;
    }
    /* Walk chunks until we find 'data'. */
    int sr = 0, nch = 0, bps = 0;
    uint32_t data_bytes = 0;
    while (1) {
        char id[4];
        uint32_t sz;
        if (fread(id, 1, 4, f) != 4 || fread(&sz, 4, 1, f) != 1)
            break;
        if (!memcmp(id, "fmt ", 4)) {
            uint8_t fmt[40] = {0};
            fread(fmt, 1, sz < sizeof(fmt) ? sz : sizeof(fmt), f);
            if (sz > sizeof(fmt))
                fseek(f, sz - sizeof(fmt), SEEK_CUR);
            nch = fmt[2] | (fmt[3] << 8);
            sr = fmt[4] | (fmt[5] << 8) | (fmt[6] << 16) | (fmt[7] << 24);
            bps = fmt[14] | (fmt[15] << 8);
        } else if (!memcmp(id, "data", 4)) {
            data_bytes = sz;
            break;
        } else {
            fseek(f, sz, SEEK_CUR);
        }
    }
    if (sr != 16000 || nch != 1 || bps != 16) {
        fprintf(stderr, "want 16kHz mono 16-bit, got sr=%d nch=%d bps=%d\n", sr, nch, bps);
        fclose(f);
        return nullptr;
    }
    size_t n = data_bytes / 2;
    int16_t* s16 = (int16_t*) malloc(n * sizeof(int16_t));
    fread(s16, 2, n, f);
    fclose(f);
    float* pcm = (float*) malloc(n * sizeof(float));
    for (size_t i = 0; i < n; i++)
        pcm[i] = (float) s16[i] / 32768.0f;
    free(s16);
    *n_out = n;
    return pcm;
}

int main(int argc, char** argv) {
    GEIST_REQUIRE_ARGS(argc, 4, "<wav> <mel_constants.bin> <out_mel.bin>");
    size_t n_pcm = 0;
    float* pcm = read_wav_mono16k(argv[1], &n_pcm);
    if (!pcm)
        return 1;
    fprintf(stderr, "loaded %zu samples (%.2fs)\n", n_pcm, (double) n_pcm / 16000.0);

    struct MelState* mel = mel_create(argv[2]);
    if (!mel) {
        free(pcm);
        return 1;
    }

    /* Build the padded waveform: prepend 160 zeros to match HF semicausal. */
    const size_t pad_left = MEL_FRAME_LENGTH / 2; /* 160 */
    size_t n_padded = pad_left + n_pcm;
    float* padded = (float*) calloc(n_padded, sizeof(float));
    memcpy(padded + pad_left, pcm, n_pcm * sizeof(float));

    /* HF unfolds with frame_size = frame_length + 1 = 321 (the +1 is consumed
     * by preemphasis which we skip; equivalent to needing 320 valid samples
     * starting from offset i*hop). The number of frames is:
     *     num_frames = (n_padded - 321) / hop + 1
     * if at least 321 samples are available. */
    if (n_padded < MEL_FRAME_LENGTH + 1) {
        fprintf(stderr, "audio too short\n");
        return 1;
    }
    size_t n_frames = (n_padded - (MEL_FRAME_LENGTH + 1)) / HOP_LENGTH + 1;
    fprintf(stderr, "computing %zu mel frames\n", n_frames);

    float* out = (float*) malloc(n_frames * MEL_N_MEL * sizeof(float));
    for (size_t i = 0; i < n_frames; i++) {
        const float* frame_in = padded + i * HOP_LENGTH;
        mel_frame_compute(mel, frame_in, out + i * MEL_N_MEL);
    }

    FILE* of = fopen(argv[3], "wb");
    fwrite(out, sizeof(float), n_frames * MEL_N_MEL, of);
    fclose(of);
    fprintf(stderr,
            "wrote %zu mel frames (%zu floats) to %s\n",
            n_frames,
            n_frames * MEL_N_MEL,
            argv[3]);

    free(out);
    free(padded);
    free(pcm);
    mel_destroy(mel);
    return 0;
}
