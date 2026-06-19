/*
 * src/archs/audio_conformer/arch.h — audio Conformer encoder (Gemma 4).
 *
 * Layer: ARCHITECTURE. Implements the geist_arch_ops_encoder vtable for
 * the Gemma 4 audio tower (Conformer-based, stateless forward).
 *
 * Defined in (Phase B-5):
 *   src/archs/audio_conformer/arch.c           — descriptor, encode entry
 *   src/archs/audio_conformer/conformer_loop.c — per-block layer sequence
 *   src/archs/audio_conformer/mel_pipeline.c   — FFTW3/vDSP wrapper
 *   src/archs/audio_conformer/projector.c      — audio_dim → LM_dim linear
 */
#ifndef GEIST_INTERNAL_ARCH_AUDIO_CONFORMER_H
#define GEIST_INTERNAL_ARCH_AUDIO_CONFORMER_H

#ifndef GEIST_INTERNAL_ARCH_LAYER
#error "audio_conformer/arch.h is internal to the architecture layer."
#endif

#include <geist.h>

/* Encoder arch_ops vtable — stateless modality encoders (audio, vision).
 *
 * Encoder runs are session-independent (no recurrent state across calls);
 * the encoder weights live in encoder_state owned by the model and shared
 * across all sessions that consume the model. */
struct geist_arch_ops_encoder {
    const char *name;

    /* state_create: load encoder weights + auxiliary data (mel constants
     * for audio, normalization stats for vision). Returns the encoder
     * state pointer or nullptr on failure. */
    void *(*state_create)(struct geist_backend *be, const char *aux_search_root);

    /* state_destroy: free encoder weights. */
    void (*state_destroy)(void *encoder_state);

    /* encode_pcm: 16 kHz int16 PCM → soft-token sequence. Caller provides
     * out_soft buffer of size (max_soft × soft_token_dim() floats). Returns
     * the number of soft tokens produced (≤ max_soft), or 0 on error. */
    size_t (*encode_pcm)(void          *encoder_state,
                         const int16_t *pcm,
                         size_t         n_samples,
                         float         *out_soft,
                         size_t         max_soft);

    /* soft_token_dim: dimensionality of each soft-token vector (1536 for
     * Gemma 4 audio tower). */
    size_t (*soft_token_dim)(const void *encoder_state);
};

/* Concrete descriptor for the Gemma 4 audio Conformer. */
extern const struct geist_arch_ops_encoder geist_arch_audio_conformer;

#endif /* GEIST_INTERNAL_ARCH_AUDIO_CONFORMER_H */
