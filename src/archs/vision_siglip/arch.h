/*
 * src/archs/vision_siglip/arch.h — Gemma 4 vision tower (SigLIP-derived).
 *
 * Layer: ARCHITECTURE. Implements the geist_arch_ops_vision vtable for
 * the Gemma 4 vision tower (16-layer ViT, RMSNorm, GELU-tanh, 2D RoPE
 * theta=100, kernel-3 avg-pool projector, 280 soft tokens per image).
 *
 * Defined in (Phase P1 skeleton, fleshed out P2-P8):
 *   src/archs/vision_siglip/arch.c            — descriptor, encode entry
 *   src/archs/vision_siglip/vision_encoder.c  — tower forward + weight load
 *   src/archs/vision_siglip/vision_kernels.c  — patch-embed, pool, 2D RoPE
 *   src/archs/vision_siglip/image_pipeline.c  — bicubic resize, patchify
 *   src/archs/vision_siglip/video_pipeline.c  — frame batching
 */
#ifndef GEIST_INTERNAL_ARCH_VISION_SIGLIP_H
#define GEIST_INTERNAL_ARCH_VISION_SIGLIP_H

#ifndef GEIST_INTERNAL_ARCH_LAYER
#error "vision_siglip/arch.h is internal to the architecture layer."
#endif

#include <geist.h>

/* Vision encoder arch_ops vtable. Parallel to geist_arch_ops_encoder but
 * with image/video signatures that don't fit the PCM-shaped surface.
 *
 * Encoder runs are session-independent (no recurrent state across calls);
 * weights live in encoder_state owned by the model and shared across all
 * sessions that consume the model. */
struct geist_arch_ops_vision {
    const char *name;

    /* state_create: load tower weights from vision_tower.safetensors.
     * Returns the encoder state pointer or nullptr on failure (missing
     * weight file, OOM, etc.). aux_search_root mirrors the audio path
     * — typically the directory holding the GGUF. */
    void *(*state_create)(struct geist_backend *be, const char *aux_search_root);

    /* state_destroy: free tower weights. */
    void (*state_destroy)(void *encoder_state);

    /* encode_image: RGB uint8 image (H, W, 3) row-major → soft-token
     * sequence. Caller provides out_soft buffer of size (max_soft ×
     * soft_token_dim() floats). Returns the number of soft tokens
     * produced (≤ max_soft), or 0 on error.
     *
     * Image preprocessing (aspect-preserving bicubic resize, patchify,
     * bilinear pos-embed interp) is owned by the encoder — caller hands
     * over already-decoded RGB pixels at whatever native resolution. */
    size_t (*encode_image)(void *encoder_state,
                            const uint8_t *rgb,
                            size_t height, size_t width,
                            float *out_soft, size_t max_soft);

    /* encode_video: stack of n_frames RGB uint8 images, each (H, W, 3).
     * Frames are tower-encoded in one batched pass for SGEMM amortization.
     * Soft tokens are concatenated across frames in input order. Returns
     * total soft-token count (≤ max_soft), or 0 on error.
     *
     * Frame sampling (picking n_frames from a longer clip) is the
     * caller's responsibility — geist does not link a video decoder. */
    size_t (*encode_video)(void *encoder_state,
                            const uint8_t *frames,
                            size_t n_frames, size_t height, size_t width,
                            float *out_soft, size_t max_soft);

    /* soft_token_dim: dimensionality of each soft-token vector. Projector
     * output dim — matches LM hidden_size so soft tokens splice directly
     * into the residual stream. */
    size_t (*soft_token_dim)(const void *encoder_state);
};

/* Concrete descriptor for the Gemma 4 vision tower. */
extern const struct geist_arch_ops_vision geist_arch_vision_siglip;

#endif /* GEIST_INTERNAL_ARCH_VISION_SIGLIP_H */
