/*
 * video_pipeline — frame batching for the Gemma 4 vision tower.
 *
 * Video frames pass through the same per-frame preprocessing as images,
 * with one important shape difference: per-frame soft-token budget is
 * smaller (70 vs 280) so 32 frames fit in the LM context. The tower is
 * then run on the full batch (M = n_frames * patches_per_frame) so the
 * GEMMs amortize across frames.
 *
 * Frame sampling (picking 32 from a longer clip) is OUT of scope —
 * geist consumes a caller-supplied frame buffer.
 *
 * Phase status:
 *   P1 ⇐ THIS — header + stubs
 *   P6        — batched tower path, parity vs HF
 */
#ifndef VIDEO_PIPELINE_H
#define VIDEO_PIPELINE_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#define VIDEO_TARGET_FRAMES 32  /* per Gemma 4 video_processor.num_frames */

/* Plan for one video. Per-frame plan + batch dims. */
struct video_plan {
    size_t n_frames;
    size_t frame_h, frame_w;            /* native frame dims */
    size_t resized_h, resized_w;
    size_t grid_h, grid_w;
    size_t pool_h, pool_w;
    size_t soft_tokens_per_frame;
    size_t soft_tokens_total;
};

/* Compute a plan that respects the per-frame soft-token cap.
 * Returns false if frames are degenerate. */
bool video_pipeline_plan(size_t n_frames, size_t frame_h, size_t frame_w,
                          size_t max_soft_per_frame,
                          struct video_plan *out);

/* Preprocess n_frames into a batched fp32 patch tensor.
 *   frames_in:    (n_frames, frame_h, frame_w, 3) row-major uint8
 *   out_patches:  (n_frames * grid_h * grid_w, 3 * 16 * 16) fp32
 * P1: stub returns false. */
bool video_pipeline_preprocess(const uint8_t *frames_in,
                                const struct video_plan *plan,
                                float *out_patches);

#endif
