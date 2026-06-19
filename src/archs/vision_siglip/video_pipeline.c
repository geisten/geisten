/*
 * video_pipeline — P6 impl. Per-frame call into image_pipeline.
 *
 * Per the HF reference (transformers/models/gemma4/modeling_gemma4.py
 * line 2437, get_video_features → `pixel_values_videos.flatten(0, 1)`),
 * HF runs each frame as an independent batch element of the same vision
 * tower used for images — no cross-frame attention, no shared mask.
 * The sequential-per-frame loop in vision_encoder_run_video is therefore
 * semantically identical to HF's batched call.
 *
 * P6 video_plan reuses the image planner with max_soft=70, then carries
 * n_frames + soft_tokens_total so the encoder can size its output
 * buffer.
 */
#include "video_pipeline.h"

#include "image_pipeline.h"

#include <stddef.h>

bool video_pipeline_plan(size_t             n_frames,
                         size_t             frame_h,
                         size_t             frame_w,
                         size_t             max_soft_per_frame,
                         struct video_plan *out) {
    if (out == nullptr || n_frames == 0 || frame_h == 0 || frame_w == 0 ||
        max_soft_per_frame == 0) {
        return false;
    }
    struct image_plan per_frame;
    if (!image_pipeline_plan(frame_h, frame_w, max_soft_per_frame, &per_frame)) {
        return false;
    }
    *out = (struct video_plan) {
            .n_frames              = n_frames,
            .frame_h               = frame_h,
            .frame_w               = frame_w,
            .resized_h             = per_frame.resized_h,
            .resized_w             = per_frame.resized_w,
            .grid_h                = per_frame.grid_h,
            .grid_w                = per_frame.grid_w,
            .pool_h                = per_frame.pool_h,
            .pool_w                = per_frame.pool_w,
            .soft_tokens_per_frame = per_frame.soft_tokens,
            .soft_tokens_total     = per_frame.soft_tokens * n_frames,
    };
    return true;
}

bool video_pipeline_preprocess(const uint8_t           *frames_in,
                               const struct video_plan *plan,
                               float                   *out_patches) {
    if (frames_in == nullptr || plan == nullptr || out_patches == nullptr) {
        return false;
    }
    struct image_plan per_frame = {
            .in_h        = plan->frame_h,
            .in_w        = plan->frame_w,
            .resized_h   = plan->resized_h,
            .resized_w   = plan->resized_w,
            .grid_h      = plan->grid_h,
            .grid_w      = plan->grid_w,
            .pool_h      = plan->pool_h,
            .pool_w      = plan->pool_w,
            .soft_tokens = plan->soft_tokens_per_frame,
    };
    const size_t frame_stride_in  = plan->frame_h * plan->frame_w * 3;
    const size_t patch_px         = 16 * 16 * 3;
    const size_t frame_stride_out = plan->grid_h * plan->grid_w * patch_px;
    for (size_t f = 0; f < plan->n_frames; f++) {
        if (!image_pipeline_preprocess(frames_in + f * frame_stride_in,
                                       &per_frame,
                                       out_patches + f * frame_stride_out)) {
            return false;
        }
    }
    return true;
}
