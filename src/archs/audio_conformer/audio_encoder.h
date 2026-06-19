/*
 * audio_encoder — Gemma 4 audio tower (incremental bringup).
 *
 * Public API per laborbuch design (Phase 3-1 thru 3-9). This header tracks
 * implementation phases as they land:
 *
 *   Phase 1 ✅ — mel pipeline (separate module, mel_pipeline.{c,h})
 *   Phase 2  ⇐  THIS — subsample conv-stage; partial encoder for testing
 *   Phase 3-5    Conformer layers + projections (forthcoming)
 *   Phase 8      pthread+CV wiring for blockable pull
 *
 * Until the full pipeline lands, the streaming push/pull API is not yet
 * exposed. Phase-2 surface: create with safetensors path, load weights,
 * run the subsample stage on an externally-supplied mel buffer.
 */
#ifndef AUDIO_ENCODER_H
#define AUDIO_ENCODER_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

struct AudioEncoder;

struct AudioEncoder *audio_encoder_create(const char *safetensors_path);
void                 audio_encoder_destroy(struct AudioEncoder *);

/* Phase 2 only — direct exposure of subsample stage for validation.
 *   mel_in:  (n_mel_frames, 128)  fp32
 *   mask_in: (n_mel_frames,)      bool — true = valid frame (NULL = all valid).
 *            HF zeros masked time positions before each conv layer.
 *   out:     (n_out, 1024)        fp32, where n_out = ((n_mel_frames+1)/2 + 1) / 2
 * Returns n_out (number of subsampled frames produced). */
size_t audio_encoder_subsample_run(const struct AudioEncoder *,
                                   const float *mel_in,
                                   const bool  *mask_in,
                                   size_t       n_mel_frames,
                                   float       *out);

/* Phase 3 — single Conformer layer forward. Operates on (n, 1024) hidden.
 * Caller passes the chunked attention mask (1, num_blocks, chunk_size,
 * context_size) bool, and the precomputed relative position embeddings
 * (P, 1024). Both come from audio_encoder helpers below. */
void audio_encoder_layer_run(const struct AudioEncoder *,
                             int          layer_idx,
                             const float *h_in,
                             size_t       n,
                             const float *pos_emb,
                             const bool  *attn_mask_5d,
                             float       *h_out);

/* Compute relative position embeddings used by all attention layers.
 * Returns malloc'd (P, 1024) buffer; caller frees. P = context_size / 2 + 1,
 * with context_size = chunk_size + max_past_horizon + max_future_horizon. */
float *audio_encoder_compute_pos_emb(const struct AudioEncoder *);

/* Build the chunked-attention mask for `n_subsample_frames` valid frames.
 * Returns malloc'd (num_blocks, chunk_size, context_size) bool buffer. */
bool *audio_encoder_compute_attn_mask(const struct AudioEncoder *, size_t n);

/* Full audio-tower pipeline: mel → subsample → 12× Conformer → output_proj →
 * embed_audio. Output is the (T_sub, 1536) soft-token sequence ready for
 * the LM. Caller provides padded mel buffer + per-frame mask analogous to
 * audio_encoder_subsample_run.
 * Returns n_softtokens produced. */
#define AUDIO_SOFT_TOKEN_DIM 1536
size_t audio_encoder_run(const struct AudioEncoder *,
                         const float *mel_in,
                         const bool  *mel_mask_in,
                         size_t       n_mel_frames,
                         float       *softtokens_out);

/* === Phase 8 — streaming push/pull API (thread-safe) ===
 *
 * Frontend pattern (single-thread / stdin-style):
 *   while ((n = read(stdin, pcm))) audio_encoder_push_pcm(enc, pcm, n);
 *   audio_encoder_end_input(enc);
 *   while ((m = audio_encoder_pull_softtokens(enc, soft, 32, 0))) handle(m);
 *
 * Frontend pattern (multi-thread / mic-style):
 *   audio thread:   audio_encoder_push_pcm(enc, samples, n);  // fast, memcpy
 *   user button:    audio_encoder_end_input(enc);
 *   inf thread:     while (running) {
 *                       n = audio_encoder_pull_softtokens(enc, buf, 32, -1);
 *                       if (n) ... else if (audio_encoder_segment_done(enc)) ...
 *                   }
 *
 * Internal model: PCM is buffered as it arrives. The audio_encoder_run
 * pipeline triggers on first pull AFTER end_input(), producing the full
 * soft-token sequence at once. Pull-calls drain that sequence in chunks
 * of up to max_out tokens.
 *
 * (True chunk-streaming — incremental encode while audio still arrives —
 * is a future Phase 8b. Current semantics suit file-based and
 * push-to-talk frontends; live continuous mic needs Phase 8b.) */

/* Append PCM samples to the internal buffer. Returns 0 on success, -1 on
 * overflow (>30s buffered = audio_seq_length limit) or after shutdown. */
int audio_encoder_push_pcm(struct AudioEncoder *, const int16_t *samples, size_t n);

/* Mark end-of-input. The next pull call will trigger encoder execution.
 * After end_input, pull calls drain the soft-token sequence and then
 * audio_encoder_segment_done() returns true. */
void audio_encoder_end_input(struct AudioEncoder *);

/* Drain up to `max_out` soft-tokens (each AUDIO_SOFT_TOKEN_DIM=1536 floats).
 * timeout_ms: 0 = non-blocking, -1 = block until ready or shutdown,
 *             >0 = block up to N ms. Returns count copied (0 if none ready
 * or timed out). After the segment is fully drained returns 0. */
size_t
audio_encoder_pull_softtokens(struct AudioEncoder *, float *out, size_t max_out, int timeout_ms);

/* True iff end_input was called AND all soft-tokens have been pulled.
 * Use to decide when to call audio_encoder_reset() for a new utterance. */
bool audio_encoder_segment_done(const struct AudioEncoder *);

/* Clear streaming state (PCM buffer, output queue) for a new utterance.
 * Weights stay loaded. */
void audio_encoder_reset(struct AudioEncoder *);

/* === Phase 8b: chunk-streaming forward (parity API for tests). ===
 * These are the building blocks for true incremental encode. The push_pcm
 * path will eventually drive them from a background worker; for now they
 * are exposed so tests can drive them directly and validate parity
 * against the monolithic audio_encoder_run. */
struct audio_stream_state;
size_t                     audio_encoder_stream_push(const struct AudioEncoder *,
                                                     struct audio_stream_state *,
                                                     const float *mel_full,
                                                     const bool  *mel_mask,
                                                     size_t       n_mel_total,
                                                     bool         is_final);
struct audio_stream_state *audio_encoder_stream_state(struct AudioEncoder *);
const float               *audio_stream_state_soft(const struct audio_stream_state *);
size_t                     audio_stream_state_n_soft(const struct audio_stream_state *);

/* Wake any blocked pull calls so threads can exit. After shutdown,
 * push/pull all return immediately as no-ops. */
void audio_encoder_shutdown(struct AudioEncoder *);

#endif
