/*
 * geist — Pure C23 multimodal-realtime inference library.
 *
 * Public API. All declarations in this file are part of the supported
 * surface; everything else (src/engine/, src/archs/, src/backends/) is
 * internal and may break between versions without notice.
 *
 * Stability tags per declaration:
 *   @stability STABLE        — won't break in 0.x; deprecation cycle for 1.x.
 *   @stability EXPERIMENTAL  — signature may change in any minor version.
 *
 * See README.md and docs/ARCHITECTURE.md for the design rationale.
 */
#ifndef GEIST_H
#define GEIST_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ====================================================================== */
/* Version                                                                 */
/* ====================================================================== */

#define GEIST_VERSION_MAJOR 0
#define GEIST_VERSION_MINOR 1
#define GEIST_VERSION_PATCH 2
#define GEIST_VERSION_STRING "0.1.2"

/* @stability STABLE since 0.1.0 */
const char *geist_version_string(void);
void        geist_version_components(int *major, int *minor, int *patch);

/* ====================================================================== */
/* Status / Errors                                                         */
/* ====================================================================== */

enum geist_status {
    GEIST_OK = 0,

    /* Generic */
    GEIST_E_OOM,             /* allocation failed */
    GEIST_E_INVALID_ARG,     /* nullptr where not allowed, bad enum, etc. */
    GEIST_E_INTERNAL,        /* programmer error, shouldn't happen */

    /* I/O */
    GEIST_E_FILE_NOT_FOUND,
    GEIST_E_IO,              /* read/write failed */
    GEIST_E_FORMAT,          /* corrupt file, wrong magic, etc. */

    /* Capability */
    GEIST_E_UNSUPPORTED,     /* backend cannot run this op/dtype/layout */
    GEIST_E_NOT_FOUND,       /* tensor name not in model, etc. */
    GEIST_E_BACKEND,         /* backend-specific failure */

    /* Lifecycle */
    GEIST_E_INVALID_STATE,   /* op called in wrong order */
    GEIST_E_TOO_MANY_TOKENS, /* hit max_seq_len */
};

/* @stability STABLE since 0.1.0
 * Returns a stable identifier string for the status code. Never returns
 * nullptr. Useful for log messages. */
const char *geist_status_to_string(enum geist_status s);

/* @stability STABLE since 0.1.0
 * Thread-local fallback for errors during create-time API (e.g.
 * geist_backend_create) where no handle exists yet. After a handle is
 * obtained, prefer the per-handle errmsg accessors. */
const char *geist_last_create_error(void);

/* ====================================================================== */
/* Logging                                                                 */
/* ====================================================================== */

enum geist_log_level {
    GEIST_LOG_ERROR = 0,
    GEIST_LOG_WARN  = 1,
    GEIST_LOG_INFO  = 2,
    GEIST_LOG_DEBUG = 3,
    GEIST_LOG_TRACE = 4,
};

/* @stability EXPERIMENTAL — categories and call frequency may evolve. */
typedef void (*geist_log_callback_t)(enum geist_log_level level,
                                     const char           *category,
                                     const char           *message,
                                     void                 *user_data);

/* ====================================================================== */
/* Memory / Allocator                                                      */
/* ====================================================================== */

/* @stability STABLE since 0.1.0 */
struct geist_allocator {
    void *(*alloc)(void *ctx, size_t bytes, size_t alignment);
    void  (*free)(void *ctx, void *ptr);
    void  (*free_all)(void *ctx); /* optional, arena-style; nullptr for malloc-based */
    void *ctx;
};

/* libc malloc/free wrapper. Default if user passes nullptr to *_create. */
extern const struct geist_allocator geist_libc_allocator;

/* ====================================================================== */
/* Tensor types — dtype = logical, layout = physical storage               */
/* ====================================================================== */

enum geist_dtype {
    GEIST_DTYPE_F32,
    GEIST_DTYPE_F16,
    GEIST_DTYPE_BF16,

    GEIST_DTYPE_I8,
    GEIST_DTYPE_U8,

    /* GGUF traditional quants (32-element blocks). */
    GEIST_DTYPE_Q4_0,
    GEIST_DTYPE_Q8_0,

    /* GGUF k-quants (256-element super-blocks). Logically signed
     * fixed-point with per-block scale (and optional min for Q4_K/Q5_K).
     * Use layout=GEIST_LAYOUT_BLOCK_QUANTIZED. */
    GEIST_DTYPE_Q3_K,
    GEIST_DTYPE_Q4_K,
    GEIST_DTYPE_Q5_K,
    GEIST_DTYPE_Q6_K,

    /* GGUF IQ-quants (256-element super-blocks). Codebook-based lookups
     * instead of straight quantization grid: ~2.56 bpw (IQ2_S) and
     * ~3.44 bpw (IQ3_S). Both used by IQ2_M / IQ3_S model variants.
     * Use layout=GEIST_LAYOUT_BLOCK_QUANTIZED. */
    GEIST_DTYPE_IQ2_S,
    GEIST_DTYPE_IQ3_S,

    /* Ternary GGUF quants for 1.58-bit models (BitNet, future). 256-elem
     * super-blocks with one fp16 scale; TQ1_0 packs 5 trits per byte
     * (1.6875 bpw), TQ2_0 packs 4 trits per byte (2.0625 bpw). Logical
     * values {-1, 0, +1} × scale. layout=GEIST_LAYOUT_BLOCK_QUANTIZED. */
    GEIST_DTYPE_TQ1_0,
    GEIST_DTYPE_TQ2_0,

    GEIST_DTYPE_BINARY,  /* 1-bit values; storage via layout */
    GEIST_DTYPE_TERNARY, /* {-1, 0, +1}; storage via layout */

    GEIST_DTYPE_CUSTOM,  /* user-extended via geist_quant_desc.flags */
};

enum geist_layout {
    GEIST_LAYOUT_DENSE,
    GEIST_LAYOUT_PACKED_1BIT,
    GEIST_LAYOUT_PACKED_2BIT,
    GEIST_LAYOUT_TERNARY_BITPLANE,
    GEIST_LAYOUT_TERNARY_BASE3,
    GEIST_LAYOUT_BLOCK_QUANTIZED, /* k-quants family — see quant_desc */
    GEIST_LAYOUT_CUSTOM,
};

enum geist_buffer_role {
    GEIST_BUFFER_WEIGHT,     /* read-only, lifetime = model */
    GEIST_BUFFER_ACTIVATION, /* read+write, lifetime = forward pass */
    GEIST_BUFFER_KV_CACHE,   /* read+write, lifetime = session */
    GEIST_BUFFER_SCRATCH,    /* short-lived workspace */
    GEIST_BUFFER_IO,         /* user-provided host buffer */
    GEIST_BUFFER_STAGING,    /* host-mapped, for upload/download */
};

enum geist_memory_flags {
    GEIST_MEMORY_AUTO         = 0,
    GEIST_MEMORY_HOST         = 1 << 0,
    GEIST_MEMORY_DEVICE       = 1 << 1,
    GEIST_MEMORY_HOST_VISIBLE = 1 << 2,
    GEIST_MEMORY_MAPPED       = 1 << 3,
    /* The buffer's host pointer aliases external storage (e.g. an mmap
     * region owned by the GGUF reader). buffer_destroy frees only the
     * buffer-handle metadata; the caller retains ownership of the
     * underlying bytes. Created via buffer_create_aliased. */
    GEIST_MEMORY_ALIASED      = 1 << 4,
};

/* Quantization metadata for block-quantized layouts.
 * bits_per_value as rational num/den so 1.58-bit (158/100) is exact. */
struct geist_quant_desc {
    int bits_per_value_num;
    int bits_per_value_den;

    int block_size;
    int values_per_block;

    enum geist_dtype scale_dtype;
    size_t           scale_offset;

    enum geist_dtype zero_dtype;
    size_t           zero_offset;

    unsigned int flags; /* CUSTOM-dtype subtype tag */
};

/* Buffer = byte-oriented handle. impl is backend-private. */
struct geist_buffer;

/* Tensor = view onto a buffer with dtype, layout, shape, offset.
 * Strides only meaningful for DENSE layouts. */
struct geist_tensor {
    struct geist_buffer *buffer;
    size_t               offset;

    enum geist_dtype  dtype;
    enum geist_layout layout;

    int     ndim;
    int64_t shape[8];
    int64_t stride[8];

    struct geist_quant_desc quant;
};

/* ====================================================================== */
/* Backend Op Vocab + Capability                                           */
/* ====================================================================== */

enum geist_op {
    /* Shared */
    GEIST_OP_LINEAR,           /* y = x @ W */
    GEIST_OP_RMSNORM,
    GEIST_OP_RESIDUAL_ADD,
    GEIST_OP_SILU_GATE,
    GEIST_OP_EMBEDDING_LOOKUP,

    /* Transformer-specific */
    GEIST_OP_ATTENTION,        /* fused QK^T → softmax → V */
    GEIST_OP_ROPE,

    /* Mamba-specific (added when arch lands; reserved here) */
    GEIST_OP_SSM_STEP,
    GEIST_OP_SSM_SCAN,
    GEIST_OP_CONV1D,

    GEIST_OP_COUNT,
};

enum geist_support {
    GEIST_SUPPORT_NONE,     /* backend cannot do this combination */
    GEIST_SUPPORT_EMULATED, /* backend can do it, but slow path */
    GEIST_SUPPORT_NATIVE,   /* backend has native fast implementation */
};

/* Storage description — what kind of tensor (without the data). */
struct geist_tensor_format {
    enum geist_dtype  dtype;
    enum geist_layout layout;

    int storage_bits_per_value_num;
    int storage_bits_per_value_den;

    int block_values;
    int block_bytes;
};

/* Capability query — describes a complete op signature. */
struct geist_op_support_query {
    enum geist_op op;

    int                       input_count;
    struct geist_tensor_format inputs[8];

    int                       output_count;
    struct geist_tensor_format outputs[4];
};

/* ====================================================================== */
/* Backend                                                                 */
/* ====================================================================== */

struct geist_backend;

struct geist_backend_opts {
    /* @stability EXPERIMENTAL — additional fields may be added. */
    int max_threads;             /* hint; 0 = backend default */
    int max_concurrent_sessions; /* hint for scratch-pool sizing */

    geist_log_callback_t log_cb;
    void                *log_user_data;
    enum geist_log_level log_level_max; /* WARN by default */

    bool enable_op_profiling; /* opt-in expensive timing */
};

/* @stability STABLE since 0.1.0
 * Create a backend by name (e.g. "cpu_neon", "cpu_scalar", "auto"). The
 * special name "auto" picks the best linked backend for the host. Pass
 * nullptr opts/alloc for defaults. */
enum geist_status geist_backend_create(const char                    *name,
                                       const struct geist_backend_opts *opts,
                                       const struct geist_allocator   *alloc,
                                       struct geist_backend          **out);

void              geist_backend_destroy(struct geist_backend *be);
const char       *geist_backend_name(const struct geist_backend *be);
const char       *geist_backend_errmsg(const struct geist_backend *be);
enum geist_status geist_backend_errcode(const struct geist_backend *be);

/* @stability EXPERIMENTAL */
enum geist_support geist_backend_supports_op(struct geist_backend                 *be,
                                             const struct geist_op_support_query *query);

/* ====================================================================== */
/* Model                                                                   */
/* ====================================================================== */

struct geist_model;

/* @stability STABLE since 0.1.0
 * Loads a GGUF model file. Architecture is detected from the GGUF
 * `general.architecture` metadata key; returns GEIST_E_UNSUPPORTED if
 * no architecture matching this build's compiled set is registered. */
enum geist_status geist_model_load(const char            *path,
                                   struct geist_backend  *be,
                                   struct geist_model   **out);

void        geist_model_destroy(struct geist_model *m);
const char *geist_model_errmsg(const struct geist_model *m);

/* @stability EXPERIMENTAL — name may evolve. */
const char *geist_model_arch(const struct geist_model *m);

/* ====================================================================== */
/* Session                                                                 */
/* ====================================================================== */

struct geist_session;

typedef int32_t geist_token_t;

/* @stability EXPERIMENTAL — per-session KV-cache quantization mode.
 * AUTO = env / platform default; FP32/INT8/KIVI = explicit override. */
enum geist_kv_mode {
    GEIST_KV_AUTO = 0,
    GEIST_KV_FP32 = 1,
    GEIST_KV_INT8 = 2,
    GEIST_KV_KIVI = 3,
};

struct geist_session_opts {
    /* Sequence length cap; 0 = use model default. */
    size_t max_seq_len;

    /* Sampler configuration. Applied per-session at session_create time
     * via the architecture's set_session_opts hook; not yet overridable
     * on individual decode_step calls.
     *
     *   temperature  0.0    → greedy argmax (default). >0 → softmax-sample.
     *   top_k        0 or 1 → no top-k filter (or argmax when temp=0).
     *                >1     → keep the top_k largest logits before sampling.
     *   top_p        1.0    → no nucleus filter (default).
     *                0<p<1  → smallest set whose cumulative prob exceeds p.
     *   random_seed  0      → architecture picks a default fixed seed.
     *                != 0   → use this value (deterministic across runs).
     *
     * When both top_k>1 and top_p<1 are set, top_k takes precedence. */
    float    temperature;
    float    top_p;
    int      top_k;
    uint64_t random_seed;

    /* @stability EXPERIMENTAL — AWQ (Activation-aware Weight Quantization)
     * scales file. nullptr = no AWQ. When set, the arch loads scales from
     * the given path and folds attn_norm/ffn_norm gamma at load time plus
     * applies the o_proj/down_proj input scale at runtime. Orthogonal to
     * the weight quantization format. */
    const char *awq_scales_path;

    /* @stability EXPERIMENTAL — per-session KV cache quantization mode.
     * AUTO = take the env-var / platform default (GEIST_KV_KIVI > GEIST_KV_INT8
     *        > Apple FP32 / non-Apple INT8); other values override the env.
     * Different sessions on the same model may use different modes. */
    enum geist_kv_mode kv_mode;

    /* @stability EXPERIMENTAL — verify-forward batch cap. Sizes scratch
     * buffers + KIVI residual ring. 0 = arch default (transformer = 64). */
    size_t m_max;
};

/* @stability STABLE since 0.1.0 */
enum geist_status geist_session_create(struct geist_model              *m,
                                       struct geist_backend            *be,
                                       const struct geist_session_opts *opts,
                                       struct geist_session           **out);

void        geist_session_destroy(struct geist_session *s);
const char *geist_session_errmsg(const struct geist_session *s);

/* @stability STABLE since 0.1.0
 * Reset KV / logits state for a new conversation, keep weights and the
 * session's sampler config. */
enum geist_status geist_session_reset(struct geist_session *s);

/* @stability STABLE since 0.1.0 */
enum geist_status geist_session_set_prompt(struct geist_session *s, const char *prompt);

/* @stability EXPERIMENTAL — tokenize without prefilling. Lets the caller
 * inspect the token IDs that would be produced by set_prompt, e.g. to
 * seed a speculative-decode drafter's history buffer with the prompt
 * tokens. Writes up to `out_capacity` IDs to `out_ids` and the actual
 * count to `*n_out`. Returns GEIST_E_NOT_FOUND if no tokenizer is
 * loaded, GEIST_E_INVALID_ARG on overflow. */
[[nodiscard]] enum geist_status
geist_session_tokenize(struct geist_session *s,
                       const char           *text,
                       size_t                out_capacity,
                       geist_token_t         out_ids[static out_capacity],
                       size_t               *n_out);

/* @stability STABLE since 0.1.0 — bypass tokenization: caller supplies
 * token IDs directly. Useful for testing and for integrations that
 * already have a tokenizer (set_prompt is the wrapper that does the
 * tokenize-then-prefill flow when a tokenizer.bin is available).
 *
 * Appends `n` tokens to the KV cache. After return the next call to
 * geist_session_decode_step yields the prediction for the position
 * following ids[n-1]. */
[[nodiscard]] enum geist_status
geist_session_prefill_tokens(struct geist_session *s, size_t n,
                             const geist_token_t ids[static n]);

/* @stability EXPERIMENTAL — soft-token injection semantics may change.
 *
 * PCM is consumed as 16-bit signed mono at the indicated `sample_rate`.
 * Only 16 kHz is currently supported (returns GEIST_E_UNSUPPORTED
 * otherwise). */
enum geist_status geist_session_attach_audio(struct geist_session *s,
                                             size_t                n_samples,
                                             const int16_t         pcm_samples[static n_samples],
                                             int                   sample_rate);

/* @stability EXPERIMENTAL — vision-tower soft-token injection.
 *
 * RGB is consumed as height × width × 3 uint8 row-major (i.e. HWC,
 * channels innermost). The vision encoder owns aspect-preserving
 * resize, patchification, the 16-block ViT, kernel-3 avg-pool, and
 * the multimodal projector — calling code only needs to supply
 * decoded pixels at native resolution.
 *
 * Returns GEIST_E_NOT_FOUND if vision_tower.safetensors was not found
 * at model-load time. */
enum geist_status geist_session_attach_image(struct geist_session *s,
                                              size_t                height,
                                              size_t                width,
                                              const uint8_t         rgb[static height * width * 3]);

/* @stability EXPERIMENTAL — vision-tower soft-token injection for video.
 *
 * Frames are consumed as n_frames × height × width × 3 uint8 row-major,
 * with all frames at the same resolution (caller's responsibility).
 * Each frame contributes ≤ 70 soft tokens (per Gemma 4 video-processor
 * default) so the LM context fits all 32 frames at ≈ 2240 soft tokens.
 *
 * Frame sampling (selecting 32 frames from a longer clip) is the
 * caller's responsibility — geist does not link a video decoder.
 *
 * Returns GEIST_E_NOT_FOUND if vision_tower.safetensors was not found
 * at model-load time. */
enum geist_status geist_session_attach_video(struct geist_session *s,
                                              size_t                n_frames,
                                              size_t                height,
                                              size_t                width,
                                              const uint8_t         frames[static n_frames * height * width * 3]);

/* @stability EXPERIMENTAL — KV-cache layout API.
 *
 * Pin `n` prefix tokens into the KV cache. After pin_prefix returns
 * GEIST_OK, the session's cache holds those tokens' KV state and any
 * subsequent geist_session_reset() truncates the cache back to this
 * prefix length (rather than 0). Use this to amortize a constant system
 * prompt across many chat turns. The arch decides whether to support
 * pin_prefix at all; transformer (Gemma 4) does, Mamba2 does not.
 *
 * Returns GEIST_E_UNSUPPORTED if the active architecture does not
 * implement prefix pinning. */
enum geist_status geist_session_pin_prefix(struct geist_session *s,
                                            size_t                n,
                                            const geist_token_t   ids[static n]);

/* @stability STABLE since 0.1.0
 * Decode one token autoregressively. Returns GEIST_OK and writes the
 * token id to *out_token. EOS is signalled by token-id, not status. */
enum geist_status geist_session_decode_step(struct geist_session *s, geist_token_t *out_token);

/* @stability EXPERIMENTAL — raw-logits accessor for evaluation / scoring.
 *
 * Returns a pointer to the cached next-position logits and writes the
 * vocab size to *n_logits. Returns nullptr (and sets *n_logits=0) if no
 * logits are pending — call geist_session_prefill_tokens / set_prompt /
 * decode_step first. Pointer is valid until the next mutating call on
 * the session. Returns nullptr if the active architecture does not
 * implement peek_logits (Mamba2 currently). CPU-backend only — GPU
 * backends will need a copy variant; out of scope for 0.1.0. */
const float *geist_session_peek_logits(struct geist_session *s, size_t *n_logits);

/* @stability EXPERIMENTAL — speculative-decode API.
 *
 * One speculative-decode step: drafts up to k_max candidate tokens via
 * an internal n-gram lookup over `history`, then verifies them in a
 * single batched forward pass. Writes the emitted tokens (1..k_max+1)
 * to `out_tokens` and the count to `*n_out`.
 *
 * The drafter's first guess is always the model's own argmax over the
 * already-pending logits (zero cost), so spec_step emits at least 1
 * token per call even when the n-gram drafter has no proposal.
 *
 * `history` should hold every token committed to the cache so far
 * (prompt + previously emitted). The drafter searches it for suffix
 * matches; passing nullptr or history_n=0 degrades to single-token
 * decode. `out_capacity` must be at least k_max + 1.
 *
 * Sampler config: each position is sampled through the session's
 * configured sampler (argmax / top_k / top_p / temperature), same as
 * decode_step.
 *
 * Distribution caveat: under greedy decoding (temperature = 0), the
 * emitted stream is numerically equivalent to running decode_step
 * `*n_out` times. Under stochastic decoding (temperature > 0) the
 * emitted stream is valid (tokens sampled correctly per position) but
 * not distribution-preserving — the simple argmax-style accept-reject
 * loses the rejection-sampling step that would match the target
 * model's exact joint distribution. For strict stochastic equivalence,
 * use decode_step.
 *
 * Falls back to single-token decode if the active architecture lacks
 * the speculative primitives. */
[[nodiscard]] enum geist_status
geist_session_decode_speculative(struct geist_session *s,
                                  size_t                k_max,
                                  size_t                history_n,
                                  const geist_token_t   history[static history_n],
                                  size_t                out_capacity,
                                  geist_token_t         out_tokens[static out_capacity],
                                  size_t               *n_out);

/* @stability STABLE since 0.1.0
 * Translate a token id back to its surface form. Returns nullptr for
 * unknown / control tokens. The pointer is stable for the session's
 * lifetime. */
const char *geist_session_token_to_str(struct geist_session *s, geist_token_t t);

/* ====================================================================== */
/* Stats / Telemetry                                                       */
/* ====================================================================== */

struct geist_session_stats {
    /* Wired (CLOCK_MONOTONIC-based, ~1ns precision). total_decode_ns
     * covers geist_session_decode_step + decode_speculative; the
     * speculative path's verify-forward time counts as decode. */
    uint64_t n_tokens_decoded;
    uint64_t total_decode_ns;
    uint64_t total_prefill_ns;
    uint64_t total_audio_encode_ns;

    /* Stubbed at zero. Backend-side counters not yet plumbed — these
     * land with an opt-in geist_backend_opts.enable_op_profiling
     * configuration in a future revision. */
    uint64_t buffer_alloc_count;
    uint64_t buffer_alloc_bytes_peak;
    uint64_t buffer_alloc_bytes_current;
    uint64_t per_op_ns[GEIST_OP_COUNT];
    uint64_t per_op_calls[GEIST_OP_COUNT];
};

/* @stability EXPERIMENTAL */
enum geist_status geist_session_get_stats(const struct geist_session *s,
                                          struct geist_session_stats *out);
enum geist_status geist_session_reset_stats(struct geist_session *s);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* GEIST_H */
