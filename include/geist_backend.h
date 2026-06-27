/*
 * geist_backend.h — extension API for backend authors.
 *
 * Include this in addition to <geist.h> when implementing a new backend
 * (cpu_neon, cpu_scalar, rknn-npu, etc.). Defines the vtable shape, the
 * descriptor each backend exports, and the engine-side registration
 * mechanism.
 *
 * @stability EXPERIMENTAL — vtable layout may evolve until 1.0.
 */
#ifndef GEIST_BACKEND_H
#define GEIST_BACKEND_H

#include <geist.h>
#include <geist_types.h>  /* tensor / op / dtype types the vtable speaks in */
#include <geist_weight.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ====================================================================== */
/* Backend Vtable                                                          */
/* ====================================================================== */

/* Execution phase the arch layer is about to enter. Backends may tune their
 * parallelism regime per phase (see parallel_region_begin). Prefill is
 * compute-bound and scales with cores; decode (m=1 GEMV) is memory-bound. */
enum geist_parallel_region {
    GEIST_REGION_PREFILL_BATCH,
    GEIST_REGION_DECODE_STEP,
};

/* Optional backend command batching scope. GPU backends may record several
 * higher-level ops into one command sequence and submit once at end. CPU
 * backends leave the hooks null. */
enum geist_command_sequence_kind {
    GEIST_COMMAND_SEQUENCE_DECODE_LAYER_LOOP,
    GEIST_COMMAND_SEQUENCE_DECODE_GREEDY_STEP,
    GEIST_COMMAND_SEQUENCE_PREFILL_TEXT,
    GEIST_COMMAND_SEQUENCE_VERIFY_GREEDY,
};

/* Optional backend accelerator capabilities. Backends that can expose a
 * device-resident transformer route fill this by value; callers initialize
 * struct_size to sizeof(struct geist_backend_accel_caps). Missing optional
 * features are represented by false flags, not by failing the query. */
struct geist_backend_accel_caps {
    size_t struct_size;

    bool device_resident_buffers;
    bool compute_queue;
    bool pipeline_cache;
    bool subgroup_basic;
    bool shader_integer_dot_product;
    bool descriptor_indexing;
    bool timeline_semaphore;

    uint64_t device_local_bytes;
    char     device_name[128];
};

/* Optional Level-3 GEGLU FFN block contract.
 *
 * Computes the transformer FFN block:
 *   pre_ff  = rmsnorm(residual, ffn_norm_weight, eps)
 *   gate    = gate_weight @ pre_ff
 *   up      = up_weight   @ pre_ff
 *   gate    = gelu_tanh(gate) * up
 *   ffn_out = down_weight @ gate
 *   out     = residual + (post_norm(ffn_out) when post_ffw_norm_weight != NULL
 *                         else ffn_out)
 *
 * All activation/scratch/output tensors are F32 DENSE backend buffers. Weight
 * tensors are backend buffers with shapes [inter, d_model] for gate/up and
 * [d_model, inter] for down; backends may restrict dtype/layout and return
 * GEIST_E_UNSUPPORTED for non-native forms. The caller owns every buffer and
 * provides all scratch; implementations must not retain pointers after return. */
struct geist_backend_ffn_geglu_block {
    size_t struct_size;

    size_t seq;
    size_t d_model;
    size_t inter;
    float  eps;

    const struct geist_tensor *residual;
    const struct geist_tensor *ffn_norm_weight;
    const struct geist_tensor *gate_weight;
    const struct geist_tensor *up_weight;
    const struct geist_tensor *down_weight;
    const struct geist_tensor *post_ffw_norm_weight; /* nullable */

    struct geist_tensor *pre_ff_scratch;
    struct geist_tensor *gate_scratch;
    struct geist_tensor *up_scratch;
    struct geist_tensor *ffn_out_scratch;
    struct geist_tensor *post_ff_scratch; /* required when post_ffw_norm_weight != NULL */
    struct geist_tensor *out;
};

/* Optional Level-3 attention block contract for device-resident decode.
 *
 * Computes the Gemma-style attention block for a single token:
 *   normed = rmsnorm(residual, attn_norm_weight, eps)
 *   q/k/v  = q/k/v_proj_weight @ normed
 *   q      = rmsnorm(q per head, q_norm_weight, eps)
 *   k      = rmsnorm(k per head, k_norm_weight, eps)
 *   v      = rmsnorm(v per head, v_norm_weight, eps)
 *   q/k    = rope(q/k, cos, sin)
 *   append k/v into F32 KV cache at q_position
 *   attn   = causal/sliding-window attention(q, k_cache, v_cache)
 *   o      = o_proj_weight @ attn
 *   out    = residual + optional rmsnorm(o, post_attn_norm_weight, eps)
 *
 * This is intentionally narrower than the decomposed architecture path: it is
 * a GPU decode fast path for seq==1 with F32 KV. cos/sin may either be
 * row views for the current seq or full RoPE tables with at least
 * q_position + seq rows. Unsupported model features return
 * GEIST_E_UNSUPPORTED so the caller can fall back to the scalar sequence
 * of Level-2 ops. */
struct geist_backend_attention_block {
    size_t struct_size;

    size_t q_position;
    size_t kv_len;
    size_t d_model;
    size_t q_heads;
    size_t kv_heads;
    size_t head_dim;
    size_t sliding_window;
    float  eps;

    const struct geist_tensor *residual;
    const struct geist_tensor *attn_norm_weight;
    const struct geist_tensor *q_proj_weight;
    const struct geist_tensor *k_proj_weight;
    const struct geist_tensor *v_proj_weight;
    const struct geist_tensor *q_norm_weight;
    const struct geist_tensor *k_norm_weight;
    const struct geist_tensor *v_norm_weight;
    const struct geist_tensor *cos;
    const struct geist_tensor *sin;
    const struct geist_tensor *k_cache;
    const struct geist_tensor *v_cache;
    const struct geist_tensor *o_proj_weight;
    const struct geist_tensor *post_attn_norm_weight; /* nullable */

    struct geist_tensor *normed_scratch;
    struct geist_tensor *q_scratch;
    struct geist_tensor *k_scratch;
    struct geist_tensor *v_scratch;
    struct geist_tensor *attn_scratch;
    struct geist_tensor *o_scratch;
    struct geist_tensor *post_attn_scratch; /* required when post_attn_norm_weight != NULL */
    struct geist_tensor *out;
};

/* Optional Level-3 attention query block for KV-shared decode layers.
 *
 * Computes the Gemma-style attention block for a single token when K/V were
 * produced by an earlier source layer:
 *   normed = rmsnorm(residual, attn_norm_weight, eps)
 *   q      = q_proj_weight @ normed
 *   q      = rmsnorm(q per head, q_norm_weight, eps)
 *   q      = rope(q, cos, sin)
 *   attn   = causal/sliding-window attention(q, existing k_cache, v_cache)
 *   o      = o_proj_weight @ attn
 *   out    = residual + optional rmsnorm(o, post_attn_norm_weight, eps)
 *
 * This is the KV-shared counterpart to geist_backend_attention_block: it does
 * not compute, normalize, rotate, or append K/V. cos/sin follow the same
 * row-view-or-full-table contract as attention_block. Unsupported model
 * features return GEIST_E_UNSUPPORTED so the caller can fall back to
 * decomposed ops. */
struct geist_backend_attention_query_block {
    size_t struct_size;

    size_t q_position;
    size_t kv_len;
    size_t d_model;
    size_t q_heads;
    size_t kv_heads;
    size_t head_dim;
    size_t sliding_window;
    float  eps;

    const struct geist_tensor *residual;
    const struct geist_tensor *attn_norm_weight;
    const struct geist_tensor *q_proj_weight;
    const struct geist_tensor *q_norm_weight;
    const struct geist_tensor *cos;
    const struct geist_tensor *sin;
    const struct geist_tensor *k_cache;
    const struct geist_tensor *v_cache;
    const struct geist_tensor *o_proj_weight;
    const struct geist_tensor *post_attn_norm_weight; /* nullable */

    struct geist_tensor *normed_scratch;
    struct geist_tensor *q_scratch;
    struct geist_tensor *attn_scratch;
    struct geist_tensor *o_scratch;
    struct geist_tensor *post_attn_scratch; /* required when post_attn_norm_weight != NULL */
    struct geist_tensor *out;
};

/* Optional Level-3 Gemma PLE injection block for device-resident decode.
 *
 * Computes:
 *   gate = per_layer_gate_weight @ hidden
 *   gate = gelu_tanh(gate) * per_layer_input
 *   proj = per_layer_proj_weight @ gate
 *   proj = rmsnorm(proj, post_per_layer_norm_weight, eps)
 *   out  = hidden + proj
 *
 * The caller provides all activation scratch. This block is intentionally
 * narrow: seq==1 decode is the primary target, and backends may reject other
 * shapes/dtypes with GEIST_E_UNSUPPORTED. */
struct geist_backend_ple_block {
    size_t struct_size;

    size_t seq;
    size_t d_model;
    size_t hidden_per_layer;
    float  eps;

    const struct geist_tensor *hidden;
    const struct geist_tensor *per_layer_input;
    const struct geist_tensor *per_layer_gate_weight;
    const struct geist_tensor *per_layer_proj_weight;
    const struct geist_tensor *post_per_layer_norm_weight;

    struct geist_tensor *gate_scratch;
    struct geist_tensor *proj_scratch;
    struct geist_tensor *out;
};

/* Optional Level-3 greedy output-head contract.
 *
 * Computes the single-row decode head:
 *   normed = rmsnorm(hidden, norm_weight, eps)
 *   logits = lm_head_weight @ normed
 *   out_token = argmax(logits)
 *
 * `hidden`, `normed_scratch`, and `logits` are F32 DENSE backend buffers.
 * `lm_head_weight` may be F32 DENSE or a backend-supported quantized
 * [vocab_size, d_model] tensor. The caller owns all tensors and the token
 * output; implementations must not retain pointers after return. */
struct geist_backend_greedy_head {
    size_t struct_size;

    size_t d_model;
    size_t vocab_size;
    size_t token_output_offset;
    float  eps;

    const struct geist_tensor *hidden;
    const struct geist_tensor *norm_weight;
    const struct geist_tensor *lm_head_weight;

    struct geist_tensor *normed_scratch;
    struct geist_tensor *logits;
};

/* Optional Level-3 batched greedy output-head contract.
 *
 * Computes `row_count` independent greedy heads from hidden[row]:
 *   normed[row] = rmsnorm(hidden[row], norm_weight, eps)
 *   logits[row] = lm_head_weight @ normed[row]
 *   out_tokens[row] = argmax(logits[row])
 *
 * `hidden`, `normed_scratch`, and `logits` are F32 DENSE backend buffers.
 * `hidden` and `normed_scratch` are [row_count, d_model]; `logits` is
 * [row_count, vocab_size]. Implementations may record work and defer token
 * readback when inside a command sequence; in that case token slots start at
 * `token_output_offset` and callers read them via command_sequence_read_tokens. */
struct geist_backend_greedy_head_batch {
    size_t struct_size;

    size_t d_model;
    size_t vocab_size;
    size_t row_count;
    size_t token_output_offset;
    float  eps;

    const struct geist_tensor *hidden;
    const struct geist_tensor *norm_weight;
    const struct geist_tensor *lm_head_weight;

    struct geist_tensor *normed_scratch;
    struct geist_tensor *logits;
};

/* Backends fill in this struct (statically) and reference it from their
 * descriptor. Engine dispatches op-calls through here. */
struct geist_backend_vtbl {
    /* ---- Lifecycle ---- */

    /* Optional create-time hook. Backend allocates per-instance state and
     * stashes it into geist_backend->state. Returns GEIST_OK on success.
     * If non-OK, engine reclaims geist_backend memory and propagates. */
    enum geist_status (*create)(struct geist_backend           *be,
                                              const struct geist_backend_opts *opts);

    /* Required. Tear down per-instance state and any held buffers. */
    void (*destroy)(struct geist_backend *be);

    /* ---- Capability ---- */

    /* Pre-flight check: can this backend execute this op signature? */
    enum geist_support (*supports_op)(struct geist_backend                 *be,
                                      const struct geist_op_support_query *query);

    /* Optional backend-wide accelerator probe. nullptr means this backend has
     * no device-resident acceleration route. The query must not allocate hot
     * path resources; it only reports create-time device/backend facts. */
    enum geist_status (*query_accel_caps)(struct geist_backend            *be,
                                          struct geist_backend_accel_caps *out);

    /* Optional diagnostics for benchmarks/profiling. These hooks are not part
     * of inference semantics and must not be required by production forward
     * paths. nullptr means the backend has no benchmark-visible profile
     * counters. */
    enum geist_status (*profile_reset)(struct geist_backend *be);
    enum geist_status (*profile_dump)(struct geist_backend *be);

    /* ---- Buffer ops ---- */

    /* Allocate a buffer of the given size and role. Backend may pick
     * device-local vs host-coherent based on memory_flags. */
    enum geist_status (*buffer_create)(struct geist_backend      *be,
                                                     size_t                     bytes,
                                                     enum geist_buffer_role     role,
                                                     unsigned int               memory_flags,
                                                     struct geist_buffer      **out);

    void (*buffer_destroy)(struct geist_backend *be, struct geist_buffer *buf);

    /* Create a buffer that aliases an external host-resident region (e.g.
     * an mmap-backed weight tensor whose lifetime is bound to the GGUF
     * reader). The backend wraps host_ptr in a geist_buffer with the
     * GEIST_MEMORY_ALIASED bit set; buffer_destroy releases only the
     * buffer-handle struct and never frees host_ptr. CPU backends return
     * host_ptr unchanged from buffer_map; GPU backends MAY return nullptr
     * (caller must fall back). nullptr means the backend doesn't support
     * aliasing — caller must use buffer_create + buffer_upload. */
    enum geist_status (*buffer_create_aliased)(struct geist_backend  *be,
                                               void                  *host_ptr,
                                               size_t                 n_bytes,
                                               enum geist_buffer_role role,
                                               struct geist_buffer  **out);

    /* Copy host bytes into the buffer. Caller-provided source array. */
    enum geist_status (*buffer_upload)(struct geist_buffer *buf,
                                                     size_t               n_bytes,
                                                     const uint8_t        src[static n_bytes]);

    /* Copy buffer contents back to host. Caller-provided destination. */
    enum geist_status (*buffer_download)(size_t                     n_bytes,
                                                       uint8_t                    dst[static n_bytes],
                                                       const struct geist_buffer *buf);

    /* Copy bytes between two backend buffers without forcing a host mapping.
     * Backends should support overlapping ranges when src == dst. */
    enum geist_status (*buffer_copy)(struct geist_buffer       *dst,
                                     size_t                     dst_offset,
                                     const struct geist_buffer *src,
                                     size_t                     src_offset,
                                     size_t                     n_bytes);

    /* CPU shortcut: returns a host pointer that aliases the buffer.
     * Returns nullptr if the backend cannot produce a host alias for this
     * buffer (e.g. device-only GPU memory). For CPU backends this is the
     * fast path; production code should call sparingly on GPU backends. */
    void *(*buffer_map)(struct geist_buffer *buf);

    /* Counterpart to buffer_map; no-op on CPU, sync on GPU. */
    void (*buffer_unmap)(struct geist_buffer *buf);

    /* ---- Load-time weight resolver (P1.1, refactor v2) ----
     *
     * Inspect a weight tensor's dtype + shape and write direct function
     * pointers into `w->linear_m1` and `w->linear_mN`. Runs once per
     * weight at model load. Subsequent forward calls go through the
     * resolved pointers without per-call dispatch. Optionally allocate
     * `w->aux_fp32` via heap.h for pre-folded data (AWQ etc.).
     *
     * nullable: backends that don't yet implement the new flow (or that
     * fundamentally can't pre-resolve, e.g. a future fully-dynamic GPU
     * backend) leave this slot null. Callers fall back to the legacy
     * per-op vtable path. */
    enum geist_status (*resolve_weight)(struct geist_backend *be,
                                        struct geist_weight  *w);

    /* Optional backend-specific immutable weight preparation. GPU backends
     * use this at model/accelerator setup time to build device-local packed
     * layouts and record later command buffers against those buffers. This
     * hook may allocate and copy; callers must not invoke it from inference
     * hot paths. nullptr means no extra preparation is required. */
    enum geist_status (*prepare_weight_layout)(struct geist_backend      *be,
                                               const struct geist_tensor *w);

    /* Optional load-time weight preparation from caller-owned host bytes.
     * This lets GPU backends build/cache backend-specific packed layouts
     * while the GGUF/raw payload is still in host memory, avoiding a later
     * device-to-host readback in prepare_weight_layout. The backend must copy
     * or upload anything it needs before returning; raw is only valid for the
     * duration of the call. nullptr means callers use prepare_weight_layout. */
    enum geist_status (*prepare_weight_layout_from_host)(
        struct geist_backend      *be,
        const struct geist_tensor *w,
        size_t                     raw_bytes,
        const uint8_t              raw[static raw_bytes]);

    /* ---- Primitive Ops (Level 2 per Q17) ---- */
    /* Each op takes geist_tensor inputs/outputs whose .buffer was created
     * via this same backend. Return GEIST_OK on success; on error, set
     * the backend error slot via geist_backend_set_error_*. */

    /* (P2.e) The legacy `linear` op vtable slot was dropped after the
     * resolver path (resolve_weight + geist_weight::linear_m1/_mN) covered
     * every production dtype. All callers go through linear_w_or_legacy
     * in src/archs/transformer/forward.c, which dispatches solely on the
     * pre-resolved kernel pointers. Adding a new linear path means adding
     * a resolver case, not a vtable entry. */

    /* Optional device-resident decode matvec:
     *   y[n_out] = w[n_out, n_in] @ x[n_in]
     * All tensors are F32 DENSE and must be backend buffers. This is not the
     * host-pointer geist_weight path above; GPU fastpaths use it as a narrow
     * building block while the block-fused transformer route is assembled.
     * y must not overlap x or w. nullptr means unsupported. */
    enum geist_status (*matvec_f32_dense)(struct geist_backend      *be,
                                          const struct geist_tensor *x,
                                          const struct geist_tensor *w,
                                          struct geist_tensor       *y);

    /* Optional device-resident F32 dense matmul:
     *   y[rows, n_out] = x[rows, n_in] @ transpose(w[n_out, n_in])
     * All tensors are F32 DENSE backend buffers. This covers prefill-sized
     * generic linear_w calls that are not yet inside a higher-level fused
     * block. nullptr means unsupported. */
    enum geist_status (*matmul_f32_dense)(struct geist_backend      *be,
                                          const struct geist_tensor *x,
                                          const struct geist_tensor *w,
                                          struct geist_tensor       *y);

    /* Optional device-resident decode matvec for GGUF Q4_K weights:
     *   y[n_out] = dequant_q4_K(w[n_out, n_in]) @ x[n_in]
     * x/y are F32 DENSE, w is Q4_K BLOCK_QUANTIZED [n_out, n_in].
     * This is the narrow Q4_K_M decode path; prefill and repacked layouts are
     * separate fast paths. nullptr means unsupported. */
    enum geist_status (*matvec_q4k)(struct geist_backend      *be,
                                    const struct geist_tensor *x,
                                    const struct geist_tensor *w,
                                    struct geist_tensor       *y);
    enum geist_status (*matmul_q4k)(struct geist_backend      *be,
                                    const struct geist_tensor *x,
                                    const struct geist_tensor *w,
                                    struct geist_tensor       *y);

    /* Optional device-resident decode matvec for GGUF Q6_K weights.
     * Same shape contract as matvec_q4k, but w is Q6_K BLOCK_QUANTIZED.
     * This covers Gemma Q4_K_M's tied embedding/lm-head path. */
    enum geist_status (*matvec_q6k)(struct geist_backend      *be,
                                    const struct geist_tensor *x,
                                    const struct geist_tensor *w,
                                    struct geist_tensor       *y);
    enum geist_status (*matmul_q6k)(struct geist_backend      *be,
                                    const struct geist_tensor *x,
                                    const struct geist_tensor *w,
                                    struct geist_tensor       *y);

    /* Optional greedy sampler primitive. Returns the lowest index with the
     * largest logit. Input is F32 DENSE, either [n_vocab] or [1, n_vocab].
     * This lets GPU backends keep logits device-resident and download only
     * the selected token id. nullptr means callers map logits and use the
     * CPU sampler. */
    enum geist_status (*argmax_f32)(struct geist_backend      *be,
                                    const struct geist_tensor *logits,
                                    geist_token_t            *out_token);

    /* Batched greedy sampler primitive over F32 DENSE logits [rows, n_vocab].
     * Returns the lowest max-logit index for each row. GPU backends should
     * use this to avoid one argmax dispatch per row in verify/prefill heads.
     * nullptr means callers use argmax_f32 per row or map logits to CPU. */
    enum geist_status (*argmax_f32_batch)(
        struct geist_backend      *be,
        const struct geist_tensor *logits,
        geist_token_t              out_tokens[static logits->shape[0]]);

    /* y = x * w * rsqrt(mean(x^2) + eps). w broadcasts across feature dim.
     * All tensors are F32 DENSE. x and y can be the same tensor (in-place). */
    enum geist_status (*rmsnorm)(struct geist_backend      *be,
                                               const struct geist_tensor *x,
                                               const struct geist_tensor *w,
                                               float                      eps,
                                               struct geist_tensor       *y);

    /* y = a + b. All F32 DENSE, same shape. y can alias a or b. */
    enum geist_status (*add)(struct geist_backend      *be,
                                           const struct geist_tensor *a,
                                           const struct geist_tensor *b,
                                           struct geist_tensor       *y);

    /* y = a * b (element-wise). All F32 DENSE, same shape. */
    enum geist_status (*mul)(struct geist_backend      *be,
                                           const struct geist_tensor *a,
                                           const struct geist_tensor *b,
                                           struct geist_tensor       *y);

    /* y = x * scale. All F32 DENSE, same shape. y can alias x. */
    enum geist_status (*scale_f32)(struct geist_backend      *be,
                                   const struct geist_tensor *x,
                                   float                      scale,
                                   struct geist_tensor       *y);

    /* y = gelu_tanh(x). F32 DENSE, x and y can be the same tensor. */
    enum geist_status (*gelu_tanh)(struct geist_backend      *be,
                                                 const struct geist_tensor *x,
                                                 struct geist_tensor       *y);

    /* y = gelu_tanh(x) * z. F32 DENSE. Optional FFN fast path for GEGLU;
     * callers fall back to gelu_tanh + mul when nullptr. */
    enum geist_status (*gelu_tanh_mul)(struct geist_backend      *be,
                                       const struct geist_tensor *x,
                                       const struct geist_tensor *z,
                                       struct geist_tensor       *y);

    /* y[t,j] = gelu_tanh(x[t,j]) * z[t,j] * scale[j].
     * Optional GEGLU+AWQ fusion for transformer FFNs. scale is per-channel
     * across the last dimension. nullptr means callers use gelu_tanh_mul and
     * a separate scale pass. */
    enum geist_status (*gelu_tanh_mul_scaled)(struct geist_backend      *be,
                                              const struct geist_tensor *x,
                                              const struct geist_tensor *z,
                                              const float               *scale,
                                              struct geist_tensor       *y);

    /* y = max(x, 0)^2. F32 DENSE, x and y can be the same tensor.
     * Squared ReLU is BitNet b1.58 2B-4T's FFN activation; combining
     * the threshold + the square in one pass halves memory traffic
     * vs. relu(x) followed by mul(y, y). May be nullptr on backends
     * that don't implement it; callers must check. */
    enum geist_status (*relu_squared)(struct geist_backend      *be,
                                                    const struct geist_tensor *x,
                                                    struct geist_tensor       *y);

    /* y = silu(x) = x / (1 + exp(-x)). F32 DENSE, x and y can be the
     * same tensor. SiLU is Llama 2/3 + BitNet b1.58 3B's SwiGLU
     * activation. */
    enum geist_status (*silu)(struct geist_backend      *be,
                                            const struct geist_tensor *x,
                                            struct geist_tensor       *y);

    /* Rotary position embeddings, applied in place.
     *   x   shape [seq_len, n_heads, head_dim]   (F32 DENSE)
     *   cos shape [seq_len, head_dim]             (F32 DENSE)
     *   sin shape [seq_len, head_dim]             (F32 DENSE)
     * All shapes derived from tensor metadata. Rotates the first
     * n_rotated_dims columns of each head; n_rotated_dims is encoded as
     * cos->shape[-1] (typically == head_dim for full rotation). */
    enum geist_status (*rope_apply)(struct geist_backend      *be,
                                                  struct geist_tensor       *x,
                                                  const struct geist_tensor *cos,
                                                  const struct geist_tensor *sin);

    /* Embedding lookup: out = embed_table[token_id, :].
     *   embed_table shape [vocab_size, d_model]
     *   out         shape [d_model] (1D) or [1, d_model] (2D)
     * GPU backends may support backend-native compressed table layouts and
     * return a dequantized F32 row.
     * Returns GEIST_E_INVALID_ARG if token_id is out of range. */
    enum geist_status (*embedding_lookup)(struct geist_backend      *be,
                                                        const struct geist_tensor *embed_table,
                                                        geist_token_t              token_id,
                                                        struct geist_tensor       *out);

    /* Embedding lookup with fused scalar multiply:
     *   out = embed_table[token_id, :] * scale
     * Optional device-resident decode path for architectures that apply an
     * embedding scale before the transformer block. Backends may support
     * quantized table dtypes here even when embedding_lookup only supports
     * dense F32. nullptr means callers use embedding_lookup/dequant fallback. */
    enum geist_status (*embedding_lookup_scaled)(
        struct geist_backend      *be,
        const struct geist_tensor *embed_table,
        geist_token_t              token_id,
        float                      scale,
        struct geist_tensor       *out);

    /* Scaled dot-product attention with MQA broadcast and causal+window mask.
     *   q   shape [n_q,  n_q_heads,  head_dim]   (F32 DENSE)
     *   k   shape [n_kv, n_kv_heads, head_dim]   (F32 DENSE)
     *   v   shape [n_kv, n_kv_heads, head_dim]   (F32 DENSE)
     *   out shape [n_q,  n_q_heads,  head_dim]
     *
     *   q_offset       — position of q[0] in the absolute sequence;
     *                    causal mask permits q[t] → k[s] iff s <= q_offset + t.
     *   sliding_window — 0 = unbounded causal; >0 = additionally
     *                    s > q_offset + t - sliding_window. */
    enum geist_status (*attention)(struct geist_backend      *be,
                                                 const struct geist_tensor *q,
                                                 const struct geist_tensor *k,
                                                 const struct geist_tensor *v,
                                                 size_t                     q_offset,
                                                 size_t                     sliding_window,
                                                 struct geist_tensor       *out);

    /* Additional ops added in subsequent commits:
     *   silu_gate, ssm_step, ssm_scan, conv1d
     */

    /* ---- Optional Level-3 fast paths ---- */

    /* Device-resident GEGLU FFN block. Backends can implement this above
     * individual level-2 ops to avoid host maps and to later record/submit a
     * persistent command sequence for the whole FFN. nullptr means decomposed
     * architecture code runs. */
    enum geist_status (*ffn_geglu_block)(
        struct geist_backend                       *be,
        const struct geist_backend_ffn_geglu_block *block);

    /* Device-resident Gemma-style attention decode block. Backends can use
     * this above Level-2 ops to keep QKV, RoPE, KV append, attention, O-proj
     * and residual in one submitted command sequence. nullptr means callers
     * run the decomposed architecture path. */
    enum geist_status (*attention_block)(
        struct geist_backend                       *be,
        const struct geist_backend_attention_block *block);

    /* Device-resident Gemma-style attention decode block for KV-shared
     * layers. Backends can keep the query path, attention over existing KV,
     * O-proj and residual in one submitted command sequence. nullptr means
     * callers run the decomposed architecture path. */
    enum geist_status (*attention_query_block)(
        struct geist_backend                             *be,
        const struct geist_backend_attention_query_block *block);

    /* Device-resident Gemma PLE injection block. nullptr means callers run
     * the decomposed per-layer gate/activation/proj/norm/add sequence. */
    enum geist_status (*ple_block)(
        struct geist_backend                 *be,
        const struct geist_backend_ple_block *block);

    /* Device-resident greedy output head. Backends can fuse output RMSNorm,
     * LM-head projection, and argmax into one submitted command sequence,
     * returning only the selected token to host. nullptr means callers run
     * rmsnorm + linear + argmax/sampler separately. */
    enum geist_status (*greedy_head)(
        struct geist_backend                    *be,
        const struct geist_backend_greedy_head  *head,
        geist_token_t                           *out_token);

    /* Device-resident batched greedy output head. This is the verify/prefill
     * counterpart to greedy_head and is intended to record one batch-shaped
     * head region instead of k single-row heads. nullptr means callers use
     * greedy_head per row or the CPU-visible batched logits path. */
    enum geist_status (*greedy_head_batch)(
        struct geist_backend                          *be,
        const struct geist_backend_greedy_head_batch  *head,
        geist_token_t                                  out_tokens[static head->row_count]);

    /* Optional command-sequence capture hooks. `begin` writes an opaque token
     * owned by the backend. `end(..., submit=false)` aborts/discards the
     * sequence after an architecture-level error. Callers must not nest
     * sequences. nullptr means normal per-op submission. */
    enum geist_status (*command_sequence_begin)(
        struct geist_backend          *be,
        enum geist_command_sequence_kind kind,
        int                           *out_token);
    enum geist_status (*command_sequence_end)(
        struct geist_backend *be,
        int                   token,
        bool                  submit);
    /* Optional readback for a token produced by a captured greedy_head in the
     * most recently submitted command sequence. Needed because greedy_head's
     * synchronous out_token pointer cannot be retained while GPU work is only
     * recorded. nullptr means command-sequence capture must not include a
     * token-producing greedy_head. */
    enum geist_status (*command_sequence_read_token)(
        struct geist_backend *be,
        geist_token_t        *out_token);
    enum geist_status (*command_sequence_read_tokens)(
        struct geist_backend *be,
        size_t                n,
        geist_token_t         out_tokens[static n]);

    /* Optional replay of a previously captured single-token greedy decode
     * graph. Backends return GEIST_E_UNSUPPORTED until a compatible graph has
     * been captured and retained. On success the backend updates its dynamic
     * decode parameters for token_id/q_position, submits the retained graph,
     * and writes the selected token. */
    enum geist_status (*command_sequence_replay_decode_greedy_step)(
        struct geist_backend *be,
        geist_token_t         token_id,
        size_t                q_position,
        geist_token_t        *out_token);

    /* Experimental text-FFN fast path for Gemma-style GEGLU:
     *   y = down(gelu_tanh(gate(x)) * up(x) * optional_down_scale)
     * Backends may return GEIST_E_UNSUPPORTED when dtype/shape/layout do not
     * match their fused kernel. The caller then falls back to decomposed ops. */
    enum geist_status (*ffn_geglu_q4q6_mN)(struct geist_backend      *be,
                                           const float               *x,
                                           size_t                     m,
                                           size_t                     d_model,
                                           size_t                     inter,
                                           const struct geist_weight *gate,
                                           const struct geist_weight *up,
                                           const struct geist_weight *down,
                                           const float               *down_scale,
                                           float                     *y);

    /* Fused transformer block — backends with batched-submit (GPU) can
     * implement this to amortize dispatch overhead. nullptr means engine
     * decomposes into level-2 ops. */
    enum geist_status (*transformer_block)(struct geist_backend      *be,
                                           const struct geist_tensor *x,
                                           const void                *layer_weights,
                                           struct geist_tensor       *y);

    /* ---- Optional parallelism-regime hooks ----
     *
     * Let the arch layer ask the backend to enter a thread regime tuned for
     * an execution phase, keeping host-threading details (OpenMP, thread
     * pools) out of arch code. parallel_region_begin returns an opaque token
     * that MUST be passed back to parallel_region_end to restore the prior
     * regime; the token is 0 when nothing was changed. Backends that don't
     * manage host parallelism (e.g. GPU) leave both slots null — the arch
     * layer then runs at the ambient setting. Both null or both set. */
    int  (*parallel_region_begin)(struct geist_backend       *be,
                                  enum geist_parallel_region  region);
    void (*parallel_region_end)(struct geist_backend *be, int token);
};

/* ====================================================================== */
/* Backend Descriptor                                                      */
/* ====================================================================== */

/* Each backend exports one of these as a `const` extern. The engine's
 * registry array points at descriptors of compiled-in backends. */
struct geist_backend_descriptor {
    const char *name;

    /* Vtable function pointers. */
    const struct geist_backend_vtbl *vtbl;

    /* Capability matrix — pointer to array of n_caps queries this backend
     * supports natively (or via emulation). May be nullptr if the backend
     * answers all capability queries dynamically via vtbl->supports_op. */
    const struct geist_op_support_query *caps;
    size_t                               n_caps;
};

/* ====================================================================== */
/* Engine-Side Internals Visible to Backends                               */
/* ====================================================================== */

/* The full struct geist_backend definition. Backends need read access to
 * .alloc (for routing internal allocations through the user-provided
 * allocator) and to the error slot (for setting detailed messages). */
struct geist_backend {
    const struct geist_backend_descriptor *desc;
    struct geist_allocator                 alloc;

    /* Backend-private state, set during create(). */
    void *state;

    /* Error slot — set via geist_backend_set_error*. */
    enum geist_status err_code;
    char              err_msg[512];
};

/* Helpers backends call to record an error. */
void geist_backend_set_error(struct geist_backend *be, enum geist_status code,
                             const char *fmt, ...);

/* Allocator convenience: route a backend allocation through be->alloc. */
[[nodiscard]] void *geist_backend_alloc(struct geist_backend *be, size_t bytes,
                                        size_t alignment);
void                geist_backend_free(struct geist_backend *be, void *ptr);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* GEIST_BACKEND_H */
