/*
 * src/archs/transformer/arch.h — transformer decoder architecture.
 *
 * Layer: ARCHITECTURE. Implements the geist_arch_ops_decoder vtable for
 * Gemma 4 / Llama / Mistral / similar transformer-style decoders.
 *
 * Defined in (Phase B-4):
 *   src/archs/transformer/arch.c              — descriptor, state lifecycle
 *   src/archs/transformer/attention_loop.c    — per-layer attention sequence
 *   src/archs/transformer/ffn_loop.c          — per-layer FFN sequence
 *   src/archs/transformer/kv_cache.c          — KV-cache layout, pin_prefix
 *   src/archs/transformer/rope.c              — RoPE helpers
 */
#ifndef GEIST_INTERNAL_ARCH_TRANSFORMER_H
#define GEIST_INTERNAL_ARCH_TRANSFORMER_H

#ifndef GEIST_INTERNAL_ARCH_LAYER
#error "transformer/arch.h is internal to the architecture layer."
#endif

#include <geist.h>

/* Decoder arch_ops vtable — what every decoder-arch must implement.
 *
 * The vtable operates on opaque `void *arch_state` rather than a
 * `struct geist_session` to keep arch implementations decoupled from
 * the engine's full session definition. The engine extracts what each
 * call needs (model + backend + opts at create time, tokens at runtime)
 * and passes only those.
 *
 * arch_state holds the architecture-specific recurrent state — for a
 * transformer this means the LM + KV cache; for Mamba it means the
 * SSM state vector. */
struct geist_arch_ops_decoder {
    const char *name;

    /* state_create: allocate arch_state on backend. Returns nullptr on
     * failure (engine reports OOM/IO via error path). Caller passes
     * the GGUF path that model_load used to find the file. opts is
     * typically nullptr at model-load time (no session yet); session
     * options arrive later via set_session_opts. */
    void *(*state_create)(struct geist_backend            *be,
                          const char                      *gguf_path,
                          const struct geist_session_opts *opts);

    /* state_create_from_memory: like state_create but the GGUF is already in
     * memory (e.g. embedded in the binary). The buffer is aliased read-only and
     * must outlive the arch_state. No aux files (tokenizer.bin / vision / audio
     * safetensors) are searched — text-only with the GGUF-embedded tokenizer.
     * nullptr if the arch does not support memory loading. */
    void *(*state_create_from_memory)(struct geist_backend            *be,
                                      const void                      *data,
                                      size_t                           size,
                                      const struct geist_session_opts *opts);

    /* state_destroy: tear down arch_state. nullptr is a no-op. */
    void (*state_destroy)(void *arch_state);

    /* Optional: push session opts into arch_state. Engine calls this
     * from geist_session_create so per-session sampler config (temperature,
     * top_p, top_k, random_seed) reaches the decode hot path. nullptr
     * means the architecture ignores session opts (greedy-only). */
    void (*set_session_opts)(void *arch_state, const struct geist_session_opts *opts);

    /* state_reset: drop conversational state (KV / SSM hidden), keep
     * weights. Used by geist_session_reset. */
    void (*state_reset)(void *arch_state);

    /* prefill: append `n` tokens to the recurrent state. */
    void (*prefill)(void *arch_state, size_t n, const geist_token_t ids[static n]);

    /* decode_step: one greedy autoregressive step, returns next token. */
    geist_token_t (*decode_step)(void *arch_state);

    /* Optional: pin prefix into KV cache so reset() restores to it
     * instead of clearing. nullptr if architecture doesn't support it. */
    void (*pin_prefix)(void *arch_state, size_t n, const geist_token_t ids[static n]);

    /* Optional: append audio soft-tokens (1536-dim per token for Gemma 4)
     * to the recurrent state. nullptr if no audio path. */
    void (*prefill_audio)(void *arch_state, size_t n, const float *soft_tokens);

    /* Optional: append vision soft-tokens (1536-dim per token for Gemma 4)
     * to the recurrent state. Same wire format as prefill_audio — both
     * modalities feed d_model-dim floats into the residual stream — so
     * the transformer impl is shared. nullptr if no vision path. */
    void (*prefill_image)(void *arch_state, size_t n, const float *soft_tokens);

    /* Optional: pointer to the cached next-token logits. Writes the vocab
     * size to `*n_logits` on success. Returns nullptr (and sets *n_logits=0)
     * if logits aren't materialized yet. Pointer is valid until the next
     * mutating call. CPU-only contract — GPU backends that need a copy
     * should populate this via a session-owned scratch buffer. */
    const float *(*peek_logits)(void *arch_state, size_t *n_logits);

    /* Speculative-decode primitives. Optional — leave nullptr if the
     * architecture has no batched verify path or no truncatable cache.
     * When any of these is nullptr, geist_session_decode_speculative
     * falls back to sequential decode_step.
     *
     * peek_next_token: the architecture's already-computed argmax for the
     *   immediate next position, or -1 if no valid logits are pending.
     *   "Free" — must not run a forward pass.
     * verify_forward: feed k candidate tokens through the full stack,
     *   advance kv_len by k, write k per-position samples to out_tokens.
     * kv_truncate: shrink recurrent state to new_len. Subsequent prefill
     *   overwrites from new_len onwards.
     * kv_len: current recurrent-state length (positions filled). */
    geist_token_t (*peek_next_token)(void *arch_state);
    enum geist_status (*verify_forward)(void               *arch_state,
                                        size_t              k,
                                        const geist_token_t ids[static k],
                                        geist_token_t       out_tokens[static k]);
    void (*kv_truncate)(void *arch_state, size_t new_len);
    size_t (*kv_len)(const void *arch_state);

    /* Multi-session lifecycle (P1.2.f). Each engine-level geist_session
     * may own its own per-session arch state (KV cache, scratch pool,
     * sampler RNG, etc.); the model owns the immutable weight set.
     *
     * session_alloc: allocate a fresh per-session arch state on the
     *   model. Returns the opaque session_meta or nullptr on OOM.
     * session_free: tear down a previously-allocated session_meta.
     * session_attach: install session_meta as the active session on
     *   the model state. Subsequent vtable calls (prefill, decode_step,
     *   verify_forward, ...) operate on this session's KV/scratch.
     *   nullptr re-installs the model's default session. The engine
     *   re-attaches before each dispatch — single-active by design.
     *
     * All three nullptr → architecture stays single-session-per-model
     * (engine falls back to using the model's default session). */
    void *(*session_alloc)(void *arch_state, const struct geist_session_opts *opts);
    void (*session_free)(void *arch_state, void *session_meta);
    void (*session_attach)(void *arch_state, void *session_meta);
};

/* Concrete descriptor for the transformer decoder. */
extern const struct geist_arch_ops_decoder geist_arch_transformer;

#endif /* GEIST_INTERNAL_ARCH_TRANSFORMER_H */
