/*
 * src/archs/transformer/arch.h — transformer decoder architecture.
 *
 * Layer: ARCHITECTURE. Implements the geist_arch_ops_decoder vtable for
 * Gemma 4 / Llama / Mistral / similar transformer-style decoders.
 *
 * The vtable shape itself lives in <geist_arch.h> (engine-owned interface);
 * this header only exports the concrete descriptor.
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
#include <geist_arch.h>

/* Concrete descriptor for the transformer decoder. */
extern const struct geist_arch_ops_decoder geist_arch_transformer;

#endif /* GEIST_INTERNAL_ARCH_TRANSFORMER_H */
