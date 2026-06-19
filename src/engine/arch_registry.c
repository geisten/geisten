/*
 * src/engine/arch_registry.c — compiled-in architecture list.
 *
 * Layer: ENGINE. Mirror of backend_registry.c — each architecture is
 * gated by GEIST_ARCH_<NAME> at compile time; the registry is NULL-
 * terminated and ordered by GGUF-name match preference.
 *
 * Adding a new architecture:
 *   1. Implement src/archs/<name>/arch.c exporting
 *      'extern const struct geist_arch_ops_decoder geist_arch_<name>'.
 *   2. Add mk/arch-<name>.mk setting SRCS_ARCH += src/archs/<name>/...c
 *   3. Add #if GEIST_ARCH_<NAME> block below.
 *   4. Build with `make ARCHS="transformer <name>"`.
 */
#define GEIST_INTERNAL_ENGINE_LAYER

#include "arch_registry.h"

#define GEIST_INTERNAL_ARCH_LAYER
#include "../archs/audio_conformer/arch.h"
#include "../archs/transformer/arch.h"
#include "../archs/vision_siglip/arch.h"
#undef GEIST_INTERNAL_ARCH_LAYER

#include <string.h>

/* Match a GGUF general.architecture value against arch names. Currently
 * matches by exact string. Gemma 4 uses "gemma3" (a holdover from the
 * v3 codebase — Gemma 4 weights are sometimes labelled gemma3 in the
 * GGUF; both map to the transformer decoder). */
static bool arch_matches(const char *arch_name, const char *gguf_arch) {
    if (arch_name == nullptr || gguf_arch == nullptr) {
        return false;
    }
    if (strcmp(arch_name, gguf_arch) == 0) {
        return true;
    }
    /* The transformer descriptor handles a few Gemma generations. */
    if (strcmp(arch_name, "transformer") == 0) {
        if (strncmp(gguf_arch, "gemma", 5) == 0)
            return true;
        if (strncmp(gguf_arch, "llama", 5) == 0)
            return true;
        if (strncmp(gguf_arch, "mistral", 7) == 0)
            return true;
    }
    return false;
}

/* Compiled-in list. GEIST_ARCH_TRANSFORMER is defined by default since
 * the only model the engine currently supports is Gemma 4. */
#define GEIST_ARCH_TRANSFORMER 1

#if GEIST_ARCH_TRANSFORMER
static const struct geist_arch_descriptor desc_transformer = {
        .name               = "transformer",
        .decoder_ops        = &geist_arch_transformer,
        .audio_encoder_ops  = &geist_arch_audio_conformer,
        .vision_encoder_ops = &geist_arch_vision_siglip,
};
#endif

const struct geist_arch_descriptor *const geist_arch_registry[] = {
#if GEIST_ARCH_TRANSFORMER
        &desc_transformer,
#endif
        nullptr,
};

const struct geist_arch_descriptor *geist_arch_registry_lookup(const char *gguf_arch) {
    for (size_t i = 0; geist_arch_registry[i] != nullptr; i++) {
        if (arch_matches(geist_arch_registry[i]->name, gguf_arch)) {
            return geist_arch_registry[i];
        }
    }
    /* Fallback: first registered arch (transformer for Gemma 4 builds). */
    return geist_arch_registry[0];
}
