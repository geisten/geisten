/*
 * src/archs/audio_conformer/arch.h — audio Conformer encoder (Gemma 4).
 *
 * Layer: ARCHITECTURE. Implements the geist_arch_ops_encoder vtable for
 * the Gemma 4 audio tower (Conformer-based, stateless forward).
 *
 * The vtable shape itself lives in <geist_arch.h> (engine-owned interface);
 * this header only exports the concrete descriptor.
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
#include <geist_arch.h>

/* Concrete descriptor for the Gemma 4 audio Conformer. */
extern const struct geist_arch_ops_encoder geist_arch_audio_conformer;

#endif /* GEIST_INTERNAL_ARCH_AUDIO_CONFORMER_H */
