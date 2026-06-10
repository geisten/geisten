/*
 * src/engine/allocator.h — internal arena allocator + libc wrapper.
 *
 * Layer: ENGINE.
 *
 * Defined in (Phase B-4):
 *   src/engine/allocator.c
 *
 * The libc wrapper is also exposed publicly via geist_libc_allocator
 * (see <geist.h>). The arena type is internal — backends use it for
 * scratch pools without exposing it across the layer boundary.
 */
#ifndef GEIST_INTERNAL_ALLOCATOR_H
#define GEIST_INTERNAL_ALLOCATOR_H

#ifndef GEIST_INTERNAL_ENGINE_LAYER
#error "allocator.h is internal to the engine layer."
#endif

#include <geist.h>

/* Arena: bump-pointer allocator for short-lived activation buffers. */
struct geist_arena;

struct geist_arena *geist_arena_create(size_t                          initial_bytes,
                                       const struct geist_allocator   *parent);
void                geist_arena_destroy(struct geist_arena *a);

/* Reset the bump pointer; previous allocations become invalid. The
 * underlying buffer is retained for re-use (no malloc/free churn). */
void  geist_arena_reset(struct geist_arena *a);
void *geist_arena_alloc(struct geist_arena *a, size_t bytes, size_t alignment);

/* Wrap an arena as a geist_allocator interface so it can be passed
 * everywhere a struct geist_allocator is expected. */
struct geist_allocator geist_arena_as_allocator(struct geist_arena *a);

#endif /* GEIST_INTERNAL_ALLOCATOR_H */
