/*
 * src/engine/allocator.c — libc allocator + arena, both routed through
 * the project-wide heap.h interface (per AGENT.md).
 *
 * Layer: ENGINE.
 *
 * Default allocator: heap_alloc_aligned + safe_free wrappers.
 * Arena allocator: thin wrapper around heap.h's struct memory_arena.
 */
#define GEIST_INTERNAL_ENGINE_LAYER

#include "allocator.h"

#include "heap.h"

#include <stddef.h>
#include <stdint.h>

/* ====================================================================== */
/* libc-backed allocator (default when caller passes nullptr to *_create) */
/* ====================================================================== */

static void *libc_alloc(void *ctx, size_t bytes, size_t alignment) {
    (void) ctx;
    if (bytes == 0) {
        return nullptr;
    }
    /* heap.h enforces power-of-2 alignment >= 8 internally; route through. */
    return heap_alloc_aligned(bytes, alignment > 0 ? alignment : OPTIMAL_ALIGNMENT);
}

static void libc_free(void *ctx, void *ptr) {
    (void) ctx;
    if (ptr == nullptr) {
        return;
    }
    safe_free(&ptr);
}

const struct geist_allocator geist_libc_allocator = {
    .alloc    = libc_alloc,
    .free     = libc_free,
    .free_all = nullptr, /* libc has no free-all; arena does */
    .ctx      = nullptr,
};

/* ====================================================================== */
/* Arena allocator                                                         */
/* ====================================================================== */

struct geist_arena {
    struct memory_arena         arena;   /* heap.h's underlying arena */
    const struct geist_allocator *parent; /* allocator that owns the backing buffer */
};

struct geist_arena *geist_arena_create(size_t                          initial_bytes,
                                       const struct geist_allocator   *parent) {
    if (parent == nullptr) {
        parent = &geist_libc_allocator;
    }

    struct geist_arena *a = parent->alloc(parent->ctx, sizeof(*a), alignof(struct geist_arena));
    if (a == nullptr) {
        return nullptr;
    }

    if (!try_create_memory_arena(&a->arena, initial_bytes)) {
        parent->free(parent->ctx, a);
        return nullptr;
    }

    a->parent = parent;
    return a;
}

void geist_arena_destroy(struct geist_arena *a) {
    if (a == nullptr) {
        return;
    }
    free_memory_arena(&a->arena);
    a->parent->free(a->parent->ctx, a);
}

void geist_arena_reset(struct geist_arena *a) {
    if (a == nullptr) {
        return;
    }
    a->arena.used = 0;
}

void *geist_arena_alloc(struct geist_arena *a, size_t bytes, size_t alignment) {
    if (a == nullptr || bytes == 0) {
        return nullptr;
    }
    if (alignment == 0) {
        alignment = OPTIMAL_ALIGNMENT;
    }
    return arena_allocate_aligned(&a->arena, bytes, alignment);
}

/* ---- Wrap an arena as a geist_allocator ---- */

static void *arena_as_alloc_alloc(void *ctx, size_t bytes, size_t alignment) {
    return geist_arena_alloc((struct geist_arena *) ctx, bytes, alignment);
}

static void arena_as_alloc_free(void *ctx, void *ptr) {
    /* Bump-pointer arenas don't free individual allocations. */
    (void) ctx;
    (void) ptr;
}

static void arena_as_alloc_free_all(void *ctx) {
    geist_arena_reset((struct geist_arena *) ctx);
}

struct geist_allocator geist_arena_as_allocator(struct geist_arena *a) {
    return (struct geist_allocator){
        .alloc    = arena_as_alloc_alloc,
        .free     = arena_as_alloc_free,
        .free_all = arena_as_alloc_free_all,
        .ctx      = a,
    };
}
