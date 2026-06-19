/*
 * test_heap_unit — heap.h allocator + memory_arena safety contract.
 *
 * heap.c is the project-wide allocation interface (AGENT.md). These cases
 * lock down the safety guarantees that the engine and the GGUF loaders rely
 * on, in particular the ones added to stop silent integer-overflow
 * under-allocation:
 *
 *   - zero-size / empty inputs return nullptr (well-defined, no alloc)
 *   - minimal valid allocations succeed, are aligned, and are writable
 *   - invalid (non-power-of-2) alignment is refused, not fed to the mask
 *   - size_t-overflowing sizes are refused, not wrapped to a tiny buffer
 *   - heap_calloc_aligned refuses count*size overflow
 *   - arena: alloc / alignment / capacity exhaustion / overflow / reset
 *   - try_create_memory_arena fails cleanly on null arg / overflow size
 *   - create_memory_arena returns a null arena on failure (does NOT exit)
 *   - safe_free tolerates null and nulls the caller's pointer
 *
 * Deterministic, allocation-bounded (the overflow cases never reach
 * aligned_alloc — the guards return before that), ASan/UBSan friendly.
 */
#include "test_helpers.h"

#include "heap.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define CHECK(cond, msg)                          \
    do {                                          \
        if (!(cond)) {                            \
            fprintf(stderr, "FAIL: %s\n", (msg)); \
            return GEIST_TEST_FAIL;               \
        }                                         \
    } while (0)

int main(void) {
    /* ---- heap_alloc_aligned -------------------------------------------- */

    /* Zero size: well-defined nullptr, no allocation. */
    CHECK(heap_alloc_aligned(0, 64) == nullptr, "alloc(0) should be null");

    /* Minimal valid: aligned to at least OPTIMAL_ALIGNMENT and writable. */
    void *p = heap_alloc_aligned(1, alignof(float));
    CHECK(p != nullptr, "alloc(1) should succeed");
    CHECK(((uintptr_t) p & (OPTIMAL_ALIGNMENT - 1)) == 0, "alloc not OPTIMAL_ALIGNMENT-aligned");
    memset(p, 0xAB, 1); /* writable; ASan would catch under-allocation */
    safe_free(&p);
    CHECK(p == nullptr, "safe_free should null the pointer");

    /* Explicit large power-of-2 alignment is honored. */
    void *pa = heap_alloc_aligned(32, 128);
    CHECK(pa != nullptr, "alloc with align=128 should succeed");
    CHECK(((uintptr_t) pa & 127) == 0, "align=128 not honored");
    safe_free(&pa);

    /* Non-power-of-2 alignment must be refused (would corrupt the mask /
     * is UB for aligned_alloc). */
    CHECK(heap_alloc_aligned(64, 24) == nullptr, "non-pow2 align must be null");
    CHECK(heap_alloc_aligned(64, 96) == nullptr, "non-pow2 align>OPTIMAL must be null");

    /* Overflow: rounding SIZE_MAX up to alignment would wrap to a tiny
     * value. Must be refused, not under-allocate. */
    CHECK(heap_alloc_aligned(SIZE_MAX, 64) == nullptr, "SIZE_MAX alloc must be refused");
    CHECK(heap_alloc_aligned(SIZE_MAX - 16, 64) == nullptr, "near-SIZE_MAX alloc must be refused");

    /* ---- heap_calloc_aligned ------------------------------------------- */

    CHECK(heap_calloc_aligned(0, 4, 64) == nullptr, "calloc(0,*) null");
    CHECK(heap_calloc_aligned(4, 0, 64) == nullptr, "calloc(*,0) null");

    /* count*size overflow must be refused. */
    CHECK(heap_calloc_aligned(SIZE_MAX, 2, 64) == nullptr, "calloc overflow must be refused");

    int32_t *zeros = heap_calloc_array_aligned(int32_t, 16);
    CHECK(zeros != nullptr, "calloc(16 int32) should succeed");
    for (int i = 0; i < 16; i++) {
        CHECK(zeros[i] == 0, "calloc memory not zeroed");
    }
    safe_free((void **) &zeros);

    /* ---- memory_arena: create / try-create ----------------------------- */

    /* Null arena pointer: clean false, no crash. */
    CHECK(!try_create_memory_arena(nullptr, 1024), "null arena -> false");

    /* Overflow size: refused, no allocation, fields zeroed. */
    struct memory_arena bad = {.memory = (void *) 0x1, .size = 1, .used = 1};
    CHECK(!try_create_memory_arena(&bad, SIZE_MAX), "overflow size -> false");
    CHECK(bad.memory == nullptr && bad.size == 0 && bad.used == 0,
          "failed try-create must zero the arena");

    /* create_memory_arena must NOT exit() on failure — it returns a null
     * arena that the caller can detect. */
    struct memory_arena failed = create_memory_arena(SIZE_MAX);
    CHECK(failed.memory == nullptr, "create on overflow must return null arena");

    /* ---- memory_arena: normal use -------------------------------------- */

    struct memory_arena arena = create_memory_arena(4096);
    CHECK(arena.memory != nullptr, "4K arena should allocate");
    CHECK(arena.used == 0, "fresh arena used should be 0");

    void *a1 = arena_allocate_aligned(&arena, 100, alignof(float));
    void *a2 = arena_allocate_aligned(&arena, 100, alignof(float));
    CHECK(a1 != nullptr && a2 != nullptr, "small arena allocs should succeed");
    CHECK(((uintptr_t) a1 & (OPTIMAL_ALIGNMENT - 1)) == 0, "arena alloc not OPTIMAL-aligned");
    CHECK((uintptr_t) a2 > (uintptr_t) a1, "arena allocs must advance");

    /* Non-pow2 alignment refused. */
    CHECK(arena_allocate_aligned(&arena, 16, 48) == nullptr, "arena non-pow2 align must be null");

    /* Overflow size refused (must not wrap past the bounds check). */
    CHECK(arena_allocate_aligned(&arena, SIZE_MAX, 64) == nullptr,
          "arena SIZE_MAX alloc must be refused");

    /* Capacity exhaustion: ask for more than remains -> null, no overflow. */
    CHECK(arena_allocate_aligned(&arena, 8192, 64) == nullptr,
          "arena over-capacity alloc must be null");

    /* Null / no-memory arena handled. */
    CHECK(arena_allocate_aligned(nullptr, 16, 16) == nullptr, "null arena alloc");

    /* Reset (used := 0) reuses the base address. */
    size_t used_before = arena.used;
    CHECK(used_before > 0, "arena should have advanced");
    arena.used     = 0;
    void *a1_again = arena_allocate_aligned(&arena, 100, alignof(float));
    CHECK(a1_again == a1, "arena reset should reuse the base");

    free_memory_arena(&arena);
    CHECK(arena.memory == nullptr && arena.size == 0 && arena.used == 0,
          "free_memory_arena must zero the arena");

    /* free_memory_arena is idempotent / null-safe. */
    free_memory_arena(&arena);
    free_memory_arena(nullptr);

    /* safe_free null tolerance. */
    void *nullp = nullptr;
    safe_free(&nullp);
    safe_free(nullptr);

    printf("PASS: heap alloc/calloc/arena — zero/overflow/alignment/capacity guards\n");
    return GEIST_TEST_PASS;
}
