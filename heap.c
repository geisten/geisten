//
// Created by germar on 09.03.25.
//
#include "heap.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

// Ensure alignment is a power of 2 (e.g., 16, 32, 64)
uintptr_t aligned_size(const size_t size, const size_t alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
}

uintptr_t optimal_aligned_size(const size_t size) {
    return aligned_size(size, OPTIMAL_ALIGNMENT);
}

/**
 * Creates a memory arena with the specified size.
 *
 * The arena allocator is a simple and efficient way to manage memory for
 * short-lived allocations. Instead of calling malloc/free for each small
 * allocation, we allocate a large block upfront and distribute small chunks
 * from it.
 *
 * @param size Size of the arena in bytes
 * @return Initialized memory arena structure
 */
struct memory_arena create_memory_arena(const size_t size) {
    struct memory_arena arena = {};
    if (!try_create_memory_arena(&arena, size)) {
        perror("Failed to allocate memory arena");
        exit(EXIT_FAILURE);
    }
    return arena;
}

bool try_create_memory_arena(struct memory_arena* arena, const size_t size) {
    if (!arena) {
        return false;
    }
    /* Make sure the size is a multiple of the optimal alignment */
    const size_t aligned_size = optimal_aligned_size(size);

    /* Allocate memory with optimal alignment for best performance */
    void* memory = aligned_alloc(OPTIMAL_ALIGNMENT, aligned_size);
    if (!memory) {
        arena->memory = nullptr;
        arena->size = 0;
        arena->used = 0;
        return false;
    }

    /* Initialize the arena structure */
    arena->memory = memory;
    arena->size = aligned_size;
    arena->used = 0;
    return true;
}

/**
 * Allocates aligned memory from the arena.
 *
 * This function returns a pointer to a block of memory with the specified size
 * and alignment. The memory comes from the arena's pre-allocated block. This is
 * much faster than calling malloc for many small allocations.
 *
 * @param arena Pointer to the memory arena
 * @param size Number of bytes to allocate
 * @param alignment Required alignment (must be power of 2)
 * @return Pointer to allocated memory or nullptr if allocation fails
 */
void* arena_allocate_aligned(struct memory_arena* arena, size_t size, size_t alignment) {
    if (!arena || !arena->memory) {
        return nullptr;
    }

    /* Verwende mindestens OPTIMAL_ALIGNMENT */
    if (alignment < OPTIMAL_ALIGNMENT) {
        alignment = OPTIMAL_ALIGNMENT;
    }

    /* Calculate aligned address */
    const uintptr_t current = (uintptr_t) arena->memory + arena->used;
    const uintptr_t aligned = aligned_size(current, alignment);
    const size_t offset = aligned - (uintptr_t) arena->memory;

    /* Round size up to a multiple of alignment for better data locality */
    const size_t aligned_size_value = aligned_size(size, alignment);

    /* Check if we have enough space */
    if (offset + aligned_size_value > arena->size) {
        fprintf(stderr,
                "Memory out of bounds (old: %zu + object size %zu > total: %zu)\n",
                offset,
                aligned_size_value,
                arena->size);
        return nullptr;
    }

    /* Update used space */
    arena->used = offset + aligned_size_value;

    /* Return aligned pointer */
    return (void*) aligned;
}

void* heap_alloc_aligned(const size_t size, size_t alignment) {
    void* memory = nullptr;
    size_t aligned = 0;

    if (size == 0u) {
        return nullptr;
    }
    if (alignment < OPTIMAL_ALIGNMENT) {
        alignment = OPTIMAL_ALIGNMENT;
    }
    aligned = aligned_size(size, alignment);
    memory = aligned_alloc(alignment, aligned);
    return memory;
}

void* heap_calloc_aligned(const size_t count, const size_t size, const size_t alignment) {
    void* memory = nullptr;
    if (count == 0u || size == 0u) {
        return nullptr;
    }
    if (count > (SIZE_MAX / size)) {
        return nullptr;
    }
    memory = heap_alloc_aligned(count * size, alignment);
    if (!memory) {
        return nullptr;
    }
    memset(memory, 0, count * size);
    return memory;
}

/**
 * safe_free - a safer way to free dynamically allocated memory
 * @ptr: pointer to memory location
 *
 * Description: This safe_free() function takes care of freeing
 * dynamically allocated memory while ensuring the pointer
 * @ptr passed to it is not nullptr before trying to free it.
 * Also, after freeing the memory, it sets the pointer @ptr
 * to nullptr to avoid the issue of dangling pointers
 */
void safe_free(void** ptr) {
    if (ptr != nullptr && *ptr != nullptr) {
        free(*ptr);
        *ptr = nullptr;
    }
}

/**
 * Frees the entire memory arena.
 *
 * This function releases all memory allocated for the arena. After calling this
 * function, any pointers obtained from arena_allocate or arena_allocate_aligned
 * are invalid.
 *
 * @param arena Pointer to the memory arena
 */
void free_memory_arena(struct memory_arena* arena) {
    if (arena && arena->memory) {
        safe_free(&arena->memory);
        arena->memory = nullptr;
        arena->size = 0;
        arena->used = 0;
    }
}
