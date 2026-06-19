/*
 * src/archs/transformer/forward/profile.h - private forward profiler.
 */
#pragma once

#ifndef GEIST_INTERNAL_ARCH_LAYER
#error "forward/profile.h is a private architecture-layer header"
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

struct transformer_forward_profile {
    const char        *title;
    const char *const *stage_names;
    size_t             stage_count;
    uint64_t          *ns;
    uint64_t          *calls;
    bool               registered;
};

bool     transformer_profile_enabled(struct transformer_forward_profile *profile);
uint64_t transformer_profile_now_ns(void);
void     transformer_profile_add(struct transformer_forward_profile *profile,
                                 size_t                              stage,
                                 uint64_t                            t0);
