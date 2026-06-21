/*
 * src/backends/cpu_x86/backend.c — x86_64 backend, Phase 0 shell.
 *
 * Layer: BACKEND.
 *
 * Today this descriptor reuses cpu_scalar's vtbl wholesale: cpu_x86 binds
 * and runs, but every op routes through scalar code. Phase 1a/1b/2 (see
 * docs/LINUX_X86_SPEC.md) replace individual vtbl entries with native
 * AVX-512 / VNNI / BF16 kernels. The kernel TUs live in this directory,
 * keyed off the per-ISA `-march=` flags in mk/backend-cpu_x86.mk.
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "backend.h"

#include <geist_backend.h>

/* Borrowed from cpu_scalar/backend.c — exported there for this purpose. */
extern const struct geist_backend_vtbl cpu_scalar_vtbl;

const struct geist_backend_descriptor geist_backend_cpu_x86 = {
        .name   = "cpu_x86",
        .vtbl   = &cpu_scalar_vtbl, /* ponytail: scalar passthrough until Phase 1a */
        .caps   = nullptr,          /* dynamic via supports_op */
        .n_caps = 0,
};
