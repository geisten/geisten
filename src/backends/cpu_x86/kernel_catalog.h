/*
 * src/backends/cpu_x86/kernel_catalog.h - x86 CPU kernel policy skeleton.
 *
 * Layer: BACKEND (cpu_x86, internal).
 *
 * This is the x86 analogue of cpu_neon's kernel catalog. It deliberately
 * contains no intrinsics and can be compiled on non-x86 hosts; runtime
 * feature facts come from geist_hw_probe.
 */
#ifndef GEIST_INTERNAL_BACKEND_CPU_X86_KERNEL_CATALOG_H
#define GEIST_INTERNAL_BACKEND_CPU_X86_KERNEL_CATALOG_H

#ifndef GEIST_INTERNAL_BACKEND_LAYER
#error "cpu_x86/kernel_catalog.h is internal to the backend layer."
#endif

#include "hw_probe.h"

#include <geist.h>

#include <stdbool.h>
#include <stdint.h>

typedef uint32_t cpu_x86_isa_mask;
enum {
    CPU_X86_ISA_SSE2       = 1u << 0,
    CPU_X86_ISA_AVX2       = 1u << 1,
    CPU_X86_ISA_AVX512F    = 1u << 2,
    CPU_X86_ISA_AVX512VNNI = 1u << 3,
    CPU_X86_ISA_AMX_INT8   = 1u << 4,
};

struct cpu_x86_kernel_policy {
    cpu_x86_isa_mask isa;
    bool prefer_vnni_i8;
    bool prefer_amx_prefill;
    bool use_packed_weights;
};

cpu_x86_isa_mask cpu_x86_isa_from_probe(const struct geist_hw_probe *hw);
const char *cpu_x86_isa_mask_name(cpu_x86_isa_mask mask);
struct cpu_x86_kernel_policy cpu_x86_kernel_policy_default(
    const struct geist_hw_probe *hw);

#endif /* GEIST_INTERNAL_BACKEND_CPU_X86_KERNEL_CATALOG_H */
