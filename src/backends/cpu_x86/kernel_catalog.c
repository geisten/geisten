/*
 * src/backends/cpu_x86/kernel_catalog.c - x86 CPU kernel policy skeleton.
 */
#define GEIST_INTERNAL_BACKEND_LAYER

#include "kernel_catalog.h"

#include <stdio.h>

cpu_x86_isa_mask cpu_x86_isa_from_probe(const struct geist_hw_probe *hw) {
    if (hw == nullptr || hw->cpu != GEIST_HW_CPU_X86_64_GENERIC) {
        return 0;
    }
    cpu_x86_isa_mask m = CPU_X86_ISA_SSE2;
    if (hw->has_avx2) {
        m |= CPU_X86_ISA_AVX2;
    }
    if (hw->has_avx512f) {
        m |= CPU_X86_ISA_AVX512F;
    }
    if (hw->has_avx512_vnni) {
        m |= CPU_X86_ISA_AVX512VNNI;
    }
    if (hw->has_amx_int8) {
        m |= CPU_X86_ISA_AMX_INT8;
    }
    return m;
}

const char *cpu_x86_isa_mask_name(cpu_x86_isa_mask mask) {
    static char buf[96];
    int n = snprintf(buf,
                     sizeof(buf),
                     "{%s%s%s%s%s%s%s%s%s}",
                     (mask & CPU_X86_ISA_SSE2) ? "SSE2" : "",
                     ((mask & CPU_X86_ISA_SSE2) && (mask & ~CPU_X86_ISA_SSE2)) ? "," : "",
                     (mask & CPU_X86_ISA_AVX2) ? "AVX2" : "",
                     ((mask & CPU_X86_ISA_AVX2) && (mask & ~(CPU_X86_ISA_SSE2 | CPU_X86_ISA_AVX2)))
                             ? ","
                             : "",
                     (mask & CPU_X86_ISA_AVX512F) ? "AVX512F" : "",
                     ((mask & CPU_X86_ISA_AVX512F) &&
                      (mask & (CPU_X86_ISA_AVX512VNNI | CPU_X86_ISA_AMX_INT8)))
                             ? ","
                             : "",
                     (mask & CPU_X86_ISA_AVX512VNNI) ? "AVX512VNNI" : "",
                     ((mask & CPU_X86_ISA_AVX512VNNI) && (mask & CPU_X86_ISA_AMX_INT8)) ? "," : "",
                     (mask & CPU_X86_ISA_AMX_INT8) ? "AMX_INT8" : "");
    (void) n;
    return buf;
}

struct cpu_x86_kernel_policy cpu_x86_kernel_policy_default(const struct geist_hw_probe *hw) {

    const cpu_x86_isa_mask isa = cpu_x86_isa_from_probe(hw);
    return (struct cpu_x86_kernel_policy) {
            .isa                = isa,
            .prefer_vnni_i8     = (isa & CPU_X86_ISA_AVX512VNNI) != 0,
            .prefer_amx_prefill = (isa & CPU_X86_ISA_AMX_INT8) != 0,
            .use_packed_weights =
                    (isa & (CPU_X86_ISA_AVX2 | CPU_X86_ISA_AVX512VNNI | CPU_X86_ISA_AMX_INT8)) != 0,
    };
}
