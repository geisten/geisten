# mk/backend-cpu_x86.mk — x86 ISA policy catalog (AVX2 / AVX-512 dispatch
# scaffolding).
#
# NOT yet a registered backend: backend_registry.c carries no
# geist_backend_cpu_x86 descriptor, and nothing references its symbols. It is
# gated here so it compiles only when explicitly requested via
# `make BACKENDS="... cpu_x86"`, keeping the dead scaffolding out of the
# default CPU builds until a real x86 backend is wired up.

BACKEND_SOURCES += \
    src/backends/cpu_x86/kernel_catalog.c
