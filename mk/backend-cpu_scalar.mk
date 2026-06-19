# mk/backend-cpu_scalar.mk — portable scalar reference backend.
#
# Enabled via `make BACKENDS="... cpu_scalar"` (the default fallback backend
# on every target). Registered at runtime in src/engine/backend_registry.c
# under GEIST_BACKEND_CPU_SCALAR. Has no ISA dependencies, so it builds and
# runs anywhere — the correctness baseline the other backends are checked
# against.

BACKEND_SOURCES += \
    src/backends/cpu_scalar/backend.c \
    src/backends/cpu_scalar/elementwise.c \
    src/backends/cpu_scalar/transformer_ops.c \
    src/backends/cpu_scalar/weight_resolve.c
