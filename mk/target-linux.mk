# mk/target-linux.mk — generic Linux target.
#
# geist's compute kernels are currently NEON-only: arm_neon.h is included
# unconditionally across src/formats/gguf/, src/backends/common/, and
# src/formats/ptqtp/. The x86 backend (src/backends/cpu_x86/) is a policy
# skeleton, not a working compute path yet. Therefore:
#
#   * aarch64 / arm64  → supported. Generic ARMv8.2 tuning (Graviton, Ampere,
#                        generic ARM64 servers/SBCs). For a Raspberry Pi 5
#                        specifically, prefer `make TARGET=pi5` (cortex-a76).
#   * x86_64           → not supported yet. We fail fast with guidance rather
#                        than emitting a wall of arm_neon.h compile errors.
#
# Stack mirrors pi5: OpenBLAS (cblas), FFTW3 single-precision (FFT), OpenMP.
# Override library locations via OPENBLAS_LIBS / FFTW3_LIBS (see `make help`).

LINUX_ARCH := $(shell uname -m)

ifneq (,$(filter $(LINUX_ARCH),x86_64 i686 i386))
  $(error TARGET=linux on $(LINUX_ARCH): x86 is not supported yet — geist's \
    kernels are NEON-only (see src/backends/cpu_x86/ skeleton). Contributions \
    porting the hot kernels to AVX2/AVX-512 are very welcome; see CONTRIBUTING.md. \
    On ARM64 hardware this target builds; on a Raspberry Pi 5 use TARGET=pi5.)
endif

# Compiler — gcc-13+ or clang-16+ for C23 support.
CC ?= cc

BACKENDS ?= cpu_neon cpu_scalar

# Generic ARMv8.2-A tuning — runs on Graviton2+, Ampere Altra, and most
# ARM64 SBCs. No -mcpu pin so the same binary is portable across cores.
# See target-pi5.mk for the rationale behind -ffast-math and the
# -Wno-nonnull-compare / -Wno-vla-parameter relaxations under stricter GCC.
CFLAGS_TARGET := -march=armv8.2-a+fp16+dotprod -fopenmp -ffast-math \
                 -Wno-nonnull-compare -Wno-vla-parameter

# OpenBLAS via pkg-config, fallback to plain -lopenblas.
OPENBLAS_LIBS   ?= $(shell pkg-config --libs   openblas 2>/dev/null || echo '-lopenblas')
OPENBLAS_CFLAGS ?= $(shell pkg-config --cflags openblas 2>/dev/null)

# FFTW3 single-precision (libfftw3f).
FFTW3_LIBS   ?= $(shell pkg-config --libs   fftw3f 2>/dev/null || echo '-lfftw3f')
FFTW3_CFLAGS ?= $(shell pkg-config --cflags fftw3f 2>/dev/null)

CFLAGS_TARGET  += $(OPENBLAS_CFLAGS) $(FFTW3_CFLAGS)
LDFLAGS_TARGET := -fopenmp
LDLIBS_TARGET  := $(OPENBLAS_LIBS) $(FFTW3_LIBS) -lm
