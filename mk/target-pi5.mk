# mk/target-pi5.mk — Raspberry Pi 5 / Cortex-A76 target settings.
#
# Audience: Pi 5 (ARM64, Cortex-A76, 4 cores).
# Stack: OpenBLAS for cblas (dense fp32), OpenMP for threading; FFT is vendored.
# parallel kernels (m>1 prefill loops in NEON backend).
#
# Dependencies resolved via pkg-config with manual override via OPENBLAS_LIBS,
# OPENBLAS_LIBS environment variable (see `make help`).

# Compiler — prefer gcc-13 for proper C23 constexpr support.
# Cross-compile: override with CC=aarch64-linux-gnu-gcc-13.
CC ?= gcc

BACKENDS ?= cpu_neon cpu_scalar

# Cortex-A76 specialization — best codegen for NEON kernels.
#
# Pi-side GCC (14.2 on Debian Trixie) is stricter than Mac's clang/gcc on a
# few defensive-coding patterns that the static-array contract makes
# redundant but harmless: explicit NULL checks on parameters declared with
# `[static n]` (-Wnonnull-compare) trigger errors under -Werror even though
# the code is correct. Mac builds keep the stricter form for our own
# discipline; on Pi we just disable the warning rather than weakening the
# code style across the codebase.
# `-ffast-math` enables fp reassociation + finite-math assumptions,
# unlocking substantially more aggressive NEON autovectorization in
# the softmax / activation / elementwise kernels. +12% Pi 5 decode on
# BitNet 2B-4T at t=4 active wait. Greedy decode and WikiText PPL match
# strict-math within noise (verified on bitnet-2b4t-TQ2_0-v2.gguf).
# GCC's `-ffast-math` defines `__FAST_MATH__`; some code paths may opt
# out via `#pragma STDC FENV_ACCESS ON` if exact rounding ever matters.
CFLAGS_TARGET := -DGEIST_TARGET_PI5=1 -mcpu=cortex-a76 -fopenmp -ffast-math -Wno-nonnull-compare -Wno-vla-parameter

# OpenBLAS via pkg-config, fallback to plain -lopenblas.
OPENBLAS_LIBS  ?= $(shell pkg-config --libs   openblas 2>/dev/null || echo '-lopenblas')
OPENBLAS_CFLAGS ?= $(shell pkg-config --cflags openblas 2>/dev/null)

LDFLAGS_TARGET := -fopenmp

# Dense fp32 backend. Default: cblas/OpenBLAS. GEIST_BLAS_FREE=1 routes dense
# fp32 through the native NEON path (geist_gemm) and links NO external math
# libs — a fully dependency-free binary (libc/libm/libgomp only), for the
# musl-static CI artifact. Audio FFT is vendored either way (no FFTW3).
ifeq ($(GEIST_BLAS_FREE),1)
  CFLAGS_TARGET += -DGEIST_GEMM_NATIVE
  LDLIBS_TARGET := -lm
else
  CFLAGS_TARGET += $(OPENBLAS_CFLAGS)
  LDLIBS_TARGET := $(OPENBLAS_LIBS) -lm
endif
