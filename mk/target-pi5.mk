# mk/target-pi5.mk — Raspberry Pi 5 / Cortex-A76 target settings.
#
# Audience: Pi 5 (ARM64, Cortex-A76, 4 cores).
# Stack: OpenBLAS for cblas, FFTW3 (single precision) for FFT, OpenMP for
# parallel kernels (m>1 prefill loops in NEON backend).
#
# Dependencies resolved via pkg-config with manual override via OPENBLAS_LIBS,
# FFTW3_LIBS environment variables (see `make help`).

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

# FFTW3 single-precision (libfftw3f).
FFTW3_LIBS  ?= $(shell pkg-config --libs   fftw3f 2>/dev/null || echo '-lfftw3f')
FFTW3_CFLAGS ?= $(shell pkg-config --cflags fftw3f 2>/dev/null)

CFLAGS_TARGET  += $(OPENBLAS_CFLAGS) $(FFTW3_CFLAGS)
LDFLAGS_TARGET := -fopenmp
LDLIBS_TARGET  := $(OPENBLAS_LIBS) $(FFTW3_LIBS) -lm
