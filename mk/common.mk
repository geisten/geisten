# mk/common.mk — base build rules, included after target/mode setup.
#
# Producer of:
#   $(LIB_FILE)        — lib/$(TARGET)/$(MODE)/libgeist.a
#   $(BIN_TARGETS)     — bin/$(TARGET)/$(MODE)/<every-binary>
#
# Consumes from target-$(TARGET).mk:
#   CC, CFLAGS_TARGET, LDFLAGS_TARGET, LDLIBS_TARGET
#
# Source layout: 3-layer split under src/ (engine/, archs/, backends/, io/).

# ---- Layout --------------------------------------------------------------

BUILD_DIR := build/$(TARGET)/$(MODE)
LIB_DIR   := lib/$(TARGET)/$(MODE)
BIN_DIR   := bin/$(TARGET)/$(MODE)

# ---- Mode flags ----------------------------------------------------------
# release : production
# debug   : -O0 -g for gdb stepping
# asan    : sanitizer build for refactor safety net
# perf    : -O3 + symbols for perf record / sampling profilers

ifeq      ($(MODE),release)
    CFLAGS_MODE  := -O3 -DNDEBUG
    LDFLAGS_MODE :=
else ifeq ($(MODE),debug)
    CFLAGS_MODE  := -O0 -g3 -DDEBUG
    LDFLAGS_MODE :=
else ifeq ($(MODE),asan)
    CFLAGS_MODE  := -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer
    LDFLAGS_MODE := -fsanitize=address,undefined
else ifeq ($(MODE),perf)
    CFLAGS_MODE  := -O3 -g -fno-omit-frame-pointer
    LDFLAGS_MODE :=
else
    $(error Unknown MODE=$(MODE). Use one of: release, debug, asan, perf)
endif

# ---- Base CFLAGS ---------------------------------------------------------

WARNINGS_BASE   := -Wall -Wextra -Wpedantic -Werror
WARNINGS        := $(WARNINGS_BASE)
WARNINGS_STRICT := $(WARNINGS_BASE) -Wshadow -Wundef

# Include paths:
#   -Iinclude   public headers (geist.h, geist_backend.h, geist_weight.h)
#   -I.         project root — heap.h (project-wide allocation policy per
#               AGENT.md) and internal tests reaching across the engine/arch
#               boundary via path-relative includes
#               (e.g. `#include "src/archs/transformer/arch_state_v2.h"`)
CFLAGS_BASE := -std=c23 $(WARNINGS) -fno-strict-aliasing \
               -Iinclude -I. \
               -Isrc/backends/common \
               -Isrc/formats/gguf \
               -Isrc/formats/ptqtp \
               -Isrc/io \
               -Isrc/engine \
               -Isrc/archs/audio_conformer \
               -Isrc/archs/vision_siglip \
               -Ithird_party/stb

# ---- Backend selection (per Q18, Q28) ------------------------------------
# `make BACKENDS="cpu_scalar cpu_neon"` enables both at compile time;
# runtime picks via geist_backend_create("name") or "auto".
# Default is cpu_scalar so the build works on any target as fallback.
BACKENDS ?= cpu_scalar
BACKEND_DEFINES := $(foreach b,$(BACKENDS),-DGEIST_BACKEND_$(shell echo $(b) | tr '[:lower:]' '[:upper:]')=1)

CFLAGS        := $(CFLAGS_BASE) $(BACKEND_DEFINES) $(CFLAGS_MODE) $(CFLAGS_TARGET) $(EXTRA_CFLAGS)
# Strict CFLAGS for the src/ tree — adds -Wshadow -Wundef on top of CFLAGS.
CFLAGS_STRICT := -std=c23 $(WARNINGS_STRICT) -fno-strict-aliasing \
                 -Iinclude -I. \
                 -Isrc/backends/common \
                 -Isrc/formats/gguf \
                 -Isrc/formats/ptqtp \
                 -Isrc/io \
                 -Isrc/engine \
                 -Isrc/archs/audio_conformer \
                 -Isrc/archs/vision_siglip \
                 -Ithird_party/stb \
                 $(BACKEND_DEFINES) $(CFLAGS_MODE) $(CFLAGS_TARGET) $(EXTRA_CFLAGS)
LDFLAGS := $(LDFLAGS_MODE) $(LDFLAGS_TARGET) $(EXTRA_LDFLAGS)
LDLIBS  := $(LDLIBS_TARGET) $(EXTRA_LDLIBS)

# ---- Sources -------------------------------------------------------------
# Library sources: files without main(). Phase B moves these into src/.

LIB_SOURCES := \
    src/engine/error.c \
    src/engine/allocator.c \
    src/engine/backend.c \
    src/engine/backend_registry.c \
    src/engine/hw_probe.c \
    src/engine/arch_registry.c \
    src/engine/model.c \
    src/engine/sampler.c \
    src/engine/session.c \
    src/engine/sp_bpe_tokenizer.c \
    src/engine/gguf_tokenizer.c \
    src/engine/version.c \
    src/archs/transformer/arch.c \
    src/archs/transformer/arch_state.c \
    src/archs/transformer/arch_family.c \
    src/archs/transformer/exec_plan.c \
    src/archs/transformer/scratch_plan.c \
    src/archs/transformer/forward/attention.c \
    src/archs/transformer/forward/kv_store.c \
    src/archs/transformer/forward/layer.c \
    src/archs/transformer/forward/layer_attn.c \
    src/archs/transformer/forward/layer_ffn.c \
    src/archs/transformer/forward/layer_ple.c \
    src/archs/transformer/forward/linear.c \
    src/archs/transformer/forward/profile.c \
    src/archs/transformer/forward/probes.c \
    src/archs/transformer/forward/step.c \
    src/archs/transformer/forward/head.c \
    src/archs/transformer/weight_load/dtype_map.c \
    src/archs/transformer/weight_load/tensor_views.c \
    src/archs/transformer/weight_load/layer_wiring.c \
    src/archs/transformer/arch_ops.c \
    src/archs/audio_conformer/arch.c \
    src/archs/audio_conformer/audio_encoder.c \
    src/archs/audio_conformer/audio_kernels.c \
    src/archs/audio_conformer/mel_pipeline.c \
    src/archs/vision_siglip/arch.c \
    src/archs/vision_siglip/vision_encoder.c \
    src/archs/vision_siglip/vision_kernels.c \
    src/archs/vision_siglip/image_pipeline.c \
    src/archs/vision_siglip/video_pipeline.c \
    src/formats/gguf/common.c \
    src/formats/gguf/q8_0.c \
    src/formats/gguf/q4_0.c \
    src/formats/gguf/q3_K.c \
    src/formats/gguf/q4_K.c \
    src/formats/gguf/q5_K.c \
    src/formats/gguf/q6_K.c \
    src/formats/gguf/iq2_s.c \
    src/formats/gguf/iq3_s.c \
    src/formats/gguf/tq2_0.c \
    src/backends/common/geist_gemm.c \
    src/backends/common/gemma4_kernels.c \
    src/backends/common/kivi.c \
    src/formats/ptqtp/gguf_ptqtp.c \
    src/formats/ptqtp/ptqtp_kernel.c \
    src/formats/ptqtp/ptqtp_awq.c \
    src/io/gguf_reader.c \
    src/io/safetensors_reader.c \
    src/backends/cpu_scalar/backend.c \
    src/backends/cpu_scalar/elementwise.c \
    src/backends/cpu_scalar/transformer_ops.c \
    src/backends/cpu_scalar/weight_resolve.c \
    src/backends/cpu_neon/backend.c \
    src/backends/cpu_neon/elementwise.c \
    src/backends/cpu_neon/kernel_catalog.c \
    src/backends/cpu_neon/kernels/iq2_s.c \
    src/backends/cpu_neon/kernels/iq3_s.c \
    src/backends/cpu_neon/kernels/q3_K.c \
    src/backends/cpu_neon/kernels/q4_K.c \
    src/backends/cpu_neon/kernels/q5_K.c \
    src/backends/cpu_neon/kernels/q6_K.c \
    src/backends/cpu_neon/kernels/q8_0.c \
    src/backends/cpu_neon/kernels/tq2_0.c \
    src/backends/cpu_neon/parallel.c \
    src/backends/cpu_neon/tl1.c \
    src/backends/cpu_neon/transformer_ops.c \
    src/backends/cpu_neon/weight_resolve.c \
    src/backends/cpu_neon/workspace.c \
    src/backends/cpu_x86/kernel_catalog.c

# Vulkan backend lives on the `vulkan-backend` branch and is intentionally
# omitted from refactor/v2 to keep the CPU-first development trajectory
# focused. To work on the GPU path, check out that branch directly.

# Project-wide allocation policy via heap.h — lives at the workspace root,
# alongside the engine tree. Explicit per-file rule below routes the .o
# through build/<target>/<mode>/ext/ to keep src/ object layout clean.
EXT_SOURCES := heap.c
EXT_OBJS := $(patsubst %.c,$(BUILD_DIR)/ext/%.o,$(EXT_SOURCES))

# Third-party single-header libraries (stb_image, stb_image_resize2). One
# implementation TU per project — compiled with -w to silence the dozens of
# pedantic warnings inside stb.
STB_OBJ := $(BUILD_DIR)/third_party/stb/stb_impl.o

# Binary sources: files with main(). Tests/benches live in tests/, eval/profile
# demos in tools/. Each binary links against libgeist.a and mirrors its source
# path under bin/ (tests/test_foo -> bin/.../tests/test_foo; tools/eval_geist ->
# bin/.../tools/eval_geist).
# Excluded: dump_llamacpp_logits.c (requires external llama.h from llama.cpp).
TEST_SOURCES := $(wildcard tests/test_*.c tests/bench_*.c)
DEMO_SOURCES := tools/eval_geist.c tools/profile_decode.c
BIN_SOURCES  := $(TEST_SOURCES) $(DEMO_SOURCES)

# ---- Derived paths -------------------------------------------------------

LIB_OBJS := $(patsubst %.c,$(BUILD_DIR)/%.o,$(LIB_SOURCES)) $(EXT_OBJS) $(STB_OBJ)
BIN_OBJS := $(patsubst %.c,$(BUILD_DIR)/%.o,$(BIN_SOURCES))

LIB_FILE := $(LIB_DIR)/libgeist.a
BIN_TARGETS := $(patsubst %.c,$(BIN_DIR)/%,$(BIN_SOURCES))

DEPS := $(LIB_OBJS:.o=.d) $(BIN_OBJS:.o=.d)

# ---- Rules ---------------------------------------------------------------

# Object compilation. -MMD -MP generates .d files for header tracking.
# src/*.c uses CFLAGS_STRICT (adds -Wshadow -Wundef); the tools/ demos
# (eval_geist, profile_decode) and tests/ use the slightly more relaxed
# CFLAGS. Both build clean under -Wall -Wextra -Werror.
$(BUILD_DIR)/src/%.o: src/%.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS_STRICT) -MMD -MP -c $< -o $@

$(BUILD_DIR)/%.o: %.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -MMD -MP -c $< -o $@

# Root-level external sources. Explicit per-file rule keeps the .o under
# build/<target>/<mode>/ext/ instead of mixing with src/ objects.
$(BUILD_DIR)/ext/%.o: %.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -MMD -MP -c $< -o $@

# stb single-header lib implementation TU. Compiled with -w because the
# stb headers throw many warnings under -Wall -Wextra -Wpedantic that we
# can't fix in a vendored file. Only this TU; consumers including the
# headers (without the IMPLEMENTATION macro) compile clean.
$(STB_OBJ): third_party/stb/stb_impl.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS_MODE) $(CFLAGS_TARGET) -Ithird_party/stb -w -MMD -MP -c $< -o $@

# Static library
$(LIB_FILE): $(LIB_OBJS)
	@mkdir -p $(@D)
	$(AR) rcs $@ $^

# Binary linking — each binary links against libgeist.a.
$(BIN_DIR)/%: $(BUILD_DIR)/%.o $(LIB_FILE)
	@mkdir -p $(@D)
	$(CC) $(LDFLAGS) -o $@ $< $(LIB_FILE) $(LDLIBS)

# Include generated dependency files (silent if missing).
-include $(DEPS)
