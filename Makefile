# geist Makefile — entry point.
#
# Usage:
#   make                       # auto-detect TARGET, MODE=release
#   make TARGET=pi5            # cross-compile (override CC if needed)
#   make MODE=asan             # sanitizer build
#   make MODE=asan TARGET=pi5  # combinations
#   make test                  # placeholder (Phase E implements)
#   make help                  # show all options
#
# Output layout (per-target, per-mode segregated):
#   build/$(TARGET)/$(MODE)/   *.o, *.d
#   lib/$(TARGET)/$(MODE)/     libgeist.a
#   bin/$(TARGET)/$(MODE)/     tests/test_*, tests/bench_*, tools/eval_geist, ...
#
# Adding a new target architecture:
#   1. Create mk/target-<name>.mk (set CC, CFLAGS_TARGET, LDFLAGS_TARGET, LDLIBS_TARGET)
#   2. Run: make TARGET=<name>

# Auto-detect target (mac on macOS, pi5 on ARM64 Linux, linux otherwise).
TARGET ?= $(shell mk/detect-target.sh)
MODE   ?= release

# Phony targets — do not match files.
.PHONY: all lib bin clean distclean help test test-unit test-int test-e2e test-all test-py fetch-model bench bench-small bench-detailed bench-quality-small bench-quality-detailed bench-compare-ref bench-mmlu bench-tooling format format-check

# Default goal.
all: lib bin

# Pull in target settings (CC, CFLAGS_TARGET, LDFLAGS_TARGET, LDLIBS_TARGET).
include mk/target-$(TARGET).mk

# Pull in common build rules (LIB_FILE, BIN_TARGETS, object/link rules).
include mk/common.mk

# Convenience aggregate goals.
lib: $(LIB_FILE)
bin: $(BIN_TARGETS)

# Test runner — invokes mk/run-tests.sh against the test bin directory.
# FILTER is an optional substring; e.g. `make test FILTER=q3k` runs only
# tests whose binary name contains "q3k".
TEST_BIN_DIR := $(BIN_DIR)/tests
BENCH_OUT_DIR ?= bench_runs/quality_perf
BENCH_GGUF ?=
BENCH_THREADS ?=
BENCH_REF_GGUF ?=
BENCH_REF_BIN ?=

# ---- Reference test model -------------------------------------------------
# The _int / _e2e / bench suites load a real GGUF via GEIST_GGUF_PATH and skip
# cleanly when it is absent (see tests/test_helpers.h). `make fetch-model`
# downloads it once into MODEL_DIR; the suites below auto-point
# GEIST_GGUF_PATH at it when present (unless the caller already set it).
#
# Source: unsloth's Gemma 4 E2B-it GGUF (Q4_K_M, ~3.1 GB). Overridable —
# point MODEL_URL elsewhere, or pass HF_TOKEN=... for a gated mirror.
MODEL_DIR     ?= gguf_artifacts
MODEL_FILE    ?= gemma4-e2b-Q4_K_M.gguf
MODEL_PATH    := $(MODEL_DIR)/$(MODEL_FILE)
MODEL_HF_REPO ?= unsloth/gemma-4-E2B-it-GGUF
MODEL_HF_FILE ?= gemma-4-E2B-it-Q4_K_M.gguf
MODEL_URL     ?= https://huggingface.co/$(MODEL_HF_REPO)/resolve/main/$(MODEL_HF_FILE)

# `make test` / test-int / test-e2e auto-fetch the model when it is missing,
# then point GEIST_GGUF_PATH at it so the model-gated suites actually run
# instead of skipping. Set AUTO_FETCH_MODEL=0 to keep the network out of
# `make test` (suites then skip cleanly when the model is absent — handy for
# CI / offline). A caller-provided GEIST_GGUF_PATH always wins and suppresses
# the download. MODEL_PREREQ is the on-demand download dependency; it is the
# real file target ($(MODEL_PATH)), so it no-ops when the model already exists.
AUTO_FETCH_MODEL ?= 1
ifeq ($(strip $(GEIST_GGUF_PATH)),)
  ifeq ($(AUTO_FETCH_MODEL),1)
    MODEL_PREREQ := $(MODEL_PATH)
  endif
endif

# Shell prelude for GGUF-consuming recipes, evaluated at recipe time (after any
# on-demand download): prefer a caller-set GEIST_GGUF_PATH, else use the
# reference model if present. Absolute path so it resolves regardless of cwd.
GGUF_ENV = if [ -z "$$GEIST_GGUF_PATH" ] && [ -f "$(MODEL_PATH)" ]; then \
               export GEIST_GGUF_PATH="$(abspath $(MODEL_PATH))"; \
           fi;

# `make test` chains unit + int + py — daily-iteration default. The model is
# listed FIRST so the on-demand download (if any) happens up front, before the
# unit tests run, rather than mid-run between unit and int suites.
test: $(MODEL_PREREQ) test-unit test-int test-py

test-unit: bin
	@$(GGUF_ENV) mk/run-tests.sh $(TEST_BIN_DIR) "_unit"

test-int: bin $(MODEL_PREREQ)
	@$(GGUF_ENV) mk/run-tests.sh $(TEST_BIN_DIR) "_int"

test-e2e: bin $(MODEL_PREREQ)
	@$(GGUF_ENV) mk/run-tests.sh $(TEST_BIN_DIR) "_e2e"

# Python-side tests (algorithm reference impls — PTQTP, quantization tooling).
# Hermetic: no GGUF, no network. Exit non-zero on any failure.
test-py:
	@status=0; \
	for f in $(wildcard tests/test_*.py); do \
		echo "=== $$f ==="; \
		python3 "$$f" || status=$$?; \
	done; \
	if [ $$status -ne 0 ]; then echo "test-py: FAIL"; exit $$status; fi; \
	echo "test-py: PASS"

# `make test-all` adds e2e but excludes benches. Model first (see `test`).
test-all: $(MODEL_PREREQ) test-unit test-int test-py test-e2e

# Download the reference GGUF (~3.1 GB) once into MODEL_DIR. Idempotent: the
# file rule no-ops when the model already exists, so it is safe to depend on
# and cheap to re-run. Downloads to a .part file and renames on success so an
# interrupted transfer never leaves a truncated model at the final path
# (curl -C - resumes the .part on the next run). Override source via MODEL_URL;
# pass HF_TOKEN=... for gated mirrors.
fetch-model: $(MODEL_PATH)
	@echo "Reference model ready: $(MODEL_PATH)"

$(MODEL_PATH):
	@command -v curl >/dev/null 2>&1 || { echo "fetch-model: curl not found in PATH" >&2; exit 1; }
	@mkdir -p $(MODEL_DIR)
	@echo "Downloading $(MODEL_FILE) (~3.1 GB) from:"
	@echo "  $(MODEL_URL)"
	@curl -fL --retry 3 --retry-delay 2 -C - \
	  $(if $(HF_TOKEN),-H "Authorization: Bearer $(HF_TOKEN)",) \
	  -o "$@.part" "$(MODEL_URL)"
	@mv "$@.part" "$@"
	@echo "Saved to $@"

# Benches are timing tools, not tests — separate target. Each bench prints
# its own metrics; runner just reports run/skip/fail status.
bench: bin
	@$(GGUF_ENV) GEIST_INCLUDE_BENCH=1 mk/run-tests.sh $(TEST_BIN_DIR) "bench_"

# Modality-specific multimodal benches — runnable separately so a user
# benching the vision pipeline doesn't pay for audio/quality suites.
# Each just filters the bench_<modality>_* binaries; argument-less
# invocation hits the default test asset paths.
bench-vision: bin
	@GEIST_INCLUDE_BENCH=1 mk/run-tests.sh $(TEST_BIN_DIR) "bench_vision_"

bench-video: bin
	@GEIST_INCLUDE_BENCH=1 mk/run-tests.sh $(TEST_BIN_DIR) "bench_video_"

bench-audio: bin
	@GEIST_INCLUDE_BENCH=1 mk/run-tests.sh $(TEST_BIN_DIR) "bench_audio_"

# All multimodal encoders end-to-end. Useful as a single CI gate.
bench-mm: bench-vision bench-video bench-audio

# Reproducible quality/performance suites. Both update benchmark/BENCHMARK.md only
# when the current run sets a new record for the same model/host/OS/target.
bench-small: bin
	@BENCH_GGUF="$(BENCH_GGUF)" BENCH_THREADS="$(BENCH_THREADS)" \
	python3 tools/bench_quality_perf.py \
	  --suite small --target "$(TARGET)" --mode "$(MODE)" \
	  --bin-dir "$(TEST_BIN_DIR)" --out-dir "$(BENCH_OUT_DIR)" \
	  --benchmark-md benchmark/BENCHMARK.md --record

bench-detailed: bin
	@BENCH_GGUF="$(BENCH_GGUF)" BENCH_THREADS="$(BENCH_THREADS)" \
	python3 tools/bench_quality_perf.py \
	  --suite detailed --target "$(TARGET)" --mode "$(MODE)" \
	  --bin-dir "$(TEST_BIN_DIR)" --out-dir "$(BENCH_OUT_DIR)" \
	  --benchmark-md benchmark/BENCHMARK.md --record

bench-quality-small: bin
	@BENCH_GGUF="$(BENCH_GGUF)" BENCH_THREADS="$(BENCH_THREADS)" \
	BENCH_REF_GGUF="$(BENCH_REF_GGUF)" \
	python3 tools/bench_quality_perf.py \
	  --suite quality-small --target "$(TARGET)" --mode "$(MODE)" \
	  --bin-dir "$(TEST_BIN_DIR)" --out-dir "$(BENCH_OUT_DIR)" \
	  --benchmark-md benchmark/BENCHMARK.md --record

bench-quality-detailed: bin
	@BENCH_GGUF="$(BENCH_GGUF)" BENCH_THREADS="$(BENCH_THREADS)" \
	BENCH_REF_GGUF="$(BENCH_REF_GGUF)" \
	python3 tools/bench_quality_perf.py \
	  --suite quality-detailed --target "$(TARGET)" --mode "$(MODE)" \
	  --bin-dir "$(TEST_BIN_DIR)" --out-dir "$(BENCH_OUT_DIR)" \
	  --benchmark-md benchmark/BENCHMARK.md --record

bench-compare-ref: bin
	@BENCH_GGUF="$(BENCH_GGUF)" BENCH_THREADS="$(BENCH_THREADS)" \
	BENCH_REF_GGUF="$(BENCH_REF_GGUF)" BENCH_REF_BIN="$(BENCH_REF_BIN)" \
	python3 tools/bench_quality_perf.py \
	  --suite compare-ref --target "$(TARGET)" --mode "$(MODE)" \
	  --bin-dir "$(TEST_BIN_DIR)" --out-dir "$(BENCH_OUT_DIR)" \
	  --benchmark-md benchmark/BENCHMARK.md --record

# Quality: MMLU accuracy via the self-contained tools/eval_mmlu.py harness
# (drives the eval_geist REPL, tokenizes with the model's OWN GGUF tokenizer —
# no external HF tokenizer, no chat-template parity issue; 5-shot base-completion
# cloze). Needs `pip install datasets` for the real cais/mmlu set. Override
# MMLU_LIMIT/MMLU_SHOTS; MMLU_LIMIT=0 runs the full ~14k-question set.
MMLU_LIMIT ?= 200
MMLU_SHOTS ?= 5
bench-mmlu: bin $(MODEL_PREREQ)
	@$(GGUF_ENV) OMP_WAIT_POLICY=active python3 tools/eval_mmlu.py \
	  --bin $(BIN_DIR)/tools/eval_geist \
	  --gguf "$${GEIST_GGUF_PATH:-$(abspath $(MODEL_PATH))}" \
	  --hf --shuffle --limit $(MMLU_LIMIT) --shots $(MMLU_SHOTS)

# Quality: function-calling + JSON-generation via tools/eval_tooling.py (also
# self-contained — drives the eval_geist GEN command, no dataset needed). The
# probe set is curated and validates extracted JSON (valid + schema + correct
# function/arguments). TOOLING_SUITE = json | func | all.
TOOLING_SUITE ?= all
bench-tooling: bin $(MODEL_PREREQ)
	@$(GGUF_ENV) OMP_WAIT_POLICY=active python3 tools/eval_tooling.py \
	  --bin $(BIN_DIR)/tools/eval_geist \
	  --gguf "$${GEIST_GGUF_PATH:-$(abspath $(MODEL_PATH))}" \
	  --suite $(TOOLING_SUITE)

# Cleanup.
clean:
	@rm -rf build/$(TARGET)/$(MODE) lib/$(TARGET)/$(MODE) bin/$(TARGET)/$(MODE)
	@echo "Cleaned $(TARGET)/$(MODE)."

distclean:
	@rm -rf build lib bin
	@rm -f *.npy *.bin test_* eval_geist profile_decode dump_llamacpp_logits bench_sgemv summary.json module_tree.txt tokens_ref.txt
	@echo "Cleaned all targets, modes, and temporary files."

# Code formatting via clang-format. Reads .clang-format from repo root.
# `make format` rewrites in place; `make format-check` is dry-run for CI.
FORMAT_FILES := $(wildcard *.c *.h tests/*.c tests/*.h)

format:
	@clang-format -i $(FORMAT_FILES)
	@echo "Formatted $(words $(FORMAT_FILES)) files."

format-check:
	@clang-format --dry-run --Werror $(FORMAT_FILES) && \
	echo "All $(words $(FORMAT_FILES)) files conform to .clang-format"

# Help text.
help:
	@echo "geist build system"
	@echo ""
	@echo "Common usage:"
	@echo "  make                         build everything for detected target (mode=release)"
	@echo "  make MODE=debug              -O0 -g for gdb stepping"
	@echo "  make MODE=asan               AddressSanitizer + UBSan build"
	@echo "  make MODE=perf               -O3 -g for perf record / profilers"
	@echo "  make lib                     only libgeist.a"
	@echo "  make bin                     only binaries (impl. requires lib)"
	@echo "  make test                    unit + int (default; auto-fetches model if missing)"
	@echo "  make test AUTO_FETCH_MODEL=0 same, but never download (model-gated tests skip)"
	@echo "  make test-unit               only _unit tests (kernel-level, fast)"
	@echo "  make test-int                only _int tests (multi-module, seconds)"
	@echo "  make test-e2e                only _e2e tests (slow, may need GGUF)"
	@echo "  make test-all                unit + int + e2e (excludes benches)"
	@echo "  make test-unit FILTER=q3k    only _unit tests containing 'q3k'"
	@echo "  make fetch-model             download reference GGUF (~3.1 GB) for _int/_e2e"
	@echo "  make fetch-model HF_TOKEN=.. same, with HF auth for gated mirrors"
	@echo "  make bench                   run timing benchmarks (not pass/fail)"
	@echo "  make bench-small             short quality/perf suite + record new bests"
	@echo "  make bench-detailed          longer quality/perf suite + record new bests"
	@echo "  make bench-quality-small     short PPL/KL quality suite + records"
	@echo "  make bench-quality-detailed  PPL + sampled MMLU/GSM8K quality suite"
	@echo "  make bench-compare-ref       geist vs reference perf/KL; set BENCH_REF_*"
	@echo "  make bench-mmlu              MMLU accuracy (5-shot cloze; needs 'pip install datasets')"
	@echo "  make bench-mmlu MMLU_LIMIT=0 full ~14k-question MMLU set"
	@echo "  make bench-tooling           function-calling + JSON-generation quality (self-contained)"
	@echo "  make clean                   remove current TARGET/MODE artifacts"
	@echo "  make distclean               remove all artifacts"
	@echo "  make format                  rewrite all sources via clang-format"
	@echo "  make format-check            verify formatting (no rewrite, errors on diff)"
	@echo ""
	@echo "Target selection:"
	@echo "  make TARGET=mac              Mac M1+ (Apple-clang + Accelerate)"
	@echo "  make TARGET=pi5              Pi 5 (gcc-13 + OpenBLAS + FFTW3 + OpenMP)"
	@echo "  detected: $(TARGET)"
	@echo ""
	@echo "Cross-compile (host=Mac/Linux, target=Pi 5):"
	@echo "  Install aarch64 toolchain first, then:"
	@echo "    make TARGET=pi5 CC=aarch64-linux-gnu-gcc-13"
	@echo "  For non-default OpenBLAS/FFTW3 location, override:"
	@echo "    make TARGET=pi5 OPENBLAS_LIBS=\"-L/opt/lib -lopenblas\" \\"
	@echo "                    FFTW3_LIBS=\"-L/opt/lib -lfftw3f\""
	@echo ""
	@echo "Current settings:"
	@echo "  TARGET=$(TARGET)"
	@echo "  MODE=$(MODE)"
	@echo "  CC=$(CC)"
	@echo "  CFLAGS=$(CFLAGS)"
	@echo "  LDFLAGS=$(LDFLAGS)"
	@echo "  LDLIBS=$(LDLIBS)"
	@echo "  BENCH_GGUF=$(BENCH_GGUF)"
	@echo "  BENCH_THREADS=$(BENCH_THREADS)"
	@echo "  BENCH_OUT_DIR=$(BENCH_OUT_DIR)"
	@echo "  BENCH_REF_GGUF=$(BENCH_REF_GGUF)"
	@echo "  BENCH_REF_BIN=$(BENCH_REF_BIN)"
	@echo "  MODEL_PATH=$(MODEL_PATH)"
	@echo "  MODEL_URL=$(MODEL_URL)"
	@echo "  AUTO_FETCH_MODEL=$(AUTO_FETCH_MODEL)"
	@echo "  GEIST_GGUF_PATH=$(GEIST_GGUF_PATH)"
