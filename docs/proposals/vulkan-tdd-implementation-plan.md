# Vulkan TDD Implementation Plan

Status: proposal

Scope: implement the optional Vulkan transformer route in small increments.
Every step must leave the tree buildable, keep non-Vulkan environments working,
and pass the listed tests before the next step starts.

The plan assumes the optimized route remains optional:

- CPU-only builds must keep working.
- Vulkan builds must be explicitly selected.
- Runtime feature checks must decline unsupported GPUs without breaking CPU
  inference.
- The CPU scalar/NEON path remains the correctness oracle until Vulkan parity
  and performance gates are stable.

## Step 0: Test Harness Baseline

Goal: capture the current CPU behavior before touching Vulkan code.

Implementation:

- Add no Vulkan code yet.
- Identify the smallest existing unit/integration/e2e test set that exercises:
  backend creation, buffer ops, transformer load, prefill, decode, and
  verify-forward.
- Document the exact commands for local Vulkan development.

Tests:

```sh
make clean
make test-unit AUTO_FETCH_MODEL=0
make test-int AUTO_FETCH_MODEL=0
make test-py
```

With a local GGUF available:

```sh
GEIST_GGUF_PATH=/path/to/gemma4-e2b-Q4_K_M.gguf make test-int AUTO_FETCH_MODEL=0
GEIST_GGUF_PATH=/path/to/gemma4-e2b-Q4_K_M.gguf make test-e2e AUTO_FETCH_MODEL=0
```

Acceptance:

- Baseline tests pass before implementation begins.
- Any skipped model tests are explicitly due to missing `GEIST_GGUF_PATH`, not
  regressions.

## Step 1: Build-Gated Vulkan Backend Skeleton

Goal: add a backend that can be compiled only when requested.

Implementation:

- Add `src/backends/vulkan/backend.c`.
- Export `geist_backend_vulkan`.
- Register it in `src/engine/backend_registry.c` behind
  `GEIST_BACKEND_VULKAN`.
- Add build rules so `BACKENDS` without `vulkan` does not compile or link any
  Vulkan source.
- Keep the Vulkan vtable mostly unsupported except `create`, `destroy`, and
  `supports_op`.

Tests first:

- Add or extend a backend registry unit test that verifies:
  - CPU-only builds do not expose `vulkan`.
  - Vulkan-selected builds expose `vulkan` if compiled.

Tests:

```sh
make clean
make BACKENDS="cpu_scalar" test-unit AUTO_FETCH_MODEL=0
make clean
make BACKENDS="vulkan cpu_scalar" test-unit AUTO_FETCH_MODEL=0
```

Acceptance:

- CPU-only build never needs Vulkan headers or libraries.
- Vulkan-selected build compiles the skeleton.
- Unsupported ops return `GEIST_SUPPORT_NONE` or `GEIST_E_UNSUPPORTED`
  predictably.

## Step 2: Vulkan Loader and Unsupported-Environment Fallback

Goal: make Vulkan backend creation safe on machines without Vulkan support.

Implementation:

- Add Vulkan loader/device discovery.
- Prefer dynamic loading if the target platform lacks guaranteed Vulkan SDK
  availability.
- If no loader, instance, physical device, compute queue, or required base
  feature exists, backend creation must fail cleanly with a backend error.
- Ensure `geist_backend_create("auto")` still selects CPU first unless Vulkan is
  explicitly prioritized later.

Tests first:

- Add a unit test that creates `cpu_scalar` successfully even when Vulkan is
  compiled in.
- Add a Vulkan creation test that accepts either:
  - successful creation on supported hosts, or
  - a clean `GEIST_E_BACKEND`/unsupported result with a non-empty error message.

Tests:

```sh
make clean
make BACKENDS="vulkan cpu_scalar" test-unit AUTO_FETCH_MODEL=0
```

Acceptance:

- Machines without Vulkan still pass all CPU tests.
- Vulkan failure is non-fatal unless the user explicitly requested only Vulkan.

## Step 3: Vulkan Buffer API

Goal: establish correct staging upload/download and map policy.

Implementation:

- Implement `buffer_create`, `buffer_destroy`, `buffer_upload`,
  `buffer_download`, `buffer_map`, and `buffer_unmap`.
- Use host-visible staging buffers for upload/download.
- Return `nullptr` from `buffer_map` for device-local buffers that cannot be
  safely host-aliased.
- Support small host-visible buffers for debug/staging roles only.

Tests first:

- Add backend buffer tests for:
  - upload/download round trip
  - zero-byte and invalid-arg handling
  - map behavior for host-visible vs device-local buffers
  - destruction after failed partial creation

Tests:

```sh
make BACKENDS="vulkan cpu_scalar" test-unit AUTO_FETCH_MODEL=0
```

Acceptance:

- Buffer tests pass on supported Vulkan hosts.
- On unsupported Vulkan hosts, tests skip cleanly after a documented backend
  creation failure.
- CPU-only tests remain unchanged.

## Step 4: Transformer Accelerator Shim With No-Op Fallback

Goal: add the architecture-level entry point without changing behavior.

Implementation:

- Add `src/archs/transformer/accel.h` and `accel.c`.
- Add `struct transformer_accel *accel` to `transformer_arch_state`.
- Add `void *accel_session`, `logits_on_device`, and `logits_host_valid` to
  `transformer_arch_session`.
- Wire create/destroy/session hooks, but make `transformer_accel_try_create`
  return no accelerator until Vulkan support is explicitly enabled.
- Keep `transformer_decode_step`, prefill, and verify-forward behavior exactly
  CPU-equivalent.

Tests first:

- Add tests that load a model with acceleration disabled and verify existing
  decode/prefill behavior.
- Add lifecycle tests for create/destroy/session allocation when `accel` is
  null.

Tests:

```sh
make BACKENDS="cpu_scalar" test-unit AUTO_FETCH_MODEL=0
make BACKENDS="cpu_neon cpu_scalar" test-unit AUTO_FETCH_MODEL=0
GEIST_GGUF_PATH=/path/to/gemma4-e2b-Q4_K_M.gguf make BACKENDS="cpu_scalar" test-int AUTO_FETCH_MODEL=0
```

Acceptance:

- No token changes.
- No new dependency on Vulkan for CPU builds.
- Null accelerator lifecycle is leak-free.

## Step 5: Vulkan Feature Eligibility

Goal: decide at model load whether the optimized route may exist.

Implementation:

- Implement `transformer_accel_try_create` for Vulkan.
- Check:
  - backend is Vulkan
  - model family is Gemma 4 E2B
  - projection weights are supported `Q4_K_M` dtypes/layouts
  - greedy mode is available for decode path
  - required Vulkan features are present
  - memory budget is plausible
- Decline cleanly without affecting CPU fallback.

Tests first:

- Add eligibility unit tests with mocked/minimal metadata where possible.
- Add integration test that asserts unsupported model/session options stay on
  CPU.

Tests:

```sh
make BACKENDS="vulkan cpu_scalar" test-unit AUTO_FETCH_MODEL=0
GEIST_GGUF_PATH=/path/to/gemma4-e2b-Q4_K_M.gguf make BACKENDS="vulkan cpu_scalar" test-int AUTO_FETCH_MODEL=0
```

Acceptance:

- Supported hosts may create an accelerator.
- Unsupported hosts/options decline acceleration and still pass CPU inference.

## Step 6: Shader Tooling and First Compute Kernel

Goal: add reproducible shader build/runtime loading before transformer work.

Implementation:

- Add shader registry and pipeline-cache skeleton.
- Add one simple compute shader, such as vector add or scale.
- Prefer checked-in generated SPIR-V for normal builds.
- Add a developer target to rebuild shaders when `glslangValidator` exists.

Tests first:

- Add a Vulkan shader unit test comparing shader output against CPU reference.
- Add a test that missing optional shader compiler does not break normal builds.

Tests:

```sh
make BACKENDS="vulkan cpu_scalar" test-unit AUTO_FETCH_MODEL=0
```

Acceptance:

- First shader executes correctly on supported Vulkan hosts.
- Pipeline creation failures are reported cleanly.

## Step 7: Device Weight Upload and Pack Metadata

Goal: upload model weights to device-local buffers without using them for
inference yet.

Implementation:

- Add Vulkan model-level plan structs.
- Upload supported global and layer tensors through staging buffers.
- Preserve CPU buffers for fallback.
- Record per-tensor device offsets, shape, dtype, and provisional layout ids.

Tests first:

- Add tests that compare uploaded/downloaded tensor bytes for small synthetic
  tensors.
- Add model-load test that verifies accelerator weight metadata matches CPU
  tensor metadata.

Tests:

```sh
make BACKENDS="vulkan cpu_scalar" test-unit AUTO_FETCH_MODEL=0
GEIST_GGUF_PATH=/path/to/gemma4-e2b-Q4_K_M.gguf make BACKENDS="vulkan cpu_scalar" test-int AUTO_FETCH_MODEL=0
```

Acceptance:

- Model load succeeds with Vulkan enabled.
- CPU fallback still produces identical tokens.
- No Vulkan device weight is accessed through `buffer_map`.

## Step 8: Device Session Allocation

Goal: allocate GPU KV/scratch/logits/token-id buffers per session.

Implementation:

- Add Vulkan session state.
- Allocate:
  - token upload ring
  - residual stream scratch
  - Q/K/V/attention/FFN scratch
  - FP32 KV debug buffers
  - INT8 KV buffers and scales for the target path
  - logits buffer
  - token result buffer
- Keep command buffers unrecorded or empty for now.

Tests first:

- Add session allocation/destruction tests.
- Add tests for multiple sessions sharing one model-level Vulkan plan.
- Add OOM/oversized `max_seq_len` handling tests where practical.

Tests:

```sh
make BACKENDS="vulkan cpu_scalar" test-unit AUTO_FETCH_MODEL=0
GEIST_GGUF_PATH=/path/to/gemma4-e2b-Q4_K_M.gguf make BACKENDS="vulkan cpu_scalar" test-int AUTO_FETCH_MODEL=0
```

Acceptance:

- Sessions allocate and destroy cleanly.
- CPU inference still works with Vulkan session state present.

## Step 9: GPU Greedy Argmax Kernel

Goal: implement the final-token download contract independently.

Implementation:

- Add GPU argmax over a logits buffer.
- For greedy mode, skip softcap because it is monotonic.
- Download only `geist_token_t`.

Tests first:

- Add random logits argmax tests against `geist_sampler_argmax`.
- Include tie-breaking tests matching CPU semantics.
- Include vocab sizes around Gemma's `262144` and smaller synthetic sizes.

Tests:

```sh
make BACKENDS="vulkan cpu_scalar" test-unit AUTO_FETCH_MODEL=0
```

Acceptance:

- GPU argmax returns exactly the CPU token id.
- Only the token id is downloaded in the tested path.

## Step 10: One-Layer FP32 Debug Graph

Goal: prove transformer-level command-buffer routing before low-bit kernels.

Implementation:

- Add simple FP32 debug shaders for RMSNorm, RoPE, residual add, and attention
  using uploaded debug weights or small synthetic fixtures.
- Record and submit a one-layer graph for synthetic dimensions first.
- Do not optimize yet.

Tests first:

- Add one-layer synthetic cross-reference against CPU scalar.
- Add command-buffer replay test to ensure persistent recording is reusable.

Tests:

```sh
make BACKENDS="vulkan cpu_scalar" test-unit AUTO_FETCH_MODEL=0
```

Acceptance:

- One-layer synthetic output matches CPU within a fixed tolerance.
- No host synchronization occurs inside the layer graph except final test readback.

## Step 11: Q4_K_M Matvec/Matmul Shader

Goal: implement the critical compressed-weight projection kernel.

Implementation:

- Add Q4_K_M device layout reader for source layout first.
- Add repacked layout after source-layout correctness passes.
- Implement M=1 and small M>1 kernels.
- Keep compressed weights compressed; do not pre-dequantize the whole matrix.

Tests first:

- Add per-row Q4_K_M projection tests against CPU scalar/NEON reference.
- Test Gemma projection shapes where possible:
  - Q, K, V, O
  - gate/up/down
  - tied lm-head
- Test M=1 and M>1.

Tests:

```sh
make BACKENDS="vulkan cpu_scalar" test-unit AUTO_FETCH_MODEL=0
GEIST_GGUF_PATH=/path/to/gemma4-e2b-Q4_K_M.gguf make BACKENDS="vulkan cpu_scalar" test-int AUTO_FETCH_MODEL=0
```

Acceptance:

- Projection outputs match CPU within agreed tolerance.
- Repacked layout tests pass before using repack in transformer decode.

## Step 12: GPU Output Head Path

Goal: run output norm, lm-head, and argmax on GPU for a known hidden vector.

Implementation:

- Feed a hidden vector into device scratch.
- Run output RMSNorm.
- Run tied lm-head Q4_K_M projection.
- Run GPU argmax.
- Download token id.

Tests first:

- Add output-head cross-reference using hidden vectors generated by CPU.
- Validate greedy token id parity, not full-logit parity as the primary gate.
- Add optional debug mode to download logits for failure analysis.

Tests:

```sh
make BACKENDS="vulkan cpu_scalar" test-unit AUTO_FETCH_MODEL=0
GEIST_GGUF_PATH=/path/to/gemma4-e2b-Q4_K_M.gguf make BACKENDS="vulkan cpu_scalar" test-int AUTO_FETCH_MODEL=0
```

Acceptance:

- GPU output head returns the same greedy token as CPU for fixture hidden rows.
- Logit download remains debug-only.

## Step 13: Decode Graph With FP32 KV Debug Mode

Goal: run full Gemma layer stack on GPU for one decode token using FP32 KV.

Implementation:

- Add embedding lookup and PLE device path.
- Execute all layer attention/FFN blocks with FP32 KV append and attention.
- Run output head and GPU argmax.
- Advance session `kv_len` and pending token state.

Tests first:

- Add single-token decode parity test after a short CPU-prefilled prompt.
- Add `kv_len` and pending-token state tests.
- Add fallback test for unsupported sampling options.

Tests:

```sh
make BACKENDS="vulkan cpu_scalar" test-unit AUTO_FETCH_MODEL=0
GEIST_GGUF_PATH=/path/to/gemma4-e2b-Q4_K_M.gguf make BACKENDS="vulkan cpu_scalar" test-int AUTO_FETCH_MODEL=0
GEIST_GGUF_PATH=/path/to/gemma4-e2b-Q4_K_M.gguf make BACKENDS="vulkan cpu_scalar" test-e2e AUTO_FETCH_MODEL=0
```

Acceptance:

- Greedy token parity for the tested prompt.
- CPU fallback works for non-greedy sampling.
- Vulkan hot path does not map weight/KV/scratch buffers.

## Step 14: INT8 KV Append and Attention

Goal: switch the optimized path from FP32 KV debug mode to INT8 KV.

Implementation:

- Add INT8 KV quantize/append kernels.
- Add attention kernel that reads INT8 K/V plus scales.
- Keep FP32 KV as a debug/reference Vulkan mode.

Tests first:

- Add KV quantization round-trip tests.
- Add attention tests comparing INT8 KV output against CPU INT8 or FP32
  tolerance bounds.
- Re-run decode parity/token tests.

Tests:

```sh
make BACKENDS="vulkan cpu_scalar" test-unit AUTO_FETCH_MODEL=0
GEIST_GGUF_PATH=/path/to/gemma4-e2b-Q4_K_M.gguf make BACKENDS="vulkan cpu_scalar" test-int AUTO_FETCH_MODEL=0
GEIST_GGUF_PATH=/path/to/gemma4-e2b-Q4_K_M.gguf make BACKENDS="vulkan cpu_scalar" test-e2e AUTO_FETCH_MODEL=0
```

Acceptance:

- INT8 KV decode produces accepted greedy tokens for the target fixtures.
- FP32 KV debug mode remains available for bisecting precision issues.

## Step 15: Device-Resident Text Prefill

Goal: move text prefill chunks onto the Vulkan route.

Implementation:

- Upload token-id batches.
- Run embedding, PLE, full stack, KV append, and final-row output head on GPU.
- Support chunk sizes up to accelerator `m_max`.
- Keep CPU prefill fallback for unsupported sessions.

Tests first:

- Add prefill parity tests for chunk sizes 1, 2, 8, 64.
- Add prompt lengths crossing chunk boundaries.
- Add CPU fallback tests for multimodal soft-token prefill.

Tests:

```sh
make BACKENDS="vulkan cpu_scalar" test-unit AUTO_FETCH_MODEL=0
GEIST_GGUF_PATH=/path/to/gemma4-e2b-Q4_K_M.gguf make BACKENDS="vulkan cpu_scalar" test-int AUTO_FETCH_MODEL=0
GEIST_GGUF_PATH=/path/to/gemma4-e2b-Q4_K_M.gguf make BACKENDS="vulkan cpu_scalar" test-e2e AUTO_FETCH_MODEL=0
```

Acceptance:

- Final pending token after prefill matches CPU greedy behavior.
- No host round trip occurs between layers during GPU prefill.

## Step 16: Verify-Forward Graph

Goal: support speculative decode verification on GPU for greedy mode.

Implementation:

- Upload candidate token ids.
- Run batched full-stack graph.
- Run batched output head and per-row GPU argmax.
- Download `k` token ids.
- Preserve existing CPU rollback/truncate semantics.

Tests first:

- Add verify-forward parity for representative `k`.
- Add accept/reject/truncate tests around speculative decode.
- Add fallback for unsupported `k > m_max`.

Tests:

```sh
make BACKENDS="vulkan cpu_scalar" test-unit AUTO_FETCH_MODEL=0
GEIST_GGUF_PATH=/path/to/gemma4-e2b-Q4_K_M.gguf make BACKENDS="vulkan cpu_scalar" test-int AUTO_FETCH_MODEL=0
GEIST_GGUF_PATH=/path/to/gemma4-e2b-Q4_K_M.gguf make BACKENDS="vulkan cpu_scalar" test-e2e AUTO_FETCH_MODEL=0
```

Acceptance:

- Speculative verification token outputs match CPU for greedy fixtures.
- KV truncate behavior remains identical.

## Step 17: Pack Cache

Goal: avoid expensive Vulkan repack on every model load.

Implementation:

- Define pack-cache key and version.
- Store packed weight layouts and metadata.
- Validate cache compatibility with backend/device/driver/shader layout/model
  hash before use.
- Fall back to repack on cache miss or invalid cache.

Tests first:

- Add pack-cache hit/miss/version-mismatch tests with synthetic tensors.
- Add corrupt-cache fallback test.
- Add model-load test proving cached and freshly packed routes produce the same
  greedy token outputs.

Tests:

```sh
make BACKENDS="vulkan cpu_scalar" test-unit AUTO_FETCH_MODEL=0
GEIST_GGUF_PATH=/path/to/gemma4-e2b-Q4_K_M.gguf make BACKENDS="vulkan cpu_scalar" test-int AUTO_FETCH_MODEL=0
```

Acceptance:

- Cache hit path is functionally identical to fresh repack.
- Invalid cache never causes incorrect inference.

## Step 18: Performance Gates and Telemetry

Goal: prove the path is actually faster and stays device-resident.

Implementation:

- Add counters for:
  - accelerator eligibility result
  - buffer downloads/uploads on hot path
  - command submission time
  - shader time when timestamp queries are available
  - pack-cache hit/miss time
  - GPU memory use
- Add benchmark path for decode and prefill.

Tests first:

- Add telemetry unit tests where counters are deterministic.
- Add performance smoke benchmark that asserts no accidental full-logit
  download in greedy decode.

Tests:

```sh
make BACKENDS="vulkan cpu_scalar" test-unit AUTO_FETCH_MODEL=0
GEIST_GGUF_PATH=/path/to/gemma4-e2b-Q4_K_M.gguf make BACKENDS="vulkan cpu_scalar" bench AUTO_FETCH_MODEL=0
```

Acceptance:

- Greedy decode downloads one token id, not full logits.
- No `buffer_map` appears in the Vulkan hot path except explicit debug/logit
  download paths.
- Benchmarks report Vulkan route status and timing.

## Step 19: Hardening Matrix

Goal: make the feature safe across supported and unsupported environments.

Implementation:

- Exercise:
  - CPU-only build
  - Vulkan-compiled but unavailable runtime
  - Vulkan available but missing required feature
  - Vulkan available and eligible model
  - unsupported model/dtype/session options
- Improve diagnostics for each fallback reason.

Tests:

```sh
make clean && make BACKENDS="cpu_scalar" test-unit AUTO_FETCH_MODEL=0
make clean && make BACKENDS="cpu_neon cpu_scalar" test-unit AUTO_FETCH_MODEL=0
make clean && make BACKENDS="vulkan cpu_scalar" test-unit AUTO_FETCH_MODEL=0
GEIST_GGUF_PATH=/path/to/gemma4-e2b-Q4_K_M.gguf make BACKENDS="vulkan cpu_scalar" test-all AUTO_FETCH_MODEL=0
```

Acceptance:

- Unsupported environments build and run CPU tests.
- Supported Vulkan hosts pass full parity tests.
- Fallback reasons are visible enough to debug user reports.

## Step 20: Enablement Policy

Goal: decide when Vulkan can move from experimental to opt-in production.

Implementation:

- Keep Vulkan out of default `BACKENDS` until:
  - correctness gates pass on at least one Linux Vulkan target
  - unsupported-host fallback is proven in CI or equivalent local matrix
  - pack-cache is stable
  - performance beats CPU on the target hardware by a documented margin
- Document required drivers, features, environment variables, and limitations.

Tests:

```sh
make BACKENDS="vulkan cpu_scalar" test-all AUTO_FETCH_MODEL=0
```

With model and supported hardware:

```sh
GEIST_GGUF_PATH=/path/to/gemma4-e2b-Q4_K_M.gguf make BACKENDS="vulkan cpu_scalar" test-all AUTO_FETCH_MODEL=0
GEIST_GGUF_PATH=/path/to/gemma4-e2b-Q4_K_M.gguf make BACKENDS="vulkan cpu_scalar" bench AUTO_FETCH_MODEL=0
```

Acceptance:

- Vulkan remains explicitly opt-in unless all enablement gates are met.
- Documentation clearly says what is accelerated and what falls back to CPU.
