# Vulkan Transformer Fastpath

Status: proposal

Scope: add a high-performance Vulkan route for the transformer decoder while
preserving the existing CPU scalar/NEON path as the correctness oracle and
fallback.

Step-by-step TDD implementation plan:
[`docs/proposals/vulkan-tdd-implementation-plan.md`](vulkan-tdd-implementation-plan.md).

## Decision

Do not implement Vulkan as another implementation of the existing Level-2
primitive ops (`rmsnorm`, `linear`, `attention`, `rope_apply`, `add`, ...).
Those calls are shaped around host-resident buffers and pre-resolved CPU
function pointers. A Vulkan backend that only fills the Level-2 vtable would
need frequent `buffer_map` points, host-side scratch copies, and per-op command
submission, which is the wrong performance model.

Instead, add a transformer-level accelerator path above the Level-2 ops:

1. Keep GGUF parsing, tokenizer, metadata, session orchestration, and CPU
   fallback in the existing engine/arch layers.
2. Upload weights once at model load through staging buffers.
3. Repack supported weights into Vulkan-specific device-local layouts.
4. Allocate KV, activations, logits, token-id rings, and command buffers as
   device-resident session state.
5. Route `transformer_decode_step`, text prefill, and verify-forward through a
   persistent transformer graph when the model/session matches the supported
   target path.
6. Fall back to the existing CPU path for unsupported models, dtypes, sampling
   modes, debug knobs, multimodal soft-token paths, or failed Vulkan feature
   checks.

The first target path is intentionally narrow:

- model family: Gemma 4 E2B
- weight dtype: GGUF `Q4_K_M` for projection weights
- decoding: greedy only (`temperature == 0`)
- activations: FP32
- KV mode: INT8 KV for the optimized path, FP32 KV as a validation/debug mode
- tokenizer/GGUF/model metadata: CPU
- multimodal, KIVI, Q6/IQ/I2, stochastic sampling: follow-up work

## Current Code Pressure Points

The current CPU route is efficient for host execution because it chooses almost
everything at load time and uses direct CPU kernel pointers:

- `include/geist_weight.h` defines the host-pointer contract for
  `linear_m1`/`linear_mN`.
- `src/archs/transformer/forward/linear.c` dispatches through resolved CPU
  kernels.
- `src/archs/transformer/forward/step.c` maps scratch buffers and copies rows
  in `transformer_run_all_layers`.
- `src/archs/transformer/forward/layer_attn.c`,
  `src/archs/transformer/forward/layer_ffn.c`, and
  `src/archs/transformer/forward/head.c` contain additional hot-path
  `buffer_map` usage for scaling, sampling, row copies, PLE gather, and debug
  hooks.

Those are useful CPU choices, but they mean Vulkan must enter above
`transformer_run_all_layers` or at least above `transformer_forward_one_layer`.
The GPU route must never depend on `buffer_map` for normal weight, scratch, KV,
or logits flow.

## Proposed File Layout

Add a Vulkan backend and a transformer-private accelerator shim:

```text
src/backends/vulkan/
  backend.c              Vulkan backend descriptor, instance/device/queue setup
  buffer.c               VkBuffer allocation, staging upload/download, map policy
  features.c             device feature probing and route eligibility
  pipeline_cache.c       VkPipelineCache load/store integration
  shader_registry.c      SPIR-V module creation and pipeline lookup
  transformer_plan.c     device weight descriptors and layer graph metadata
  transformer_session.c  device KV/scratch/logits/token-id session state
  transformer_run.c      prefill/decode/verify command-buffer entry points
  shaders/
    *.comp               GLSL compute sources or checked-in generated SPIR-V

src/archs/transformer/accel.h
src/archs/transformer/accel.c
```

`accel.c` is an architecture-private dispatch layer. It should expose a small
backend-neutral interface to the transformer code:

```c
struct transformer_accel;

enum transformer_accel_op {
    TRANSFORMER_ACCEL_PREFILL_TEXT,
    TRANSFORMER_ACCEL_DECODE_GREEDY,
    TRANSFORMER_ACCEL_VERIFY_GREEDY,
};

struct transformer_accel_vtbl {
    void (*destroy_model)(struct transformer_accel *accel);
    enum geist_status (*session_create)(struct transformer_accel *accel,
                                        struct transformer_arch_session *sess);
    void (*session_destroy)(struct transformer_accel *accel,
                            struct transformer_arch_session *sess);
    bool (*supports_session)(struct transformer_accel *accel,
                             const struct transformer_arch_session *sess,
                             enum transformer_accel_op op);
    enum geist_status (*prefill_text)(struct transformer_accel *accel,
                                      struct transformer_arch_state *st,
                                      size_t n,
                                      const geist_token_t *ids);
    enum geist_status (*decode_greedy)(struct transformer_accel *accel,
                                       struct transformer_arch_state *st,
                                       geist_token_t input_token,
                                       geist_token_t *out_token);
    enum geist_status (*verify_greedy)(struct transformer_accel *accel,
                                       struct transformer_arch_state *st,
                                       size_t k,
                                       const geist_token_t *ids,
                                       geist_token_t *out_tokens);
    enum geist_status (*download_logits)(struct transformer_accel *accel,
                                         struct transformer_arch_state *st);
};
```

The Vulkan backend can implement this through a private header included only
when `GEIST_BACKEND_VULKAN` is compiled in. The public C ABI and normal
Level-2 backend vtable do not need transformer-specific Vulkan types.

## State Wiring

Extend `struct transformer_arch_state` with:

```c
struct transformer_accel *accel;
```

Extend `struct transformer_arch_session` with:

```c
void *accel_session;
bool logits_on_device;
bool logits_host_valid;
```

Creation and teardown:

1. `transformer_state_create_from_gguf` loads the model exactly as today for
   CPU fallback.
2. After CPU weight loading and feature/family detection, call
   `transformer_accel_try_create(st)`.
3. The Vulkan implementation validates device features, supported model
   geometry, supported dtypes, greedy path assumptions, and memory budget.
4. If accepted, it uploads and repacks all supported weights into device-local
   buffers and records immutable layer descriptors.
5. `transformer_session_alloc` asks the accelerator to allocate device KV,
   scratch, token upload, logits, and persistent command buffers for that
   session.
6. Destroy paths release accelerator session state before CPU buffers are
   released, then release the model-level accelerator.

The existing CPU buffers remain present in phase 1. That costs memory, but it
keeps fallback and cross-checking simple. A later memory-saving mode can skip
CPU-resolved weight payloads after Vulkan pack-cache validation.

## Entry Points

### Decode

Modify `transformer_decode_step`:

1. If `st->accel` supports `TRANSFORMER_ACCEL_DECODE_GREEDY`, call
   `decode_greedy`.
2. The accelerator uploads only the input token id, dispatches embedding,
   PLE, all layers, output norm, lm-head, and GPU argmax.
3. Download only one `geist_token_t` for the result.
4. Set `kv_len`, `next_token_pending`, `logits_valid`, and
   `logits_on_device`.
5. If any eligibility check fails before command submission, use the existing
   CPU path. If command submission fails, return the Vulkan backend error.

### Text Prefill

Modify `transformer_prefill_text_batch`:

1. If supported, upload token ids to a host-visible staging/ring buffer.
2. Run chunks up to the accelerator's max `m`.
3. For each chunk, execute the device-resident full-stack graph.
4. For the last row, run output norm, lm-head, and greedy argmax on GPU.
5. Download only the final token id.

The CPU path remains the fallback for multimodal soft-token prefill and any
session that requests unsupported options.

### Verify Forward

For speculative decode verification, support only greedy verification first:

1. Upload the candidate ids.
2. Run a batched full-stack graph.
3. Run batched output head and per-row GPU argmax.
4. Download `k` token ids.
5. Leave CPU rollback/truncate semantics unchanged.

## Device-Resident Graph

Record command buffers at session creation or first use. Keep one decode graph
and one prefill graph per supported chunk shape, with small dynamic state
provided by push constants or a host-visible uniform buffer:

- current `q_position`
- sequence length
- current KV length
- sliding/full attention mode
- token-id ring offset
- output token slot

Descriptor model:

- descriptor indexing for per-layer weight arrays and norm arrays
- static descriptors for global weights, RoPE tables, and pack metadata
- per-session descriptors for KV, scratch, logits, and token rings
- pipeline cache persisted by backend/device/model pack key

Synchronization:

- single compute queue for phase 1
- timeline semaphore or fence per submitted decode/prefill batch
- pipeline barriers between dependent shader passes inside the command buffer
- no host wait between layer sub-passes; wait only for final token download

## Fusion Boundary

Phase 1 should use a small number of shader passes per layer while preserving
device residency. The command buffer should cover at least a complete layer
block without host round trips.

Attention block:

1. RMSNorm over residual input.
2. QKV projection from compressed Q4_K_M weights. KV-shared layers compute Q
   only and read K/V from the source layer cache.
3. Q/K per-head norm and V norm for Gemma attention norms.
4. RoPE for Q and K.
5. INT8 KV quantize and append.
6. GQA attention against INT8 KV, with sliding-window or full causal mask.
7. O projection.
8. Post-attention norm and residual add.

FFN block:

1. RMSNorm.
2. Gate/up projection from compressed Q4_K_M weights.
3. GELU tanh, multiply, optional AWQ down-scale.
4. Down projection.
5. Post-FFN norm and residual add.
6. Layer output scale.

Output head:

1. Output RMSNorm on the final row.
2. Tied embedding/lm-head projection.
3. Greedy argmax on GPU.
4. Download `token_id` only.

For greedy decode, logit softcap is skipped because it is monotonic and does
not change argmax. If a caller later requests logits, the accelerator should
download the device logits into the existing host `scratch_logits` mirror and
clear `logits_on_device`.

## Weight Layout and Pack Cache

Vulkan needs backend-specific layouts. Do not dequantize low-bit weights into
FP16/FP32 at load time for the optimized route. Repack compressed data into
shader-friendly device layouts:

- rows grouped by output tile
- quant blocks aligned to vectorized loads
- scale/min metadata separated or interleaved based on shader access pattern
- optional per-weight metadata for AWQ inverse scales
- layer descriptors containing buffer offsets, shapes, and layout ids

The pack key should include:

- geist version and pack format version
- backend name `vulkan`
- Vulkan vendor/device/driver id
- shader pack layout version
- model content hash or GGUF tensor-table hash
- dtype/layout and relevant metadata for every packed tensor

Because `docs/proposals/geist-pack-cache.md` is not present in this checkout,
this proposal treats pack-cache as a required companion design. Without it,
model load will spend too much time repacking Q4_K_M every run.

## Vulkan Features

Required for the optimized path:

- Vulkan 1.2 or equivalent extension set
- storage buffers and compute queues
- `VK_KHR_shader_integer_dot_product`
- subgroup operations
- descriptor indexing
- timeline semaphores
- pipeline cache

Useful but optional:

- cooperative matrix paths for later hardware-specific matmul kernels
- shader float16/int8 support if the chosen kernels benefit from it
- external memory only if future interop needs it

If required features are missing, `transformer_accel_try_create` should decline
the Vulkan route and leave the CPU path active.

## Build Integration

Update the build in a way that keeps non-Vulkan builds unchanged:

1. Add `src/backends/vulkan/*.c` to `LIB_SOURCES` only when
   `BACKENDS` contains `vulkan`.
2. Add `GEIST_BACKEND_VULKAN` registration in
   `src/engine/backend_registry.c`.
3. Add target-specific Vulkan loader flags:
   - Linux: link `-lvulkan` or use dynamic loading.
   - macOS: use MoltenVK only if selected explicitly; do not make it part of
     the default `mac` target.
4. Add shader build rules. Prefer checked-in generated SPIR-V for release
   builds plus a developer target that rebuilds from GLSL when `glslangValidator`
   is available.

Default `BACKENDS` should remain CPU-only until the Vulkan path has correctness
and performance gates.

## Validation Plan

Correctness gates:

1. Backend buffer upload/download tests for Vulkan.
2. Shader unit tests for RMSNorm, RoPE, Q4_K_M matvec, INT8 KV append, and
   greedy argmax against CPU scalar.
3. One-layer cross-reference against `transformer_forward_one_layer` with fixed
   prompt ids.
4. Full decode token parity for Gemma 4 E2B Q4_K_M greedy.
5. Prefill parity for chunk sizes 1, 2, 8, 64, and prompt lengths crossing
   chunk boundaries.
6. Verify-forward parity for `k` values used by speculative decode.

Performance gates:

1. No `buffer_map` calls on the Vulkan hot path except final token/logit
   downloads and debug-only code.
2. Decode submits one persistent graph per token and downloads one token id.
3. Prefill avoids host waits inside layer loops.
4. Pack-cache hit path is substantially faster than first-run repack.
5. Report tokens/sec, prefill tokens/sec, GPU memory use, command submission
   time, shader time, and pack-cache time.

## Bring-Up Order

1. Skeleton Vulkan backend: instance/device/queue, buffers, upload/download,
   feature probe, registry, build flags.
2. Transformer accelerator shim with eligibility checks and no-op fallback.
3. Device-local global/session allocation: weights, RoPE, token ring, scratch,
   logits, FP32 KV debug mode.
4. Minimal decode graph for one layer, then all layers, using simple shaders
   and CPU parity tests.
5. GPU greedy output head and token-only download.
6. INT8 KV append and attention path.
7. Q4_K_M shader-friendly repack and pack-cache.
8. Batched prefill graph.
9. Verify-forward graph.
10. CPU fallback hardening, diagnostics, telemetry, and documentation.

## Explicit Non-Goals For Phase 1

- No Vulkan implementation of the generic Level-2 op vtable as the performance
  route.
- No stochastic sampling on GPU.
- No KIVI, IQ2/IQ3/I2/TQ2/Q6 support.
- No multimodal soft-token GPU path.
- No attempt to remove CPU fallback buffers before pack-cache and parity tests
  are stable.
- No cooperative matrix dependency in the baseline path.
