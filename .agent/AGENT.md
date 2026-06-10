Context: pure C23 inference engine / runtime core.

Priorities:
1. correctness
2. performance (cache locality, low memory, predictable behavior)
3. portability
4. testability

Hard constraints:
- Use modern C23 features where they improve safety, clarity, or compile-time checking.
- Minimize naked pointer usage in public APIs. Count parameters precede the array/buffer they constrain.
  Prefer signatures like:
  - bool fn(size_t n, const float xs[static n]);
  - error_code fn(size_t dst_n, float dst[static dst_n], size_t src_n, const float src[static src_n]);
- Caller-provided buffers for I/O and scratch; make workspace requirements explicit.
- Hot paths must be allocation-free. No hidden heap allocations, copies, conversions, or reallocations.
- No recursion, no global mutable state, no hidden ownership, no undocumented aliasing.
- No undefined-behavior tricks, no silent truncation.
- No macro-heavy fake generics unless truly justified.
- No parser/server concerns unless explicitly requested.
- Use restrict only when semantically correct.
- Use attributes ([[nodiscard]], etc.) conservatively but meaningfully.

Memory allocation rule:
- Prefer the project’s `heap.h` allocation interface whenever dynamic memory is required.
- Do not call `malloc`, `calloc`, `realloc`, or `free` directly unless there is a clear, explicitly justified reason.
- Route dynamic memory through `heap.h` so allocation behavior, tracking, limits, failure handling, and portability remain consistent with the project.
- If allocation is needed, prefer APIs that accept an explicit heap/context/allocator handle when that matches the project design.
- If no allocation is needed, prefer caller-provided buffers and explicit workspace/scratch memory.
- Do not hide heap allocation in hot paths.
- Keep allocation and deallocation responsibilities explicit.
- If a function allocates via `heap.h`, document exactly who owns the memory and how it must be released.

Inference-engine architecture:
- Separate clearly: model representation, immutable weights, runtime context, scratch memory, kernels, scheduling, token/state handling.
- Keep setup/loading separate from inference hot path.
- Distinguish: element count vs. byte count, shape vs. stride, capacity vs. used length.
- Prefer small, composable kernels with simple, predictable control flow.
- Keep data layouts and loops cache-friendly and compiler-visible; avoid abstraction layers that hurt optimization visibility.
- Numerically stable implementations for all reductions (softmax, layernorm, RMSNorm). Document precision assumptions.
- Isolate platform-specific code (NEON, SSE/AVX, CUDA) behind clear kernel boundaries.
- SIMD kernels must have scalar fallbacks. Document alignment requirements for buffers.
- Quantization: document accuracy tolerances and validate against reference implementations.

Error handling:
- Explicit error codes or tagged result structs.
- Outputs must remain well-defined on failure.
- Assertions for internal invariants; explicit error returns for external failures.

C23 usage:
- Prefer: static_assert, nullptr, constexpr objects, designated initializers, fixed-width types where width matters, inline functions over macros.
- Favor compile-time guarantees where possible.

Important build/runtime rule:
- Do not use assert().
- Do not rely on assertions for correctness, validation, or invariants.
- Assume assertions are removed in highly optimized builds.
- Express invariants through API design, explicit validation, error codes, static_assert, types, compile-time constraints, and well-defined control flow.
- If an invariant must be checked at runtime, handle it explicitly with normal control flow and defined error handling.

Testing requirements:
- Write tests as standalone test programs, not only as inline examples.
- Place new tests in `test/`.
- Follow the style and structure of the existing test programs in `test/`.
- Add boundary tests and negative tests where relevant.
- Prefer deterministic tests.
- Include tests for:
  - zero-length / empty cases where valid
  - minimal valid inputs
  - invalid arguments
  - capacity/size limit behavior
  - workspace sizing behavior
  - failure paths
  - correctness of outputs for representative cases
- Keep tests simple, explicit, and easy to run in low-level build environments.

Build:
- Code must compile cleanly under the project's strict warning flags (see build.mk).
- Code must be ASan/UBSan friendly.

Before writing code, first list briefly:
1. core data structures and buffer/array contracts
2. ownership, lifetime, and workspace/scratch model
3. error model
4. unavoidable pointer uses and why

Then provide: proposed API, implementation, performance rationale, tests (including numerical tolerance tests where applicable), and self-review against these rules.
