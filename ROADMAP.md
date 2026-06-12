# geist Roadmap

## Multi-platform distribution & the GEMM/FFT backend abstraction

**Decision (June 2026): ship per-platform static binaries via a CI matrix — *not*
a single Cosmopolitan/APE binary.** Cosmopolitan would trade away exactly what
geist exists for (per-platform SIMD + the platform's matrix accelerator) to gain
OS reach that edge inference does not need. It is also a hard blocker today:
geist's threading is OpenMP (no libgomp under Cosmopolitan) and its fast paths
use OpenBLAS / Accelerate / FFTW3, none of which link into an APE. The
industry-standard alternative (llama.cpp model) delivers the same "self-contained,
runs anywhere" experience without sacrificing a single performance path.

### Target architecture — BLAS/FFT optional, chosen per platform

The enabling refactor is **one `geist_gemm` / `geist_gemv` abstraction** that all
dense-fp32 call sites route through, with swappable backends. The same pattern
applies to the audio FFT (vDSP | pocketfft). BLAS/FFT become *optional, per
platform* — not a global "with or without".

| Platform        | Quant path  | Dense fp32             | FFT             | Binary                                  |
| :-------------- | :---------- | :--------------------- | :-------------- | :-------------------------------------- |
| **macOS-ARM**   | native int8 | Accelerate / **AMX**   | vDSP            | system-self-contained (framework always present) |
| **linux-arm64** | native int8 | **native NEON fp32**   | vendored pocketfft | **musl-static, BLAS-free, tiny**     |
| **x86-64** *(v0.2)* | AVX backend | OpenBLAS → AVX     | pocketfft       | OpenBLAS bundled (stopgap until AVX)    |

Why this maps to "fastest per platform": the quant matmuls (the bulk of text
inference) already win natively on ARM (measured: native int8 ≈ 30 t/s vs the
dequant→OpenBLAS-sgemm path ≈ 13 t/s on Pi 5). Accelerate/AMX is a real,
Apple-only hardware win for dense fp32 and is always present on macOS, so the Mac
binary keeps it for free. x86 has no native SIMD yet, so OpenBLAS is a stopgap
there until the AVX backend lands.

### Work sequence

1. **`geist_gemm` / `geist_gemv` abstraction** — route all dense-fp32 cblas call
   sites through one interface; backends selectable at build time
   (`cblas` | `native`). Foundation for everything below, including the future
   AVX backend (which slots in as a third backend).
2. **Measure first** — quantify the dense-fp32 share (`model_proj` / vision /
   audio) of the text hot path before tuning. Gate: if `model_proj` > ~10 % of
   inference, bundle OpenBLAS on linux-arm64 too instead of going BLAS-free.
3. **Native NEON fp32 GEMM/GEMV + vendored pocketfft** — unblocks the BLAS-free,
   FFTW-free build. Audio keeps vDSP on macOS, uses pocketfft on Linux.
4. **CI matrix v0.1 = ARM only** — `linux-arm64` (musl-static, Alpine CI) +
   `macos-arm64` (Accelerate). Both are already fast today — an honest,
   competitive first release.
5. **v0.2, gated on the AVX backend** — x86-64 (Linux / Intel-Mac / Windows),
   OpenBLAS stopgap → native AVX.

### Deliberate non-goals / deferred

- **Cosmopolitan / APE** — rejected (see above).
- **x86 & Windows in v0.1** — would ship slow scalar binaries (no AVX backend
  yet); a reputation risk. They wait for the AVX backend (a separate, planned
  effort the size of the current NEON backend).

### Open packaging details (not design forks)

- **Release CLI artifact** — today there is only `examples/simple_generate` and
  `tools/eval_geist`; v0.1 needs a named entry point (e.g. a `geist` CLI).
- **Windows toolchain** — MinGW vs MSVC; only relevant at v0.2.
