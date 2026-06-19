# mk/gemm-native.mk — dependency-free dense fp32 GEMM.
#
# Selected via `make GEMM_PROVIDER=native`. Routes geist_sgemm / geist_sgemv
# through the hand-written NEON (or portable scalar) path in geist_gemm.c —
# no external math library linked. This is the provider for fully
# self-contained / static builds (e.g. the musl-static ARM CI artifact).
# Slower on the fp32 prefill matmul than a tuned BLAS; the quantized decode
# kernels are unaffected (they never use BLAS).

GEMM_CFLAGS := -DGEIST_GEMM_NATIVE
GEMM_LDLIBS :=
