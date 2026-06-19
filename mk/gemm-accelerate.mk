# mk/gemm-accelerate.mk — dense fp32 GEMM via Apple Accelerate (cblas).
#
# Selected via `make GEMM_PROVIDER=accelerate` (the default on macOS targets).
# geist_gemm.c forwards geist_sgemm / geist_sgemv to cblas_sgemm / cblas_sgemv;
# Accelerate provides those symbols. No flags are added here on purpose: the
# macOS targets already link `-framework Accelerate` (and define
# HAVE_ACCELERATE) for vDSP — FFT and elementwise — so cblas comes along with
# that framework regardless of the GEMM provider. This fragment exists only to
# select the cblas path (i.e. NOT define GEIST_GEMM_NATIVE).

GEMM_CFLAGS :=
GEMM_LDLIBS :=
