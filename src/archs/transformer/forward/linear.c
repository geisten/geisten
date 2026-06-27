/*
 * src/archs/transformer/forward/linear.c - resolved-weight linear
 * dispatcher shared by transformer forward stages.
 */
#define GEIST_INTERNAL_ARCH_LAYER

#include "internal.h"

#include <geist_backend.h>

static bool tensor_is_f32_dense(const struct geist_tensor *t) {
    return t != nullptr &&
           t->buffer != nullptr &&
           t->dtype == GEIST_DTYPE_F32 &&
           t->layout == GEIST_LAYOUT_DENSE;
}

static bool tensor_2d_is_single_row(const struct geist_tensor *t,
                                    int64_t *out_n) {
    if (!tensor_is_f32_dense(t) || out_n == nullptr) {
        return false;
    }
    if (t->ndim == 1 && t->shape[0] > 0) {
        *out_n = t->shape[0];
        return true;
    }
    if (t->ndim == 2 && t->shape[0] == 1 && t->shape[1] > 0) {
        *out_n = t->shape[1];
        return true;
    }
    return false;
}

static bool can_use_matvec_f32_dense(
    const struct geist_backend_vtbl *v,
    size_t seq,
    const struct geist_tensor *t_x,
    const struct geist_tensor *t_w,
    const struct geist_tensor *t_y) {

    if (v == nullptr || v->matvec_f32_dense == nullptr || seq != 1 ||
        !tensor_is_f32_dense(t_w) || t_w->ndim != 2 ||
        t_w->shape[0] <= 0 || t_w->shape[1] <= 0) {
        return false;
    }
    int64_t n_in = 0;
    int64_t n_out = 0;
    if (!tensor_2d_is_single_row(t_x, &n_in) ||
        !tensor_2d_is_single_row(t_y, &n_out)) {
        return false;
    }
    return n_in == t_w->shape[1] && n_out == t_w->shape[0];
}

static bool can_use_matvec_q4k(
    const struct geist_backend_vtbl *v,
    size_t seq,
    const struct geist_tensor *t_x,
    const struct geist_tensor *t_w,
    const struct geist_tensor *t_y) {

    if (v == nullptr || v->matvec_q4k == nullptr || seq != 1 ||
        t_w == nullptr || t_w->buffer == nullptr ||
        t_w->dtype != GEIST_DTYPE_Q4_K ||
        t_w->layout != GEIST_LAYOUT_BLOCK_QUANTIZED ||
        t_w->ndim != 2 || t_w->shape[0] <= 0 || t_w->shape[1] <= 0) {
        return false;
    }
    int64_t n_in = 0;
    int64_t n_out = 0;
    if (!tensor_2d_is_single_row(t_x, &n_in) ||
        !tensor_2d_is_single_row(t_y, &n_out)) {
        return false;
    }
    return n_in == t_w->shape[1] && n_out == t_w->shape[0];
}

static bool can_use_matvec_q6k(
    const struct geist_backend_vtbl *v,
    size_t seq,
    const struct geist_tensor *t_x,
    const struct geist_tensor *t_w,
    const struct geist_tensor *t_y) {

    if (v == nullptr || v->matvec_q6k == nullptr || seq != 1 ||
        t_w == nullptr || t_w->buffer == nullptr ||
        t_w->dtype != GEIST_DTYPE_Q6_K ||
        t_w->layout != GEIST_LAYOUT_BLOCK_QUANTIZED ||
        t_w->ndim != 2 || t_w->shape[0] <= 0 || t_w->shape[1] <= 0) {
        return false;
    }
    int64_t n_in = 0;
    int64_t n_out = 0;
    if (!tensor_2d_is_single_row(t_x, &n_in) ||
        !tensor_2d_is_single_row(t_y, &n_out)) {
        return false;
    }
    return n_in == t_w->shape[1] && n_out == t_w->shape[0];
}

static bool can_use_matmul_f32_dense(
    const struct geist_backend_vtbl *v,
    size_t seq,
    const struct geist_tensor *t_x,
    const struct geist_tensor *t_w,
    const struct geist_tensor *t_y) {

    return v != nullptr && v->matmul_f32_dense != nullptr && seq > 1 &&
           tensor_is_f32_dense(t_x) &&
           tensor_is_f32_dense(t_w) &&
           tensor_is_f32_dense(t_y) &&
           t_x->ndim == 2 && t_w->ndim == 2 && t_y->ndim == 2 &&
           t_x->shape[0] == (int64_t) seq &&
           t_y->shape[0] == (int64_t) seq &&
           t_x->shape[1] > 0 &&
           t_w->shape[0] > 0 &&
           t_w->shape[1] == t_x->shape[1] &&
           t_y->shape[1] == t_w->shape[0];
}

static bool can_use_matmul_qk(
    const struct geist_backend_vtbl *v,
    size_t seq,
    const struct geist_tensor *t_x,
    const struct geist_tensor *t_w,
    const struct geist_tensor *t_y) {

    if (v == nullptr || seq <= 1 ||
        t_x == nullptr || t_w == nullptr || t_y == nullptr ||
        t_x->buffer == nullptr || t_w->buffer == nullptr ||
        t_y->buffer == nullptr ||
        t_x->dtype != GEIST_DTYPE_F32 ||
        t_x->layout != GEIST_LAYOUT_DENSE ||
        t_y->dtype != GEIST_DTYPE_F32 ||
        t_y->layout != GEIST_LAYOUT_DENSE ||
        t_w->layout != GEIST_LAYOUT_BLOCK_QUANTIZED ||
        t_x->ndim != 2 || t_w->ndim != 2 || t_y->ndim != 2 ||
        t_x->shape[0] != (int64_t) seq ||
        t_y->shape[0] != (int64_t) seq ||
        t_x->shape[1] <= 0 ||
        t_w->shape[0] <= 0 ||
        t_w->shape[1] != t_x->shape[1] ||
        t_y->shape[1] != t_w->shape[0]) {
        return false;
    }
    if (t_w->dtype == GEIST_DTYPE_Q4_K) {
        return v->matmul_q4k != nullptr;
    }
    if (t_w->dtype == GEIST_DTYPE_Q6_K) {
        return v->matmul_q6k != nullptr;
    }
    return false;
}

static struct geist_tensor matvec_1d_view(const struct geist_tensor *t,
                                          int64_t n) {
    struct geist_tensor out = *t;
    out.ndim = 1;
    out.shape[0] = n;
    for (int i = 1; i < 8; i++) {
        out.shape[i] = 0;
    }
    out.stride[0] = 1;
    for (int i = 1; i < 8; i++) {
        out.stride[i] = 0;
    }
    return out;
}

static enum geist_status try_matvec_f32_dense(
    struct geist_backend *be,
    const struct geist_backend_vtbl *v,
    size_t seq,
    const struct geist_tensor *t_x,
    const struct geist_tensor *t_w,
    struct geist_tensor *t_y,
    bool *out_handled) {

    if (out_handled == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    *out_handled = false;
    if (!can_use_matvec_f32_dense(v, seq, t_x, t_w, t_y)) {
        return GEIST_OK;
    }

    struct geist_tensor x = matvec_1d_view(t_x, t_w->shape[1]);
    struct geist_tensor y = matvec_1d_view(t_y, t_w->shape[0]);
    const enum geist_status s = v->matvec_f32_dense(be, &x, t_w, &y);
    if (s == GEIST_OK) {
        *out_handled = true;
    }
    return s;
}

static enum geist_status try_matvec_q4k(
    struct geist_backend *be,
    const struct geist_backend_vtbl *v,
    size_t seq,
    const struct geist_tensor *t_x,
    const struct geist_tensor *t_w,
    struct geist_tensor *t_y,
    bool *out_handled) {

    if (out_handled == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    *out_handled = false;
    if (!can_use_matvec_q4k(v, seq, t_x, t_w, t_y)) {
        return GEIST_OK;
    }

    struct geist_tensor x = matvec_1d_view(t_x, t_w->shape[1]);
    struct geist_tensor y = matvec_1d_view(t_y, t_w->shape[0]);
    const enum geist_status s = v->matvec_q4k(be, &x, t_w, &y);
    if (s == GEIST_OK) {
        *out_handled = true;
    }
    return s;
}

static enum geist_status try_matvec_q6k(
    struct geist_backend *be,
    const struct geist_backend_vtbl *v,
    size_t seq,
    const struct geist_tensor *t_x,
    const struct geist_tensor *t_w,
    struct geist_tensor *t_y,
    bool *out_handled) {

    if (out_handled == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    *out_handled = false;
    if (!can_use_matvec_q6k(v, seq, t_x, t_w, t_y)) {
        return GEIST_OK;
    }

    struct geist_tensor x = matvec_1d_view(t_x, t_w->shape[1]);
    struct geist_tensor y = matvec_1d_view(t_y, t_w->shape[0]);
    const enum geist_status s = v->matvec_q6k(be, &x, t_w, &y);
    if (s == GEIST_OK) {
        *out_handled = true;
    }
    return s;
}

static enum geist_status try_matmul_f32_dense(
    struct geist_backend *be,
    const struct geist_backend_vtbl *v,
    size_t seq,
    const struct geist_tensor *t_x,
    const struct geist_tensor *t_w,
    struct geist_tensor *t_y,
    bool *out_handled) {

    if (out_handled == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    *out_handled = false;
    if (!can_use_matmul_f32_dense(v, seq, t_x, t_w, t_y)) {
        return GEIST_OK;
    }

    const enum geist_status s = v->matmul_f32_dense(be, t_x, t_w, t_y);
    if (s == GEIST_OK) {
        *out_handled = true;
    }
    return s;
}

static enum geist_status try_matmul_qk(
    struct geist_backend *be,
    const struct geist_backend_vtbl *v,
    size_t seq,
    const struct geist_tensor *t_x,
    const struct geist_tensor *t_w,
    struct geist_tensor *t_y,
    bool *out_handled) {

    if (out_handled == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    *out_handled = false;
    if (!can_use_matmul_qk(v, seq, t_x, t_w, t_y)) {
        return GEIST_OK;
    }

    enum geist_status s = GEIST_E_UNSUPPORTED;
    if (t_w->dtype == GEIST_DTYPE_Q4_K) {
        s = v->matmul_q4k(be, t_x, t_w, t_y);
    } else if (t_w->dtype == GEIST_DTYPE_Q6_K) {
        s = v->matmul_q6k(be, t_x, t_w, t_y);
    }
    if (s == GEIST_OK) {
        *out_handled = true;
    }
    return s;
}

static enum geist_status try_device_matvec(
    struct geist_backend *be,
    const struct geist_backend_vtbl *v,
    size_t seq,
    const struct geist_tensor *t_x,
    const struct geist_tensor *t_w,
    struct geist_tensor *t_y,
    bool *out_handled) {

    enum geist_status s = try_matvec_f32_dense(be, v, seq, t_x, t_w, t_y,
                                                out_handled);
    if (s != GEIST_OK || (out_handled != nullptr && *out_handled)) {
        return s;
    }
    s = try_matvec_q4k(be, v, seq, t_x, t_w, t_y, out_handled);
    if (s != GEIST_OK || (out_handled != nullptr && *out_handled)) {
        return s;
    }
    s = try_matmul_f32_dense(be, v, seq, t_x, t_w, t_y, out_handled);
    if (s != GEIST_OK || (out_handled != nullptr && *out_handled)) {
        return s;
    }
    s = try_matmul_qk(be, v, seq, t_x, t_w, t_y, out_handled);
    if (s != GEIST_OK || (out_handled != nullptr && *out_handled)) {
        return s;
    }
    return try_matvec_q6k(be, v, seq, t_x, t_w, t_y, out_handled);
}

enum geist_status linear_w_or_legacy(
    struct geist_backend *be,
    const struct geist_backend_vtbl *v,
    struct geist_buffer *x_buf, struct geist_buffer *y_buf,
    const struct geist_weight *w,
    size_t seq,
    const struct geist_tensor *t_x, const struct geist_tensor *t_w,
    struct geist_tensor *t_y) {

    bool handled = false;
    enum geist_status s = try_device_matvec(be, v, seq, t_x, t_w, t_y,
                                            &handled);
    if (s != GEIST_OK || handled) {
        return s;
    }

    if (w == nullptr) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG,
                                "linear_w: null geist_weight");
        return GEIST_E_INVALID_ARG;
    }
    if ((seq == 1 && w->linear_m1 == nullptr) ||
        (seq >  1 && w->linear_mN == nullptr)) {
        geist_backend_set_error(be, GEIST_E_UNSUPPORTED,
                                "linear_w: backend resolver installed no kernel "
                                "for dtype=%u, seq=%zu (legacy v->linear() path "
                                "retired in P2.e)", (unsigned) w->dtype, seq);
        return GEIST_E_UNSUPPORTED;
    }
    const float *xp = (const float *) v->buffer_map(x_buf);
    float *yp = (float *) v->buffer_map(y_buf);
    if (xp == nullptr || yp == nullptr) {
        return GEIST_E_BACKEND;
    }
    /* Pass `be` so the kernel can reach its backend's workspace
     * (cpu_neon q8a scratch, etc.) without consulting file-scope TLS.
     * Engine guarantees `be->state` is valid for the lifetime of this
     * call — see the resolver fail-fast check at cpu_neon_resolve_weight. */
    if (seq == 1) {
        w->linear_m1(xp, w, be, yp);
    } else {
        w->linear_mN(xp, w, seq, be, yp);
    }
    v->buffer_unmap(x_buf);
    v->buffer_unmap(y_buf);
    return GEIST_OK;
}

enum geist_status linear_w_no_host_fallback(
    struct geist_backend *be,
    const struct geist_backend_vtbl *v,
    size_t seq,
    const struct geist_tensor *t_x, const struct geist_tensor *t_w,
    struct geist_tensor *t_y) {

    bool handled = false;
    enum geist_status s = try_device_matvec(be, v, seq, t_x, t_w, t_y,
                                            &handled);
    if (s != GEIST_OK || handled) {
        return s;
    }
    geist_backend_set_error(be, GEIST_E_UNSUPPORTED,
                            "linear_w: host fallback disabled during command "
                            "sequence");
    return GEIST_E_UNSUPPORTED;
}

enum geist_status linear_w_scaled_input_or_legacy(
    struct geist_backend *be,
    const struct geist_backend_vtbl *v,
    struct geist_buffer *x_buf, struct geist_buffer *y_buf,
    const struct geist_weight *w,
    size_t seq,
    size_t scale_n,
    const float *scale,
    const struct geist_tensor *t_x, const struct geist_tensor *t_w,
    struct geist_tensor *t_y) {

    apply_per_channel_inv_scale_inplace(v, x_buf, seq, scale_n, scale);
    return linear_w_or_legacy(be, v, x_buf, y_buf, w, seq, t_x, t_w, t_y);
}

enum geist_status linear_w_pair_or_legacy(
    struct geist_backend *be,
    const struct geist_backend_vtbl *v,
    struct geist_buffer *x_buf,
    struct geist_buffer *y0_buf, struct geist_buffer *y1_buf,
    const struct geist_weight *w0, const struct geist_weight *w1,
    size_t seq,
    const struct geist_tensor *t_x,
    const struct geist_tensor *t_w0, const struct geist_tensor *t_w1,
    struct geist_tensor *t_y0, struct geist_tensor *t_y1) {

    bool handled0 = false;
    bool handled1 = false;
    enum geist_status s = try_device_matvec(be, v, seq, t_x, t_w0,
                                            t_y0, &handled0);
    if (s != GEIST_OK) {
        return s;
    }
    s = try_device_matvec(be, v, seq, t_x, t_w1, t_y1, &handled1);
    if (s != GEIST_OK) {
        return s;
    }
    if (handled0 && handled1) {
        return GEIST_OK;
    }

    if (w0 == nullptr || w1 == nullptr) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG,
                                "linear_w_pair: null geist_weight");
        return GEIST_E_INVALID_ARG;
    }
    if ((seq == 1 && (w0->linear_m1 == nullptr || w1->linear_m1 == nullptr)) ||
        (seq >  1 && (w0->linear_mN == nullptr || w1->linear_mN == nullptr))) {
        geist_backend_set_error(be, GEIST_E_UNSUPPORTED,
                                "linear_w_pair: resolver installed no paired "
                                "kernel for seq=%zu", seq);
        return GEIST_E_UNSUPPORTED;
    }

    const float *xp = (const float *) v->buffer_map(x_buf);
    float *y0p = (float *) v->buffer_map(y0_buf);
    float *y1p = (float *) v->buffer_map(y1_buf);
    if (xp == nullptr || y0p == nullptr || y1p == nullptr) {
        if (xp != nullptr) { v->buffer_unmap(x_buf); }
        if (y0p != nullptr) { v->buffer_unmap(y0_buf); }
        if (y1p != nullptr) { v->buffer_unmap(y1_buf); }
        return GEIST_E_BACKEND;
    }
    if (seq == 1) {
        if (w0->linear_pair_m1 != nullptr &&
            w0->linear_pair_m1 == w1->linear_pair_m1 &&
            w0->n_in == w1->n_in) {
            w0->linear_pair_m1(xp, w0, w1, be, y0p, y1p);
        } else {
            w0->linear_m1(xp, w0, be, y0p);
            w1->linear_m1(xp, w1, be, y1p);
        }
    } else {
        if (w0->linear_pair_mN != nullptr &&
            w0->linear_pair_mN == w1->linear_pair_mN &&
            w0->n_in == w1->n_in) {
            w0->linear_pair_mN(xp, w0, w1, seq, be, y0p, y1p);
        } else {
            w0->linear_mN(xp, w0, seq, be, y0p);
            w1->linear_mN(xp, w1, seq, be, y1p);
        }
    }
    v->buffer_unmap(x_buf);
    v->buffer_unmap(y0_buf);
    v->buffer_unmap(y1_buf);
    return GEIST_OK;
}

enum geist_status linear_w_triple_or_legacy(
    struct geist_backend *be,
    const struct geist_backend_vtbl *v,
    struct geist_buffer *x_buf,
    struct geist_buffer *y0_buf, struct geist_buffer *y1_buf,
    struct geist_buffer *y2_buf,
    const struct geist_weight *w0, const struct geist_weight *w1,
    const struct geist_weight *w2,
    size_t seq,
    const struct geist_tensor *t_x,
    const struct geist_tensor *t_w0, const struct geist_tensor *t_w1,
    const struct geist_tensor *t_w2,
    struct geist_tensor *t_y0, struct geist_tensor *t_y1,
    struct geist_tensor *t_y2) {

    bool handled0 = false;
    bool handled1 = false;
    bool handled2 = false;
    enum geist_status s =
        try_device_matvec(be, v, seq, t_x, t_w0, t_y0, &handled0);
    if (s != GEIST_OK) {
        return s;
    }
    s = try_device_matvec(be, v, seq, t_x, t_w1, t_y1, &handled1);
    if (s != GEIST_OK) {
        return s;
    }
    s = try_device_matvec(be, v, seq, t_x, t_w2, t_y2, &handled2);
    if (s != GEIST_OK) {
        return s;
    }
    if (handled0 && handled1 && handled2) {
        return GEIST_OK;
    }

    if (w0 == nullptr || w1 == nullptr || w2 == nullptr) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG,
                                "linear_w_triple: null geist_weight");
        return GEIST_E_INVALID_ARG;
    }
    if ((seq == 1 && (w0->linear_m1 == nullptr ||
                      w1->linear_m1 == nullptr ||
                      w2->linear_m1 == nullptr)) ||
        (seq >  1 && (w0->linear_mN == nullptr ||
                      w1->linear_mN == nullptr ||
                      w2->linear_mN == nullptr))) {
        geist_backend_set_error(be, GEIST_E_UNSUPPORTED,
                                "linear_w_triple: resolver installed no "
                                "kernel for seq=%zu", seq);
        return GEIST_E_UNSUPPORTED;
    }

    const float *xp = (const float *) v->buffer_map(x_buf);
    float *y0p = (float *) v->buffer_map(y0_buf);
    float *y1p = (float *) v->buffer_map(y1_buf);
    float *y2p = (float *) v->buffer_map(y2_buf);
    if (xp == nullptr || y0p == nullptr || y1p == nullptr || y2p == nullptr) {
        if (xp != nullptr) { v->buffer_unmap(x_buf); }
        if (y0p != nullptr) { v->buffer_unmap(y0_buf); }
        if (y1p != nullptr) { v->buffer_unmap(y1_buf); }
        if (y2p != nullptr) { v->buffer_unmap(y2_buf); }
        return GEIST_E_BACKEND;
    }

    if (seq == 1) {
        if (w0->linear_pair_m1 != nullptr &&
            w0->linear_pair_m1 == w1->linear_pair_m1 &&
            w0->n_in == w1->n_in) {
            w0->linear_pair_m1(xp, w0, w1, be, y0p, y1p);
        } else {
            w0->linear_m1(xp, w0, be, y0p);
            w1->linear_m1(xp, w1, be, y1p);
        }
        w2->linear_m1(xp, w2, be, y2p);
    } else if (w0->linear_triple_mN != nullptr &&
               w0->linear_triple_mN == w1->linear_triple_mN &&
               w0->linear_triple_mN == w2->linear_triple_mN &&
               w0->n_in == w1->n_in &&
               w0->n_in == w2->n_in) {
        w0->linear_triple_mN(xp, w0, w1, w2, seq, be, y0p, y1p, y2p);
    } else if (w1->linear_pair_mN != nullptr &&
               w1->linear_pair_mN == w2->linear_pair_mN &&
               w1->n_in == w2->n_in) {
        w0->linear_mN(xp, w0, seq, be, y0p);
        w1->linear_pair_mN(xp, w1, w2, seq, be, y1p, y2p);
    } else {
        w0->linear_mN(xp, w0, seq, be, y0p);
        w1->linear_mN(xp, w1, seq, be, y1p);
        w2->linear_mN(xp, w2, seq, be, y2p);
    }

    v->buffer_unmap(x_buf);
    v->buffer_unmap(y0_buf);
    v->buffer_unmap(y1_buf);
    v->buffer_unmap(y2_buf);
    return GEIST_OK;
}
