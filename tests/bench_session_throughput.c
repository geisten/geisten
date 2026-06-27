/*
 * bench_session_throughput — measures end-to-end prefill + decode wall
 * through the new geist_session_* public API.
 *
 * Workload:
 *   - Load model (cold; once-per-run cost)
 *   - Prefill 64 tokens (warm-up)
 *   - Prefill 200 fresh tokens (measured)
 *   - Decode 50 tokens (measured)
 *
 * Reports: model_load ms, prefill 200t ms (and ms/tok), decode 50t ms
 * (and ms/tok), throughput tok/s for both phases.
 *
 * Compares directly to the existing eval_geist baseline since both
 * paths call into the same LM*; numbers should match within noise.
 */
#include "test_helpers.h"

#include <geist.h>
#include <geist_util.h>
#include <geist_backend.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double monotonic_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec * 1e3 + (double) ts.tv_nsec / 1e6;
}

/* Workload sizes are env-overridable so the run can be matched exactly to an
 * external reference (e.g. `llama-bench -p 512 -n 128`): set GEIST_BENCH_PP and
 * GEIST_BENCH_TG. Defaults keep the historical 200/50 workload. */
static size_t env_size(const char* name, size_t fallback) {
    const char* raw = getenv(name);
    if (raw == nullptr || raw[0] == '\0')
        return fallback;
    long v = atol(raw);
    return v > 0 ? (size_t) v : fallback;
}

static bool env_enabled(const char *name) {
    const char *raw = getenv(name);
    return raw != nullptr && raw[0] != '\0' && strcmp(raw, "0") != 0;
}

static enum geist_status create_bench_backend(struct geist_backend **out) {
    const char *requested = getenv("GEIST_BENCH_BACKEND");
    if (requested != nullptr && requested[0] != '\0' &&
        strcmp(requested, "auto") != 0) {
        return geist_backend_create(requested, nullptr, nullptr, out);
    }

    enum geist_status s =
        geist_backend_create("cpu_neon", nullptr, nullptr, out);
    if (s != GEIST_OK) {
        s = geist_backend_create("cpu_scalar", nullptr, nullptr, out);
    }
    return s;
}

static void print_accel_caps(struct geist_backend *be) {
    if (be == nullptr || be->desc == nullptr || be->desc->vtbl == nullptr ||
        be->desc->vtbl->query_accel_caps == nullptr) {
        printf("  accel:             host backend\n");
        return;
    }

    struct geist_backend_accel_caps caps = {
        .struct_size = sizeof(caps),
    };
    enum geist_status s = be->desc->vtbl->query_accel_caps(be, &caps);
    if (s != GEIST_OK) {
        printf("  accel:             unavailable\n");
        return;
    }
    printf("  accel:             device=%s device_resident=%s compute=%s pipeline_cache=%s subgroup=%s dot_product=%s descriptor_indexing=%s timeline=%s\n",
           caps.device_name[0] != '\0' ? caps.device_name : "(unknown)",
           caps.device_resident_buffers ? "yes" : "no",
           caps.compute_queue ? "yes" : "no",
           caps.pipeline_cache ? "yes" : "no",
           caps.subgroup_basic ? "yes" : "no",
           caps.shader_integer_dot_product ? "yes" : "no",
           caps.descriptor_indexing ? "yes" : "no",
           caps.timeline_semaphore ? "yes" : "no");
}

static enum geist_status backend_profile_reset(struct geist_backend *be) {
    if (be == nullptr || be->desc == nullptr || be->desc->vtbl == nullptr ||
        be->desc->vtbl->profile_reset == nullptr) {
        return GEIST_OK;
    }
    return be->desc->vtbl->profile_reset(be);
}

static enum geist_status backend_profile_dump(struct geist_backend *be) {
    if (be == nullptr || be->desc == nullptr || be->desc->vtbl == nullptr ||
        be->desc->vtbl->profile_dump == nullptr) {
        return GEIST_OK;
    }
    return be->desc->vtbl->profile_dump(be);
}

static void fill_prefill_ids(size_t n, geist_token_t ids[static n]) {
    for (size_t i = 0; i < n; i++) {
        ids[i] = 2 + (geist_token_t) ((i * 37) & 0xff);
    }
}

static enum geist_status run_decode_token_stream(
    const char *model_path,
    size_t prefill_n,
    size_t decode_n,
    geist_token_t tokens[static decode_n],
    uint64_t *out_checksum) {

    if (model_path == nullptr || tokens == nullptr || out_checksum == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    *out_checksum = 0;

    struct geist_backend *be = nullptr;
    enum geist_status s = create_bench_backend(&be);
    if (s != GEIST_OK) {
        fprintf(stderr, "validation backend create failed: %s\n",
                geist_last_create_error());
        return s;
    }

    struct geist_model *model = nullptr;
    s = geist_model_load(model_path, be, &model);
    if (s != GEIST_OK) {
        fprintf(stderr, "validation model_load failed: %s\n",
                geist_last_create_error());
        geist_backend_destroy(be);
        return s;
    }

    struct geist_session_opts opts = {.max_seq_len = 2048, .temperature = 0.0f};
    struct geist_session *sess = nullptr;
    s = geist_session_create(model, be, &opts, &sess);
    if (s != GEIST_OK) {
        fprintf(stderr, "validation session_create failed\n");
        geist_model_destroy(model);
        geist_backend_destroy(be);
        return s;
    }

    geist_token_t *prefill_ids = malloc(prefill_n * sizeof(geist_token_t));
    if (prefill_ids == nullptr) {
        geist_session_destroy(sess);
        geist_model_destroy(model);
        geist_backend_destroy(be);
        return GEIST_E_OOM;
    }
    fill_prefill_ids(prefill_n, prefill_ids);
    s = geist_session_prefill_tokens(sess, prefill_n, prefill_ids);
    free(prefill_ids);
    if (s != GEIST_OK) {
        fprintf(stderr, "validation prefill failed: %s\n",
                geist_session_errmsg(sess));
        geist_session_destroy(sess);
        geist_model_destroy(model);
        geist_backend_destroy(be);
        return s;
    }

    uint64_t checksum = 0;
    for (size_t i = 0; i < decode_n; i++) {
        geist_token_t tok = 0;
        s = geist_session_decode_step(sess, &tok);
        if (s != GEIST_OK) {
            fprintf(stderr, "validation decode_step[%zu] failed: %s\n",
                    i, geist_session_errmsg(sess));
            geist_session_destroy(sess);
            geist_model_destroy(model);
            geist_backend_destroy(be);
            return s;
        }
        tokens[i] = tok;
        checksum = checksum * 1315423911u + (uint32_t) tok + (uint32_t) i;
    }
    *out_checksum = checksum;

    geist_session_destroy(sess);
    geist_model_destroy(model);
    geist_backend_destroy(be);
    return GEIST_OK;
}

static char *copy_env_value(const char *name) {
    const char *raw = getenv(name);
    if (raw == nullptr) {
        return nullptr;
    }
    const size_t n = strlen(raw) + 1u;
    char *copy = malloc(n);
    if (copy != nullptr) {
        memcpy(copy, raw, n);
    }
    return copy;
}

static const char *replay_env_for_backend(const char *backend_name) {
    if (backend_name == nullptr) {
        return nullptr;
    }
    if (strcmp(backend_name, "vulkan") == 0) {
        return "GEIST_VULKAN_DECODE_REPLAY";
    }
    if (strcmp(backend_name, "metal") == 0) {
        return "GEIST_METAL_DECODE_REPLAY";
    }
    return nullptr;
}

static int validate_replay_against_no_replay(
    const char *backend_name,
    const char *model_path,
    size_t prefill_n,
    size_t decode_n,
    const geist_token_t measured_tokens[static decode_n],
    uint64_t measured_checksum) {

    if (decode_n == 0) {
        printf("  replay validation: skipped (decode=0)\n");
        return GEIST_TEST_PASS;
    }
    if (decode_n < 2) {
        fprintf(stderr,
                "replay validation requires GEIST_BENCH_TG >= 2 so at least one replay token is exercised\n");
        return GEIST_TEST_FAIL;
    }
    const char *replay_env_name = replay_env_for_backend(backend_name);
    if (replay_env_name == nullptr) {
        printf("  replay validation: skipped (backend has no replay env)\n");
        return GEIST_TEST_PASS;
    }
    const char *replay_env = getenv(replay_env_name);
    if (replay_env != nullptr && strcmp(replay_env, "0") == 0) {
        fprintf(stderr,
                "replay validation requires the measured run to keep %s enabled\n",
                replay_env_name);
        return GEIST_TEST_FAIL;
    }

    char *saved_replay_env = copy_env_value(replay_env_name);
    const bool had_replay_env = getenv(replay_env_name) != nullptr;
    if (had_replay_env && saved_replay_env == nullptr) {
        fprintf(stderr, "replay validation: env copy allocation failed\n");
        return GEIST_TEST_ERROR;
    }
    if (setenv(replay_env_name, "0", 1) != 0) {
        free(saved_replay_env);
        fprintf(stderr, "replay validation: setenv failed\n");
        return GEIST_TEST_ERROR;
    }

    geist_token_t *reference_tokens =
        malloc(decode_n * sizeof(geist_token_t));
    if (reference_tokens == nullptr) {
        if (had_replay_env) {
            (void) setenv(replay_env_name,
                          saved_replay_env != nullptr ? saved_replay_env : "",
                          1);
        } else {
            (void) unsetenv(replay_env_name);
        }
        free(saved_replay_env);
        return GEIST_TEST_ERROR;
    }

    uint64_t reference_checksum = 0;
    enum geist_status s = run_decode_token_stream(
        model_path, prefill_n, decode_n, reference_tokens,
        &reference_checksum);

    if (had_replay_env) {
        (void) setenv(replay_env_name,
                      saved_replay_env != nullptr ? saved_replay_env : "", 1);
    } else {
        (void) unsetenv(replay_env_name);
    }
    free(saved_replay_env);

    if (s != GEIST_OK) {
        free(reference_tokens);
        fprintf(stderr, "replay validation failed to run reference: %s\n",
                geist_status_to_string(s));
        return GEIST_TEST_FAIL;
    }

    int result = GEIST_TEST_PASS;
    for (size_t i = 0; i < decode_n; i++) {
        if (measured_tokens[i] != reference_tokens[i]) {
            fprintf(stderr,
                    "replay validation mismatch at token %zu: replay=%d no_replay=%d\n",
                    i, measured_tokens[i], reference_tokens[i]);
            result = GEIST_TEST_FAIL;
            break;
        }
    }
    if (result == GEIST_TEST_PASS && measured_checksum != reference_checksum) {
        fprintf(stderr,
                "replay validation checksum mismatch: replay=0x%016llx no_replay=0x%016llx\n",
                (unsigned long long) measured_checksum,
                (unsigned long long) reference_checksum);
        result = GEIST_TEST_FAIL;
    }
    if (result == GEIST_TEST_PASS) {
        printf("  replay validation: matched %zu tokens vs %s=0 checksum=0x%016llx\n",
               decode_n, replay_env_name,
               (unsigned long long) reference_checksum);
    }
    free(reference_tokens);
    return result;
}

int main(void) {
    GEIST_REQUIRE_GGUF(model_path);

    /* ---- Setup ---- */
    struct geist_backend* be = nullptr;
    enum geist_status s = create_bench_backend(&be);
    if (s != GEIST_OK) {
        fprintf(stderr, "backend create failed: %s\n", geist_last_create_error());
        return GEIST_TEST_ERROR;
    }

    double t0 = monotonic_ms();
    struct geist_model* model = nullptr;
    s = geist_model_load(model_path, be, &model);
    double t_load = monotonic_ms() - t0;
    if (s != GEIST_OK) {
        fprintf(stderr, "model_load failed: %s\n", geist_last_create_error());
        geist_backend_destroy(be);
        return GEIST_TEST_FAIL;
    }

    struct geist_session_opts opts = {.max_seq_len = 2048, .temperature = 0.0f};
    struct geist_session* sess = nullptr;
    s = geist_session_create(model, be, &opts, &sess);
    if (s != GEIST_OK) {
        fprintf(stderr, "session_create failed\n");
        geist_model_destroy(model);
        geist_backend_destroy(be);
        return GEIST_TEST_FAIL;
    }

    /* ---- Warm-up: prefill 64 tokens ---- */
    const size_t warm_n = 64;
    geist_token_t* warm_ids = malloc(warm_n * sizeof(geist_token_t));
    if (warm_ids == nullptr) {
        fprintf(stderr, "warmup token allocation failed\n");
        goto fail;
    }
    for (size_t i = 0; i < warm_n; i++) {
        warm_ids[i] = 2 + (geist_token_t) (i & 0xff);
    }
    s = geist_session_prefill_tokens(sess, warm_n, warm_ids);
    if (s != GEIST_OK) {
        fprintf(stderr, "warmup prefill failed: %s\n", geist_session_errmsg(sess));
        goto fail;
    }
    free(warm_ids);

    s = geist_session_reset(sess);
    if (s != GEIST_OK) {
        fprintf(stderr, "reset failed\n");
        goto fail;
    }
    s = geist_session_reset_stats(sess);
    if (s != GEIST_OK) {
        fprintf(stderr, "reset_stats failed\n");
        goto fail;
    }
    const bool dump_backend_profile =
        env_enabled("GEIST_BENCH_DUMP_BACKEND_PROFILE");
    const bool profile_decode_only =
        env_enabled("GEIST_BENCH_PROFILE_DECODE_ONLY");
    if (dump_backend_profile) {
        s = backend_profile_reset(be);
        if (s != GEIST_OK) {
            fprintf(stderr, "backend profile reset failed\n");
            goto fail;
        }
    }

    /* ---- Measured: prefill (GEIST_BENCH_PP, default 200) ---- */
    const size_t prefill_n = env_size("GEIST_BENCH_PP", 200);
    geist_token_t* prefill_ids = malloc(prefill_n * sizeof(geist_token_t));
    if (prefill_ids == nullptr) {
        fprintf(stderr, "prefill token allocation failed\n");
        goto fail;
    }
    fill_prefill_ids(prefill_n, prefill_ids);
    t0 = monotonic_ms();
    s = geist_session_prefill_tokens(sess, prefill_n, prefill_ids);
    double t_prefill = monotonic_ms() - t0;
    if (s != GEIST_OK) {
        fprintf(stderr, "prefill failed: %s\n", geist_session_errmsg(sess));
        free(prefill_ids);
        goto fail;
    }
    free(prefill_ids);

    if (dump_backend_profile && profile_decode_only) {
        s = backend_profile_reset(be);
        if (s != GEIST_OK) {
            fprintf(stderr, "backend profile decode reset failed\n");
            goto fail;
        }
    }

    /* ---- Measured: decode (GEIST_BENCH_TG, default 50) ---- */
    const size_t decode_n = env_size("GEIST_BENCH_TG", 50);
    geist_token_t *decoded_tokens = malloc(decode_n * sizeof(geist_token_t));
    if (decoded_tokens == nullptr) {
        fprintf(stderr, "decode token allocation failed\n");
        goto fail;
    }
    geist_token_t tok = 0;
    uint64_t token_checksum = 0;
    double t_decode_first = 0.0;
    double t_decode_rest = 0.0;
    t0 = monotonic_ms();
    for (size_t i = 0; i < decode_n; i++) {
        const double t_step = monotonic_ms();
        s = geist_session_decode_step(sess, &tok);
        const double dt_step = monotonic_ms() - t_step;
        if (s != GEIST_OK) {
            fprintf(stderr, "decode_step[%zu] failed: %s\n",
                    i, geist_session_errmsg(sess));
            free(decoded_tokens);
            goto fail;
        }
        if (i == 0) {
            t_decode_first = dt_step;
        } else {
            t_decode_rest += dt_step;
        }
        decoded_tokens[i] = tok;
        token_checksum =
            token_checksum * 1315423911u + (uint32_t) tok + (uint32_t) i;
    }
    double t_decode = monotonic_ms() - t0;

    /* ---- Report ---- */
    printf("Backend: %s  Model: %s\n", geist_backend_name(be), model_path);
    printf("  workload:          prefill=%zu decode=%zu greedy=true backend_env=%s\n",
           prefill_n,
           decode_n,
           getenv("GEIST_BENCH_BACKEND") != nullptr
               ? getenv("GEIST_BENCH_BACKEND")
               : "auto");
    print_accel_caps(be);
    printf("  model_load:        %8.1f ms  (cold)\n", t_load);
    printf("  prefill (%zu tok):  %8.1f ms  =  %5.2f ms/tok  =  %6.1f tok/s\n",
           prefill_n,
           t_prefill,
           t_prefill / (double) prefill_n,
           (double) prefill_n / (t_prefill / 1e3));
    printf("  decode  (%zu tok):   %8.1f ms  =  %5.2f ms/tok  =  %6.1f tok/s\n",
           decode_n,
           t_decode,
           decode_n > 0 ? t_decode / (double) decode_n : 0.0,
           (double) decode_n / (t_decode / 1e3));
    if (decode_n > 1) {
        printf("  decode split:       first=%5.2f ms  rest_avg=%5.2f ms/tok  rest=%6.1f tok/s\n",
               t_decode_first,
               t_decode_rest / (double) (decode_n - 1),
               (double) (decode_n - 1) / (t_decode_rest / 1e3));
    }
    printf("  decode checksum:    0x%016llx  last_token=%d\n",
           (unsigned long long) token_checksum,
           tok);
    if (env_enabled("GEIST_BENCH_VALIDATE_REPLAY")) {
        const int vr = validate_replay_against_no_replay(
            geist_backend_name(be), model_path, prefill_n, decode_n,
            decoded_tokens, token_checksum);
        if (vr != GEIST_TEST_PASS) {
            free(decoded_tokens);
            goto fail;
        }
    }

    /* Sampler-state sanity check via stats. */
    struct geist_session_stats stats;
    if (geist_session_get_stats(sess, &stats) != GEIST_OK) {
        fprintf(stderr, "get_stats failed\n");
        free(decoded_tokens);
        goto fail;
    }
    if (stats.n_tokens_decoded != (uint64_t) decode_n) {
        fprintf(stderr,
                "stats.n_tokens_decoded = %llu, expected %zu\n",
                (unsigned long long) stats.n_tokens_decoded,
                decode_n);
        free(decoded_tokens);
        goto fail;
    }
    if (decode_n > 0 && stats.total_decode_ns == 0) {
        fprintf(stderr, "stats.total_decode_ns is zero after decode\n");
        free(decoded_tokens);
        goto fail;
    }
    if (dump_backend_profile) {
        s = backend_profile_dump(be);
        if (s != GEIST_OK) {
            fprintf(stderr, "backend profile dump failed\n");
            free(decoded_tokens);
            goto fail;
        }
        s = backend_profile_reset(be);
        if (s != GEIST_OK) {
            fprintf(stderr, "backend profile final reset failed\n");
            free(decoded_tokens);
            goto fail;
        }
    }
    free(decoded_tokens);

    geist_session_destroy(sess);
    geist_model_destroy(model);
    geist_backend_destroy(be);
    return GEIST_TEST_PASS;

fail:
    geist_session_destroy(sess);
    geist_model_destroy(model);
    geist_backend_destroy(be);
    return GEIST_TEST_FAIL;
}
