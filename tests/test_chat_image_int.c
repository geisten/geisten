/*
 * test_chat_image_int — coherent image→text chat via Gemma 4 chat template.
 *
 * Wraps the image in Gemma 4's <|turn>user…<turn|>\n<|turn>model envelope:
 *
 *   <bos><|turn>user\n<|image>      ← prefix (tokenize → prefill)
 *   [280 vision soft tokens]        ← attach_image
 *   <image|>\n{question}<turn|>\n<|turn>model\n
 *                                   ← suffix (tokenize → prefill)
 *   [decode loop until <turn|> or <eos>]
 *
 * Special tokens (verified from gemma-4-E2B-it/tokenizer_config.json):
 *   BOS  = <bos>           (id  2)
 *   EOS  = <eos>           (id  1)
 *   SOT  = <|turn>         (id looked up at runtime)
 *   EOT  = <turn|>         (id looked up at runtime)
 *   BOI  = <|image>        (id 255999 — note asymmetric pipe)
 *   EOI  = <image|>        (id 258882)
 *
 * Skip cleanly if model, tokenizer.bin, or image is missing.
 */
#include "test_helpers.h"

#include "stb_image.h"

#include <geist.h>
#include <geist_util.h>
#include <geist_backend.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PROMPT_CAP 1024
#define DECODE_CAP 128

/* Tokenize via session, optionally dropping a leading BOS so we can
 * append a fragment without it auto-prefixing each call. */
static enum geist_status tokenize_fragment(struct geist_session *s,
                                           const char           *text,
                                           bool                  drop_leading_bos,
                                           geist_token_t        *out_ids,
                                           size_t                out_cap,
                                           size_t               *out_n) {
    geist_token_t     scratch[PROMPT_CAP];
    size_t            n  = 0;
    enum geist_status rc = geist_session_tokenize(s, text, PROMPT_CAP, scratch, &n);
    if (rc != GEIST_OK)
        return rc;

    size_t start = 0;
    if (drop_leading_bos && n > 0 && scratch[0] == 2 /* <bos> */)
        start = 1;
    size_t copy = n - start;
    if (copy > out_cap)
        return GEIST_E_INVALID_ARG;
    for (size_t i = 0; i < copy; i++)
        out_ids[i] = scratch[start + i];
    *out_n = copy;
    return GEIST_OK;
}

/* Resolve EOT (<turn|>) at runtime by tokenizing a tiny known string. */
static geist_token_t find_eot_token(struct geist_session *s) {
    geist_token_t     ids[8];
    size_t            n  = 0;
    enum geist_status rc = geist_session_tokenize(s, "<turn|>", 8, ids, &n);
    if (rc != GEIST_OK || n == 0)
        return -1;
    /* SentencePiece may prepend BOS; the EOT should be the last ID we got. */
    return ids[n - 1];
}

int main(int argc, char **argv) {
    GEIST_REQUIRE_GGUF(model_path);

    const char *img_path   = "vision_bench/syn_320x224.png";
    const char *question   = "What do you see in this image?";
    size_t      max_tokens = 80;
    bool        no_image   = false;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--image") == 0 && i + 1 < argc)
            img_path = argv[++i];
        else if (strcmp(argv[i], "--question") == 0 && i + 1 < argc)
            question = argv[++i];
        else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc)
            max_tokens = (size_t) atoi(argv[++i]);
        else if (strcmp(argv[i], "--no-image") == 0)
            no_image = true;
    }

    int      w = 0, h = 0, c = 0;
    uint8_t *rgb = stbi_load(img_path, &w, &h, &c, 3);
    if (rgb == nullptr) {
        GEIST_SKIP(
                "could not decode image (use --image PATH; default vision_bench/syn_320x224.png)");
    }
    printf("image: %s (%dx%d RGB)\n", img_path, h, w);
    printf("question: %s\n", question);

    struct geist_backend *be = nullptr;
    enum geist_status     s  = geist_backend_create("cpu_neon", nullptr, nullptr, &be);
    if (s != GEIST_OK)
        s = geist_backend_create("cpu_scalar", nullptr, nullptr, &be);
    if (s != GEIST_OK) {
        fprintf(stderr, "backend create failed: %s\n", geist_last_create_error());
        stbi_image_free(rgb);
        return GEIST_TEST_ERROR;
    }

    struct geist_model *model = nullptr;
    s                         = geist_model_load(model_path, be, &model);
    if (s != GEIST_OK) {
        fprintf(stderr, "model_load failed: %s\n", geist_last_create_error());
        geist_backend_destroy(be);
        stbi_image_free(rgb);
        return GEIST_TEST_FAIL;
    }

    struct geist_session_opts opts = {.max_seq_len = 2048};
    struct geist_session     *sess = nullptr;
    s                              = geist_session_create(model, be, &opts, &sess);
    if (s != GEIST_OK) {
        fprintf(stderr, "session_create failed\n");
        geist_model_destroy(model);
        geist_backend_destroy(be);
        stbi_image_free(rgb);
        return GEIST_TEST_FAIL;
    }

    /* Probe for the tokenizer up-front so we SKIP cleanly rather than
     * mid-stream. Tokenize a trivial string and see what comes back. */
    {
        geist_token_t probe[4];
        size_t        pn = 0;
        s                = geist_session_tokenize(sess, "x", 4, probe, &pn);
        if (s == GEIST_E_NOT_FOUND) {
            printf("SKIP: no tokenizer loaded — set "
                   "GEIST_TOKENIZER_PATH=gemma-4-E2B-it/tokenizer.bin or "
                   "run from a working dir where the heuristic resolves\n");
            geist_session_destroy(sess);
            geist_model_destroy(model);
            geist_backend_destroy(be);
            stbi_image_free(rgb);
            return GEIST_TEST_SKIP;
        }
    }

    const geist_token_t eot = find_eot_token(sess);
    printf("EOT token id: %d\n", eot);

    /* --- Prefix: <bos><|turn>user\n[<|image>] ----------------------- */
    geist_token_t prefix[PROMPT_CAP];
    size_t        prefix_n    = 0;
    const char   *prefix_text = no_image ? "<bos><|turn>user\n" : "<bos><|turn>user\n<|image>";
    s                         = tokenize_fragment(sess,
                                                  prefix_text,
                                                  /*drop_bos=*/false,
                                                  prefix,
                                                  PROMPT_CAP,
                                                  &prefix_n);
    if (s != GEIST_OK) {
        fprintf(stderr, "tokenize prefix failed: %s\n", geist_status_to_string(s));
        goto teardown_fail;
    }
    printf("prefix tokens (%zu):", prefix_n);
    for (size_t i = 0; i < prefix_n; i++)
        printf(" %d", prefix[i]);
    printf("\n");

    s = geist_session_prefill_tokens(sess, prefix_n, prefix);
    if (s != GEIST_OK) {
        fprintf(stderr, "prefill prefix failed: %s\n", geist_status_to_string(s));
        goto teardown_fail;
    }

    /* --- Image: 280 vision soft tokens via attach_image -------------- */
    if (!no_image) {
        s = geist_session_attach_image(sess, (size_t) h, (size_t) w, rgb);
        if (s == GEIST_E_NOT_FOUND) {
            printf("SKIP: vision_tower.safetensors not found — %s\n", geist_session_errmsg(sess));
            stbi_image_free(rgb);
            geist_session_destroy(sess);
            geist_model_destroy(model);
            geist_backend_destroy(be);
            return GEIST_TEST_SKIP;
        }
        if (s != GEIST_OK) {
            fprintf(stderr,
                    "attach_image failed: %s — %s\n",
                    geist_status_to_string(s),
                    geist_session_errmsg(sess));
            goto teardown_fail;
        }
        printf("image attached: 280 soft tokens injected\n");
    } else {
        printf("--no-image: text-only flow for triage\n");
    }
    stbi_image_free(rgb);
    rgb = nullptr;

    /* --- Suffix: [<image|>\n]{question}<turn|>\n<|turn>model\n ------ */
    char suffix_text[PROMPT_CAP];
    int  sn =
            snprintf(suffix_text,
                     sizeof suffix_text,
                     no_image ? "%s<turn|>\n<|turn>model\n" : "<image|>\n%s<turn|>\n<|turn>model\n",
                     question);
    if (sn < 0 || sn >= (int) sizeof suffix_text) {
        fprintf(stderr, "suffix too long\n");
        goto teardown_fail;
    }
    geist_token_t suffix[PROMPT_CAP];
    size_t        suffix_n = 0;
    s                      = tokenize_fragment(sess,
                                               suffix_text,
                                               /*drop_bos=*/true,
                                               suffix,
                                               PROMPT_CAP,
                                               &suffix_n);
    if (s != GEIST_OK) {
        fprintf(stderr, "tokenize suffix failed: %s\n", geist_status_to_string(s));
        goto teardown_fail;
    }
    printf("suffix tokens (%zu):", suffix_n);
    for (size_t i = 0; i < suffix_n; i++)
        printf(" %d", suffix[i]);
    printf("\n");

    s = geist_session_prefill_tokens(sess, suffix_n, suffix);
    if (s != GEIST_OK) {
        fprintf(stderr, "prefill suffix failed: %s\n", geist_status_to_string(s));
        goto teardown_fail;
    }

    /* --- Decode loop ------------------------------------------------- */
    printf("\nmodel: ");
    fflush(stdout);
    size_t generated = 0;
    for (; generated < max_tokens; generated++) {
        geist_token_t tok;
        s = geist_session_decode_step(sess, &tok);
        if (s != GEIST_OK) {
            fprintf(stderr, "\ndecode_step failed: %s\n", geist_status_to_string(s));
            goto teardown_fail;
        }
        if (tok == 1 /* <eos> */ || (eot >= 0 && tok == eot)) {
            break;
        }
        const char *t = geist_session_token_to_str(sess, tok);
        if (t != nullptr) {
            fputs(t, stdout);
            fflush(stdout);
        }
    }
    printf("\n(%zu tokens generated)\n", generated);

    geist_session_destroy(sess);
    geist_model_destroy(model);
    geist_backend_destroy(be);
    return GEIST_TEST_PASS;

teardown_fail:
    if (rgb)
        stbi_image_free(rgb);
    geist_session_destroy(sess);
    geist_model_destroy(model);
    geist_backend_destroy(be);
    return GEIST_TEST_FAIL;
}
