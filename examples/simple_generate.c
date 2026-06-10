/*
 * simple_generate.c — the smallest useful geist program.
 *
 * Loads a GGUF, prefills a text prompt, and greedy-decodes a continuation
 * to stdout. This exercises only the STABLE core of the public API
 * (include/geist.h): backend -> model -> session -> set_prompt ->
 * decode_step -> token_to_str.
 *
 * Build & run (from the repo root):
 *   make                       # build libgeist.a for the detected target
 *   make -C examples           # build this program against it
 *   OMP_WAIT_POLICY=active examples/simple_generate \
 *       gguf_artifacts/gemma4-e2b-Q4_K_M.gguf "The capital of France is"
 */
#include <geist.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> [prompt] [max_new_tokens]\n", argv[0]);
        return 2;
    }
    const char *model_path = argv[1];
    const char *prompt     = (argc > 2) ? argv[2] : "Hello, my name is";
    int         max_new    = (argc > 3) ? atoi(argv[3]) : 64;

    /* "auto" picks the best backend compiled into this build for the host. */
    struct geist_backend *be = nullptr;
    if (geist_backend_create("auto", nullptr, nullptr, &be) != GEIST_OK) {
        fprintf(stderr, "backend_create failed: %s\n", geist_last_create_error());
        return 1;
    }

    struct geist_model *model = nullptr;
    if (geist_model_load(model_path, be, &model) != GEIST_OK) {
        fprintf(stderr, "model_load failed: %s\n", geist_last_create_error());
        geist_backend_destroy(be);
        return 1;
    }
    fprintf(stderr, "loaded %s (arch: %s)\n", model_path, geist_model_arch(model));

    /* Zero-initialized opts == greedy decode (temperature 0). */
    struct geist_session_opts opts = {0};
    struct geist_session     *sess = nullptr;
    if (geist_session_create(model, be, &opts, &sess) != GEIST_OK) {
        fprintf(stderr, "session_create failed\n");
        geist_model_destroy(model);
        geist_backend_destroy(be);
        return 1;
    }

    if (geist_session_set_prompt(sess, prompt) != GEIST_OK) {
        fprintf(stderr, "set_prompt failed: %s\n", geist_session_errmsg(sess));
        return 1;
    }

    printf("%s", prompt);
    fflush(stdout);

    for (int i = 0; i < max_new; i++) {
        geist_token_t tok = 0;
        if (geist_session_decode_step(sess, &tok) != GEIST_OK) {
            fprintf(stderr, "\ndecode_step failed: %s\n", geist_session_errmsg(sess));
            break;
        }
        /* Tokens with no surface form (true control tokens) -> stop. Gemma
         * also emits bracketed specials like "<eos>" / "<end_of_turn>" that
         * DO carry a surface form; a real app would track the model's EOS id,
         * but for a self-contained demo we just stop at the first such token. */
        const char *piece = geist_session_token_to_str(sess, tok);
        if (piece == nullptr) {
            break;
        }
        size_t len = strlen(piece);
        if (len >= 2 && piece[0] == '<' && piece[len - 1] == '>') {
            break;
        }
        fputs(piece, stdout);
        fflush(stdout);
    }
    putchar('\n');

    geist_session_destroy(sess);
    geist_model_destroy(model);
    geist_backend_destroy(be);
    return 0;
}
