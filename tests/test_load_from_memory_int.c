/*
 * test_load_from_memory_int — geist_model_load_from_memory parity.
 *
 * mmaps the GGUF ourselves and loads it via geist_model_load_from_memory (the
 * path embedded builds take, minus the .incbin step), then asserts the greedy
 * token stream is bit-identical to geist_model_load(path). SKIPs without a GGUF.
 */
#include "test_helpers.h"

#include <geist.h>

#include <fcntl.h>
#include <stdio.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define N_TOK 16

/* Greedy-decode N tokens of a fixed prompt into `out`. Returns count, -1 on err. */
static int
generate(struct geist_model* model, struct geist_backend* be, geist_token_t* out, int n) {
    struct geist_session* s = nullptr;
    if (geist_session_create(model, be, nullptr, &s) != GEIST_OK) {
        return -1;
    }
    if (geist_session_set_prompt(s, "The capital of France is") != GEIST_OK) {
        geist_session_destroy(s);
        return -1;
    }
    int i = 0;
    for (; i < n; i++) {
        if (geist_session_decode_step(s, &out[i]) != GEIST_OK) {
            break;
        }
    }
    geist_session_destroy(s);
    return i;
}

int main(void) {
    GEIST_REQUIRE_GGUF(model_path);

    struct geist_backend* be = nullptr;
    if (geist_backend_create("auto", nullptr, nullptr, &be) != GEIST_OK) {
        fprintf(stderr, "backend create: %s\n", geist_last_create_error());
        return GEIST_TEST_ERROR;
    }

    /* mmap the GGUF as a plain buffer — stands in for an embedded .rodata blob. */
    int fd = open(model_path, O_RDONLY);
    if (fd < 0) {
        geist_backend_destroy(be);
        return GEIST_TEST_FAIL;
    }
    struct stat sb;
    if (fstat(fd, &sb) != 0) {
        close(fd);
        geist_backend_destroy(be);
        return GEIST_TEST_FAIL;
    }
    size_t size = (size_t) sb.st_size;
    void* buf = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (buf == MAP_FAILED) {
        geist_backend_destroy(be);
        return GEIST_TEST_FAIL;
    }

    int fail = 0;

    /* From-memory load + generate. */
    struct geist_model* m_mem = nullptr;
    geist_token_t mem_tok[N_TOK];
    int mem_n = -1;
    if (geist_model_load_from_memory(buf, size, be, &m_mem) != GEIST_OK) {
        fprintf(stderr, "FAIL: load_from_memory: %s\n", geist_last_create_error());
        fail = 1;
    } else {
        mem_n = generate(m_mem, be, mem_tok, N_TOK);
        geist_model_destroy(m_mem);
    }

    /* File load + generate (the reference). */
    struct geist_model* m_file = nullptr;
    geist_token_t file_tok[N_TOK];
    int file_n = -1;
    if (geist_model_load(model_path, be, &m_file) != GEIST_OK) {
        fprintf(stderr, "FAIL: load(path): %s\n", geist_last_create_error());
        fail = 1;
    } else {
        file_n = generate(m_file, be, file_tok, N_TOK);
        geist_model_destroy(m_file);
    }

    if (!fail) {
        if (mem_n <= 0 || mem_n != file_n) {
            fprintf(stderr, "FAIL: token count mem=%d file=%d\n", mem_n, file_n);
            fail = 1;
        } else {
            for (int i = 0; i < mem_n; i++) {
                if (mem_tok[i] != file_tok[i]) {
                    fprintf(stderr,
                            "FAIL: token %d differs mem=%d file=%d\n",
                            i,
                            mem_tok[i],
                            file_tok[i]);
                    fail = 1;
                }
            }
        }
        printf("from-memory vs file: %d tokens, %s\n", mem_n, fail ? "MISMATCH" : "bit-identical");
    }

    munmap(buf, size);
    geist_backend_destroy(be);
    return fail ? GEIST_TEST_FAIL : GEIST_TEST_PASS;
}
