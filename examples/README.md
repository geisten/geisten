# geist examples

Minimal, self-contained programs against the **STABLE** core of the public
API (`include/geist.h`).

## `simple_generate`

Loads a GGUF, prefills a prompt, and greedy-decodes a continuation.

```sh
# From the repo root:
make                 # build libgeist.a for the detected target
make -C examples     # build the examples against it

OMP_WAIT_POLICY=active examples/simple_generate \
    gguf_artifacts/gemma4-e2b-Q4_K_M.gguf "The capital of France is"
# -> The capital of France is Paris.
```

Arguments: `simple_generate <model.gguf> [prompt] [max_new_tokens]`.

The example builds with the exact same target settings as the library
(`make -C examples` includes `mk/target-$(TARGET).mk`), so it works on every
supported target — pass `TARGET=pi5` / `MODE=debug` to override.

It uses only `geist_backend_create` → `geist_model_load` →
`geist_session_create` → `geist_session_set_prompt` →
`geist_session_decode_step` → `geist_session_token_to_str`. That is the whole
stable surface needed to run text generation; everything multimodal
(`attach_audio` / `attach_image` / `attach_video`) and the
speculative / KV-mode knobs are `EXPERIMENTAL` extensions on top.
