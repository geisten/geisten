/*
 * embedded_model.S — embeds a GGUF model into the binary's read-only data.
 *
 * Compiled and linked into the `geist` CLI only when the build sets
 *   make EMBED_MODEL=path/to/model.gguf
 * (which passes -DGEIST_EMBED_MODEL_PATH='"<abs path>"'). The bytes are exposed
 * as geist_embedded_model_start / _end and handed to
 * geist_model_load_from_memory, so the engine *and* the model ship as one file.
 *
 * .balign 32 matches the GGUF general.alignment default, so the data section
 * lands 32-byte aligned in memory and tensors alias zero-copy from .rodata.
 *
 * Two dialects: Mach-O (macOS) prefixes C symbols with '_' and names sections
 * differently from ELF; this is preprocessed (.S) so the right one is picked.
 */
#if defined(__APPLE__)
    .section __TEXT,__const
    .balign 32
    .globl _geist_embedded_model_start
_geist_embedded_model_start:
    .incbin GEIST_EMBED_MODEL_PATH
    .globl _geist_embedded_model_end
_geist_embedded_model_end:
#else
    .section .rodata
    .balign 32
    .globl geist_embedded_model_start
geist_embedded_model_start:
    .incbin GEIST_EMBED_MODEL_PATH
    .globl geist_embedded_model_end
geist_embedded_model_end:
#endif
