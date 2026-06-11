# Benchmarking geist

How to produce trustworthy numbers. See [BENCHMARK.md](../BENCHMARK.md) for
recorded results and [BENCHMARK_PI5.md](../BENCHMARK_PI5.md) for the Pi 5
target.

## Perf (reproducible, in-tree)

```sh
make fetch-model                         # Gemma 4 E2B-it Q4_K_M, ~3.1 GB
OMP_WAIT_POLICY=active make bench-small   # best-of-2, records to BENCHMARK.md
OMP_WAIT_POLICY=active make bench-detailed # best-of-5
BENCH_THREADS=6 OMP_WAIT_POLICY=active make bench-detailed  # pin thread count
```

These run `bench_session_throughput` (warm-up → prefill 200 → decode 50) via
`tools/bench_quality_perf.py`, which records a tagged row per
(model, host, os, target/mode, threads), keeping the best decode run.

Raw timing probes for individual subsystems:

```sh
make bench           # all bench_* binaries
make bench-vision    # vision encoder only
make bench-audio     # audio encoder only
```

## Comparison vs llama.cpp (manual)

A fair comparison pins the reference. Build `llama.cpp` (`llama-bench`) from a
known commit against the **same** GGUF, then:

```sh
BENCH_REF_BIN=/path/to/llama-bench \
BENCH_REF_GGUF=gguf_artifacts/gemma4-e2b-Q4_K_M.gguf \
make bench-compare-ref
```

Record, every time:

- the llama.cpp **commit hash** (`llama-bench --version` or `git rev-parse`);
- thread count and `OMP_WAIT_POLICY` / llama.cpp `-t`;
- host CPU + OS;
- that the GGUF is byte-identical for both engines.

Use bit-identical greedy output as the correctness gate before quoting any
speedup — a faster engine that produces different tokens is not iso-quality.

> The `compare-ref` and `quality-*` suites need a reference toolchain (a
> llama.cpp build and/or the HF tokenizer + datasets) that is intentionally
> kept out of the hermetic `make` flow. `tools/bench_quality_perf.py` prints
> setup guidance and exits cleanly when invoked for these suites.

## Quality (perplexity / KL / MMLU)

Quality eval drives the `eval_geist` REPL through `tools/eval_runner.py`, which
tokenizes with a Hugging Face tokenizer for parity with reference
implementations:

```sh
pip install transformers
python3 tools/eval_runner.py --bin bin/<target>/release/eval_geist \
    --gguf gguf_artifacts/gemma4-e2b-Q4_K_M.gguf \
    --tokenizer google/gemma-4-E2B-it \
    generate --prompt "The capital of France is" --n 16
```

`eval_runner.py mc` scores multiple-choice options by continuation logprob
(MMLU/HellaSwag style). For PPL/KL against the reference model, capture
per-token logits with `eval_geist`'s `SCORE` command over a held-out corpus and
compare distributions; this harness is documented here rather than wired into
`make` because it depends on external datasets.

### MMLU accuracy (self-contained, `make bench-mmlu`)

`tools/eval_mmlu.py` measures MMLU accuracy through the `eval_geist` REPL,
tokenizing with the model's **own** GGUF tokenizer (the `TOK` command) — so
there is no external HF-tokenizer dependency and no risk of a tokenizer mismatch
between scoring and the model. It uses the standard log-likelihood **cloze**:
build the 5-shot prompt, then score the next-token log-prob of " A"/" B"/" C"/
" D" (each a single token in the Gemma vocab) in one `SCORE`-style prefill and
take the argmax. This is a **base-completion** eval (no chat template), which is
how MMLU is conventionally run, so it sidesteps the chat-template parity
question entirely.

```sh
pip install datasets
make bench-mmlu                       # 200 shuffled questions, 5-shot
make bench-mmlu MMLU_LIMIT=0          # full ~14k-question set
# or directly, incl. a no-dataset smoke test:
python3 tools/eval_mmlu.py --bin bin/<target>/release/eval_geist \
    --gguf model.gguf --selftest --verbose     # embedded sample
python3 tools/eval_mmlu.py --bin ... --gguf model.gguf --hf --shuffle --limit 200
```

Few-shot matters: MMLU is conventionally 5-shot. Zero-shot, a small model
collapses to a position bias (always "A") — the harness reproduces this (0/5 on
the embedded sample at `--shots 0`, 5/5 at `--shots 5`), which is a property of
the model, not the scorer. This gives a real **absolute** MMLU number for geist;
a cross-*engine* MMLU/PPL ranking still needs matched conditions (below).

### Known limitation: cross-engine Gemma 4 E2B PPL ranking

`make bench-quality-*` is deliberately stubbed (prints guidance, exits 0).
A *quality ranking* (MMLU/PPL/KL) requires logprob-identical conditions on both
sides, and for Gemma 4 E2B neither side is clean yet:

- **geist side:** `bench_quality` generates correct text (greedy output is fine
  — "What is the capital of France?" → "The capital of France is Paris."), but
  it is an **eyeball tool, not a parity-grade scoring harness**. The Gemma 4
  turn markers it uses (`<|turn>` = id 105, `<turn|>` = id 106; the literal
  strings `<start_of_turn>`/`<end_of_turn>` are *not* single tokens for this
  model and tokenize to 7 pieces each) have not been verified byte-for-byte
  against the HF reference chat template, including BOS handling. For greedy
  generation the difference is invisible; for logprob scoring it shifts the
  distribution and invalidates a cross-engine comparison.
- **reference side:** llama.cpp's perplexity reports abnormally high absolute
  values on Gemma 4 E2B, so it is not currently a reliable PPL baseline for this
  model either.

Until both harnesses are verified for gemma4-e2b, geist publishes **speed**
(reproducible, above) and **iso-quality correctness** (bit-identical greedy
output vs llama.cpp on the *same* weights — a correctness claim, not a quality
ranking) but **not** an MMLU/PPL leaderboard number. Fixing this — a
template-parity harness plus a sane reference PPL path — is tracked as an open
task and is a welcome contribution.
