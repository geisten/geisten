#!/usr/bin/env python3
"""eval_runner.py — drive the `eval_geist` REPL for evaluation/scoring.

`eval_geist` loads a GGUF once and answers line-oriented commands on stdin
(DECODE / SCORE / SCOREALT / RESET / QUIT — see the header of eval_geist.c).
This wrapper speaks that protocol so you can:

  * generate  — greedy-decode a continuation from a text prompt;
  * mc        — multiple-choice: score each option's continuation logprob and
                pick the argmax (the MMLU/HellaSwag-style harness).

Tokenization uses a Hugging Face tokenizer for parity with reference
implementations (`pip install transformers`). For protocol smoke-testing
without a tokenizer, pass raw integer token IDs with --raw-ids.

Examples:
  # Greedy generate 32 tokens (HF tokenizer for the model's vocab):
  python3 tools/eval_runner.py --bin bin/mac-omp/release/tools/eval_geist \\
      --gguf model.gguf --tokenizer google/gemma-4-E2B-it \\
      generate --prompt "The capital of France is" --n 32

  # Protocol smoke test, no tokenizer:
  python3 tools/eval_runner.py --bin bin/.../tools/eval_geist --gguf model.gguf \\
      generate --raw-ids "2 651 1234" --n 8
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


class Repl:
    """Thin wrapper around a persistent eval_geist process."""

    def __init__(self, binary: str, gguf: str, awq: str | None = None):
        cmd = [binary, gguf] + (["--awq", awq] if awq else [])
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1)
        # eval_geist emits a "READY" banner on stdout once the model is loaded.
        assert self.proc.stdout
        while True:
            line = self.proc.stdout.readline()
            if not line:
                err = self.proc.stderr.read() if self.proc.stderr else ""
                sys.exit(f"eval_geist exited before READY.\n{err}")
            if line.strip() == "READY":
                break

    def cmd(self, line: str) -> str:
        assert self.proc.stdin and self.proc.stdout
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()
        out = self.proc.stdout.readline().strip()
        if not out:
            err = self.proc.stderr.read() if self.proc.stderr else ""
            sys.exit(f"eval_geist produced no response to {line!r}.\n{err}")
        return out

    def decode(self, ids: list[int], n_decode: int) -> list[int]:
        resp = self.cmd(f"DECODE {len(ids)} {' '.join(map(str, ids))} {n_decode}")
        parts = resp.split()
        if parts[0] != "OK":
            sys.exit(f"DECODE failed: {resp}")
        return [int(x) for x in parts[2:]]

    def score(self, prompt: list[int], cont: list[int]) -> float:
        resp = self.cmd(f"SCORE {len(prompt)} {' '.join(map(str, prompt))} "
                        f"{len(cont)} {' '.join(map(str, cont))}")
        parts = resp.split()
        if parts[0] != "OK":
            sys.exit(f"SCORE failed: {resp}")
        return float(parts[1])  # total continuation logprob

    def reset(self) -> None:
        self.cmd("RESET")

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                self.cmd("QUIT")
            except SystemExit:
                pass
        self.proc.terminate()


def load_tokenizer(name: str):
    try:
        from transformers import AutoTokenizer
    except ImportError:
        sys.exit("eval_runner: --tokenizer requires `pip install transformers`. "
                 "Use --raw-ids to smoke-test the protocol without it.")
    return AutoTokenizer.from_pretrained(name)


def parse_ids(args, tok, text: str) -> list[int]:
    if args.raw_ids is not None:
        return [int(x) for x in args.raw_ids.split()]
    if tok is None:
        sys.exit("provide --tokenizer or --raw-ids")
    return tok.encode(text, add_special_tokens=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bin", required=True, help="path to eval_geist binary")
    p.add_argument("--gguf", required=True)
    p.add_argument("--awq", default=None)
    p.add_argument("--tokenizer", default=None, help="HF tokenizer name/path")
    sub = p.add_subparsers(dest="mode", required=True)

    g = sub.add_parser("generate", help="greedy-decode a continuation")
    g.add_argument("--prompt", default="")
    g.add_argument("--raw-ids", default=None, help="space-separated token IDs")
    g.add_argument("--n", type=int, default=32)

    m = sub.add_parser("mc", help="multiple-choice: argmax option by logprob")
    m.add_argument("--prompt", required=True)
    m.add_argument("--options", required=True, nargs="+",
                   help="candidate continuations (text)")

    args = p.parse_args()
    if not Path(args.bin).is_file():
        sys.exit(f"eval_runner: binary not found: {args.bin} (run `make` first)")

    tok = load_tokenizer(args.tokenizer) if args.tokenizer else None
    repl = Repl(args.bin, args.gguf, args.awq)
    try:
        if args.mode == "generate":
            ids = parse_ids(args, tok, args.prompt)
            out = repl.decode(ids, args.n)
            text = tok.decode(out) if tok else None
            print(f"prompt_ids={ids}")
            print(f"output_ids={out}")
            if text is not None:
                print(f"output_text={text!r}")
        else:  # mc
            prompt_ids = tok.encode(args.prompt, add_special_tokens=True)
            best_i, best_lp = -1, float("-inf")
            for i, opt in enumerate(args.options):
                cont = tok.encode(opt, add_special_tokens=False)
                repl.reset()
                lp = repl.score(prompt_ids, cont)
                print(f"  [{i}] logprob={lp:.3f}  {opt!r}")
                if lp > best_lp:
                    best_i, best_lp = i, lp
            print(f"answer={best_i}  {args.options[best_i]!r}")
    finally:
        repl.close()


if __name__ == "__main__":
    main()
