#!/usr/bin/env python3
"""bench_quality_perf.py — reproducible perf/quality benchmark harness for geist.

Driven by the `make bench-small / bench-detailed / bench-quality-* /
bench-compare-ref` targets. The goal is *reproducibility*: every recorded
number is tagged with the host, OS, target, mode, thread count, and model so
results from different machines never silently overwrite each other.

Perf suites (`small`, `detailed`) are fully implemented: they run the C
`bench_session_throughput` binary against a GGUF and record prefill/decode
tok/s into benchmark/BENCHMARK.md, keeping the best run per (model, host, os, target,
mode, threads) key.

Quality suites (`quality-small`, `quality-detailed`) and `compare-ref` require
a reference toolchain (HF tokenizer + datasets, and/or a llama.cpp build) that
is out of scope for a hermetic `make` invocation. They print setup guidance and
exit cleanly rather than failing the build. See benchmark/BENCHMARKING.md.

Usage (normally invoked via the Makefile):
    python3 tools/bench_quality_perf.py --suite small \\
        --target mac-omp --mode release \\
        --bin-dir bin/mac-omp/release/tests --out-dir bench_runs/quality_perf \\
        --benchmark-md benchmark/BENCHMARK.md --record

Environment:
    BENCH_GGUF      Path to the model GGUF (falls back to GEIST_GGUF_PATH).
    BENCH_THREADS   OMP thread count (sets OMP_NUM_THREADS for the child).
    BENCH_REF_GGUF  Reference GGUF for compare-ref (quality suites).
    BENCH_REF_BIN   Reference binary (e.g. llama-bench) for compare-ref.
"""
from __future__ import annotations

import argparse
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PERF_SUITES = {"small", "detailed"}
QUALITY_SUITES = {"quality-small", "quality-detailed", "compare-ref"}

# tok/s lines from bench_session_throughput, e.g.
#   prefill (200 tok):    2560.6 ms  =  12.80 ms/tok  =    78.1 tok/s
TPS_RE = re.compile(r"^\s*(prefill|decode)\b.*?=\s*([\d.]+)\s*tok/s", re.MULTILINE)


def host_id() -> str:
    """Stable-ish host label so different machines don't clobber each other."""
    return f"{platform.node()}/{platform.machine()}"


def os_id() -> str:
    return f"{platform.system()} {platform.release()}"


def resolve_gguf() -> str | None:
    g = os.environ.get("BENCH_GGUF") or os.environ.get("GEIST_GGUF_PATH")
    if g and Path(g).is_file():
        return g
    return None


def run_throughput(bin_dir: Path, gguf: str, threads: str | None) -> dict[str, float]:
    """Run bench_session_throughput once; return {'prefill': tps, 'decode': tps}."""
    exe = bin_dir / "bench_session_throughput"
    if not exe.is_file():
        sys.exit(f"bench: missing {exe} — run `make bench` to build the bench binaries first")

    env = dict(os.environ)
    env["GEIST_GGUF_PATH"] = gguf
    env.setdefault("OMP_WAIT_POLICY", "active")
    if threads:
        env["OMP_NUM_THREADS"] = threads

    proc = subprocess.run([str(exe), gguf], env=env, capture_output=True, text=True)
    out = proc.stdout + proc.stderr
    if proc.returncode not in (0, None):
        sys.stderr.write(out)
        sys.exit(f"bench: bench_session_throughput exited {proc.returncode}")

    found = {m.group(1): float(m.group(2)) for m in TPS_RE.finditer(out)}
    if "prefill" not in found or "decode" not in found:
        sys.stderr.write(out)
        sys.exit("bench: could not parse tok/s from bench_session_throughput output")
    return found


def perf_suite(args: argparse.Namespace) -> None:
    gguf = resolve_gguf()
    if gguf is None:
        print("bench: no model found (set BENCH_GGUF or GEIST_GGUF_PATH, or run "
              "`make fetch-model`). Skipping perf suite.")
        return

    threads = os.environ.get("BENCH_THREADS") or None
    # `detailed` averages more runs to shrink variance; `small` is a quick check.
    n_runs = 5 if args.suite == "detailed" else 2
    best = {"prefill": 0.0, "decode": 0.0}
    for i in range(n_runs):
        r = run_throughput(Path(args.bin_dir), gguf, threads)
        print(f"  run {i + 1}/{n_runs}: prefill {r['prefill']:.1f} tok/s, "
              f"decode {r['decode']:.1f} tok/s")
        best["prefill"] = max(best["prefill"], r["prefill"])
        best["decode"] = max(best["decode"], r["decode"])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    row = {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "model": Path(gguf).name,
        "host": host_id(),
        "os": os_id(),
        "target": args.target,
        "mode": args.mode,
        "threads": threads or "default",
        "prefill": f"{best['prefill']:.1f}",
        "decode": f"{best['decode']:.1f}",
    }
    print(f"\nbest: prefill {row['prefill']} tok/s | decode {row['decode']} tok/s "
          f"({row['host']}, {args.target}/{args.mode}, threads={row['threads']})")

    if args.record and args.benchmark_md:
        update_benchmark_md(Path(args.benchmark_md), row)
        print(f"recorded to {args.benchmark_md}")


# Marker block in benchmark/BENCHMARK.md that this script owns. Hand-written prose above
# the marker is preserved; only the auto-recorded table below it is rewritten.
MARKER = "<!-- BENCH:AUTO -->"

# Column layout of the auto-recorded table — one source of truth for both the
# header and every positional access below (no magic indices elsewhere).
COLUMNS = ["Date", "Model", "Host", "OS", "Target/Mode", "Threads",
           "Prefill tok/s", "Decode tok/s"]
(COL_DATE, COL_MODEL, COL_HOST, COL_OS, COL_TARGET_MODE,
 COL_THREADS, COL_PREFILL, COL_DECODE) = range(len(COLUMNS))
# Columns identifying a unique config (date + tok/s excluded).
KEY_COLS = (COL_MODEL, COL_HOST, COL_OS, COL_TARGET_MODE, COL_THREADS)

TABLE_HEADER = (
    "| " + " | ".join(COLUMNS) + " |\n"
    "| " + " | ".join(":---" if i < COL_THREADS else ":---:"
                      for i in range(len(COLUMNS))) + " |"
)


def _row_key(cells: list[str]) -> tuple:
    return tuple(cells[i] for i in KEY_COLS)


def update_benchmark_md(path: Path, row: dict) -> None:
    """Insert/replace this run's row, keeping the best decode tok/s per key."""
    new_cells = [row["date"], row["model"], row["host"], row["os"],
                 f"{row['target']}/{row['mode']}", row["threads"],
                 row["prefill"], row["decode"]]

    existing: dict[tuple, list[str]] = {}
    preamble = ""
    if path.is_file():
        text = path.read_text()
        if MARKER in text:
            preamble = text.split(MARKER)[0]
            for line in text.split(MARKER)[1].splitlines():
                if line.strip().startswith("|") and "tok/s" not in line and ":---" not in line:
                    cells = [c.strip() for c in line.strip().strip("|").split("|")]
                    if len(cells) == len(COLUMNS):
                        existing[_row_key(cells)] = cells
        else:
            preamble = text.rstrip() + "\n\n"

    key = _row_key(new_cells)
    prev = existing.get(key)
    # Keep whichever run had the higher decode throughput for this key.
    if prev is None or float(new_cells[COL_DECODE]) >= float(prev[COL_DECODE]):
        existing[key] = new_cells

    if not preamble:
        preamble = ("# geist Benchmarks (auto-recorded)\n\n"
                    "Rows below are appended by `make bench-small` / `bench-detailed`. "
                    "Each (model, host, os, target/mode, threads) key keeps its best "
                    "decode run. See [BENCHMARKING.md](BENCHMARKING.md) for methodology.\n\n")

    rows = sorted(existing.values(),
                  key=lambda c: (c[COL_MODEL], c[COL_HOST], c[COL_TARGET_MODE]))
    body = TABLE_HEADER + "\n" + "\n".join("| " + " | ".join(c) + " |" for c in rows) + "\n"
    path.write_text(f"{preamble}{MARKER}\n\n{body}")


def quality_suite(args: argparse.Namespace) -> None:
    ref_gguf = os.environ.get("BENCH_REF_GGUF")
    ref_bin = os.environ.get("BENCH_REF_BIN")
    print(f"bench: suite '{args.suite}' needs a reference toolchain and is not "
          "wired into the hermetic make flow yet.")
    print("  Quality (PPL / KL / MMLU) requires the HF tokenizer + datasets;")
    print("  compare-ref additionally needs a llama.cpp build.")
    if args.suite == "compare-ref":
        print(f"  BENCH_REF_GGUF={ref_gguf or '(unset)'}  BENCH_REF_BIN={ref_bin or '(unset)'}")
    print("  See benchmark/BENCHMARKING.md for the manual procedure. Exiting cleanly.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", required=True,
                   choices=sorted(PERF_SUITES | QUALITY_SUITES))
    p.add_argument("--target", default="unknown")
    p.add_argument("--mode", default="release")
    p.add_argument("--bin-dir", default="bin")
    p.add_argument("--out-dir", default="bench_runs/quality_perf")
    p.add_argument("--benchmark-md", default="benchmark/BENCHMARK.md")
    p.add_argument("--record", action="store_true")
    args = p.parse_args()

    print(f"== geist bench: suite={args.suite} target={args.target} mode={args.mode} ==")
    if args.suite in PERF_SUITES:
        perf_suite(args)
    else:
        quality_suite(args)


if __name__ == "__main__":
    main()
