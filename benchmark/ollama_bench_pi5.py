#!/usr/bin/env python3
"""benchmark/ollama_bench_pi5.py — ollama prefill+decode on a Raspberry Pi 5.

Measures ollama (its bundled llama.cpp runner) on the SAME GGUF geist and
upstream llama.cpp are benched against, so the three engines are comparable on
one board. Run ON the Pi 5 (not cross-invoked); ollama must be installed and
its service running. Reports the MEAN of N repeats after a discarded warm-up.

Two non-obvious things this script gets right — both learned the hard way on
the 4 GB board:

  1. use_mmap=True is REQUIRED. ollama defaults to loading weights resident; on
     gemma4-e2b the ~1.93 GB PLE table then sits in anonymous RAM and the
     llama-server child is OOM-killed at load ("signal: killed"). Forcing
     use_mmap keeps the weights file-backed so the model fits in 4 GB — the same
     constraint geist solves natively via GEIST_WEIGHT_MMAP. num_batch is kept
     small for the same memory reason (a large prefill batch buffer won't fit).

  2. Each request uses a UNIQUE leading prompt token. ollama caches the KV of a
     shared prompt prefix across requests, so reusing an overlapping
     prompt collapses prompt_eval_duration to ~0 and reports absurd prefill
     rates (e.g. 1700 t/s). A unique prefix per call forces a real prefill.

Thermal: the passively-cooled Pi 5 throttles. The board MUST start < ~56 °C and
is logged per length so thermal drift is visible — see benchmark/BENCHMARK_PI5.md.

Setup (one-time):
    curl -fsSL https://ollama.com/install.sh | sh
    printf 'FROM %s\n' ~/gguf_artifacts/gemma4-e2b-Q4_K_M.gguf > Modelfile.gemma4q4k
    ollama create gemma4-e2b-q4k -f Modelfile.gemma4q4k

Run:
    python3 benchmark/ollama_bench_pi5.py            # defaults below
    MODEL=gemma4-e2b-q4k SEQ_LENS=128,256,512 REPEATS=3 python3 benchmark/ollama_bench_pi5.py
"""
import json, os, subprocess, sys, time, urllib.request

MODEL = os.environ.get("MODEL", "gemma4-e2b-q4k")
API = os.environ.get("OLLAMA_API", "http://127.0.0.1:11434/api/generate")
SEQ_LENS = [int(x) for x in os.environ.get("SEQ_LENS", "128,256,512").split(",")]
REPEATS = int(os.environ.get("REPEATS", "3"))
DECODE_N = int(os.environ.get("DECODE_N", "32"))
NUM_THREAD = int(os.environ.get("NUM_THREAD", "4"))
NUM_BATCH = int(os.environ.get("NUM_BATCH", "128"))
MAX_TEMP_C = float(os.environ.get("MAX_TEMP_C", "56"))
# ~10 tokens/repeat; num_ctx must cover the longest prompt + DECODE_N.
UNIT = "The quick brown fox jumps over the lazy dog. "
NUM_CTX = max(SEQ_LENS) + DECODE_N + 256
_call = 0


def temp_c():
    try:
        out = subprocess.check_output(["vcgencmd", "measure_temp"]).decode()
        return float(out.strip().split("=")[1].rstrip("'C"))
    except Exception:
        return float("nan")


def gen(words, npred):
    global _call
    _call += 1
    # Unique leading token defeats ollama's prompt-prefix KV cache (see header).
    prompt = "REQ%05d zz%05d. %s" % (_call, _call * 7919 % 100000, UNIT * words)
    body = json.dumps({
        "model": MODEL, "prompt": prompt, "stream": False,
        "options": {"num_thread": NUM_THREAD, "num_ctx": NUM_CTX, "num_batch": NUM_BATCH,
                    "use_mmap": True, "num_predict": npred, "temperature": 0},
    }).encode()
    req = urllib.request.Request(API, data=body, headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))


def main():
    t0 = temp_c()
    if t0 == t0 and t0 > MAX_TEMP_C:  # not NaN and too hot
        sys.exit("REFUSING: board at %.1f °C > %.1f °C — let it cool first." % (t0, MAX_TEMP_C))
    print("[ollama] model=%s threads=%d num_batch=%d num_ctx=%d use_mmap=1 greedy | warmup %.1f°C"
          % (MODEL, NUM_THREAD, NUM_BATCH, NUM_CTX, t0), flush=True)
    gen(2, 8)  # discarded warm-up
    print("seq_target  ptoks  prefill_tps  dtoks  decode_tps  temp_C", flush=True)
    for tgt in SEQ_LENS:
        words = max(1, round(tgt / 10))
        pps, dps, pt, dt = [], [], 0, 0
        for _ in range(REPEATS):
            r = gen(words, DECODE_N)
            pt, dt = r["prompt_eval_count"], r.get("eval_count", 0)
            pps.append(pt / (r["prompt_eval_duration"] / 1e9))
            dps.append(dt / (r.get("eval_duration", 1) / 1e9))
        print("%9d  %5d  %10.2f  %5d  %9.3f  %.1f"
              % (tgt, pt, sum(pps) / len(pps), dt, sum(dps) / len(dps), temp_c()), flush=True)
    print("[done] %.1f°C" % temp_c(), flush=True)


if __name__ == "__main__":
    main()
