#!/usr/bin/env python3
# Total tok/s (prefill+decode end-to-end) for the three engines installed on
# the Pi, same Q4_K GGUF, 4 threads, cold-started each. total = (P+D)/(P/pp + D/tg).
# Paths are env-overridable so this survives a Pi re-clone / runs on other boxes.
import json, re, subprocess, time, urllib.request, os
HOME = os.path.expanduser("~")
M = os.environ.get("MODEL", HOME + "/gguf_artifacts/gemma4-e2b-Q4_K_M.gguf")
GEIST = os.environ.get("GEIST_BIN", HOME + "/geist/bin/pi5/release/tests/bench_perf_sweep")
LLAMA_CPU = os.environ.get("LLAMA_CPU", HOME + "/llama.cpp/build-cpu/bin/llama-bench")
LLAMA_BLAS = os.environ.get("LLAMA_BLAS", HOME + "/llama.cpp/build/bin/llama-bench")
P = int(os.environ.get("P", "128"))
D = int(os.environ.get("D", "128"))

def temp():
    try: return float(subprocess.check_output(["vcgencmd","measure_temp"]).decode().split("=")[1].split("'")[0])
    except: return -1

def cool(thr=56):
    while temp() >= thr: time.sleep(10)
    return temp()

def sh(cmd, env=None):
    e = dict(os.environ); e.update(env or {})
    return subprocess.run(cmd, capture_output=True, text=True, env=e).stdout

def geist_total():
    t0 = cool()
    out = sh([GEIST, "--gguf", M, "--seq-lens", str(P), "--decode-n", str(D),
              "--warmup", "16", "--repeats", "5"],
             {"GEIST_WEIGHT_MMAP":"1","OMP_WAIT_POLICY":"active"})
    for ln in out.splitlines():
        if ln.strip().startswith("{"):
            j = json.loads(ln); return j["prefill_tps"], j["decode_tps"], j["total_tps"], t0
    return None

def llama_total(bench, label):
    t0 = cool()
    out = sh([bench, "-m", M, "-p", str(P), "-n", str(D), "-t", "4", "-r", "3"])
    pp = tg = None
    for ln in out.splitlines():
        m = re.search(r"\bpp%d\b.*?\|\s*([\d.]+)" % P, ln)
        if m: pp = float(m.group(1))
        m = re.search(r"\btg%d\b.*?\|\s*([\d.]+)" % D, ln)
        if m: tg = float(m.group(1))
    if pp and tg:
        total = (P+D)/(P/pp + D/tg)
        return pp, tg, total, t0
    return None

def ollama_total():
    t0 = cool()
    UNIT = "The quick brown fox jumps over the lazy dog. "
    prompt = "REQxy42. " + UNIT*13  # ~128 tokens, unique prefix
    body = json.dumps({"model":"gemma4-e2b-q4k","prompt":prompt,"stream":False,
        "options":{"num_thread":4,"num_ctx":512,"num_batch":128,"use_mmap":True,
                   "num_predict":D,"temperature":0}}).encode()
    req = urllib.request.Request("http://127.0.0.1:11434/api/generate", data=body,
                                 headers={"Content-Type":"application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=600))
    pp = r["prompt_eval_count"]/(r["prompt_eval_duration"]/1e9)
    tg = r["eval_count"]/(r["eval_duration"]/1e9)
    pt, dt = r["prompt_eval_count"], r["eval_count"]
    total = (pt+dt)/((r["prompt_eval_duration"]+r["eval_duration"])/1e9)
    return pp, tg, total, t0

print("workload: P=%d prompt + D=%d decode, 4 threads, cold start each" % (P, D), flush=True)
print("%-22s %8s %8s %10s  cold" % ("engine","pp_t/s","tg_t/s","TOTAL t/s"), flush=True)
for name, fn in [("geist", geist_total),
                 ("llama.cpp (CPU)", lambda: llama_total(LLAMA_CPU, "cpu")),
                 ("llama.cpp (OpenBLAS)", lambda: llama_total(LLAMA_BLAS, "blas")),
                 ("ollama", ollama_total)]:
    try:
        pp, tg, total, t0 = fn()
        print("%-22s %8.2f %8.3f %10.3f  %.0fC" % (name, pp, tg, total, t0), flush=True)
    except Exception as e:
        print("%-22s  ERROR: %s" % (name, e), flush=True)
