"""Kaggle entry point: install, generate on both GPUs in parallel, then judge everything."""

import json
import os
import subprocess
import sys
import time
import traceback

PIP = [
    "safetune @ git+https://github.com/Lexsi-Labs/SafeTune.git",
    "circuitkit[quantization] @ git+https://github.com/Lexsi-Labs/CircuitKIT.git",
    "auditkit[transformers] @ git+https://github.com/Lexsi-Labs/AuditKIT.git",
    "optimum-quanto>=0.2", "llmcompressor", "datasets", "accelerate", "pandas", "pyarrow",
]
LLAMA_CPP = ["llama-cpp-python==0.3.36", "--only-binary", "llama-cpp-python",
             "--extra-index-url", "https://abetlen.github.io/llama-cpp-python/whl/cu125"]
KAGGLE_LIMIT_H = 12.0


def sh(args):
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args],
                       capture_output=True, text=True)
    print("pip", " ".join(a for a in args if not a.startswith("http"))[:200], "->", r.returncode)
    if r.returncode:
        print(r.stderr[-2500:])
    return r.returncode == 0


def setup(need_gguf):
    ok = sh(PIP + ["bitsandbytes"])
    gguf_ok = None
    if need_gguf:
        # The llama.cpp CUDA wheel links against CUDA 12; Kaggle ships CUDA 13, so add the
        # CUDA 12 runtime/cuBLAS wheels and point the workers' LD_LIBRARY_PATH at them.
        gguf_ok = sh(LLAMA_CPP) and sh(["nvidia-cuda-runtime-cu12", "nvidia-cublas-cu12"])
    return {"pip_ok": ok, "llama_cpp_ok": gguf_ok}


def cuda12_lib_dirs():
    import glob
    import site

    dirs = []
    for sp in site.getsitepackages():
        dirs += glob.glob(os.path.join(sp, "nvidia", "*", "lib"))
    return dirs


def _jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def _write_jsonl(path, rows):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def judge_all(out, deadline=None):
    """Label generations. Each judge writes per-configuration files as it goes, so a
    timeout keeps everything finished so far; judgments.jsonl is the merged result."""
    from .judges import HarmBenchJudge, RefusalJudge, keyword_refusal

    names = sorted(n for n in os.listdir(out)
                   if os.path.exists(os.path.join(out, n, "generations.jsonl")))
    rows = {n: _jsonl(os.path.join(out, n, "generations.jsonl")) for n in names}

    for fname, make, harmful_only in [("refusal.jsonl", RefusalJudge, False),
                                      ("harmbench.jsonl", HarmBenchJudge, True)]:
        todo = [n for n in names if not os.path.exists(os.path.join(out, n, fname))]
        print(f"{fname}: to judge {todo}", flush=True)
        if not todo:
            continue
        judge = make()
        for n in todo:
            if deadline and time.time() > deadline:
                print("judge deadline reached", flush=True)
                break
            sub = [r for r in rows[n] if r["expected"] == "refuse"] if harmful_only else rows[n]
            t = time.time()
            labels, raw = judge([r["prompt"] for r in sub], [r["response"] for r in sub])
            _write_jsonl(os.path.join(out, n, fname),
                         [{"id": r["id"], "label": l, "raw": a} for r, l, a in zip(sub, labels, raw)])
            print(f"{fname} {n}: {time.time() - t:.0f}s", flush=True)
        del judge
        _free()

    for n in names:
        rp, hp = os.path.join(out, n, "refusal.jsonl"), os.path.join(out, n, "harmbench.jsonl")
        if not os.path.exists(rp):
            continue
        ref = {r["id"]: r for r in _jsonl(rp)}
        harm = {r["id"]: r for r in _jsonl(hp)} if os.path.exists(hp) else {}
        merged = []
        for r in rows[n]:
            h = harm.get(r["id"])
            merged.append({
                "id": r["id"], "source": r["source"], "expected": r["expected"],
                "refusal_label": ref[r["id"]]["label"], "refusal_raw": ref[r["id"]]["raw"],
                "harmful": h["label"] if h else None, "harmful_raw": h["raw"] if h else None,
                "keyword_refusal": keyword_refusal(r["response"]),
            })
        _write_jsonl(os.path.join(out, n, "judgments.jsonl"), merged)


def resume_previous(out, base="/kaggle/input"):
    """Copy finished configurations (and the layer selection) from an earlier run attached
    as a kernel source, so they are not recomputed."""
    import glob
    import shutil

    copied = []
    for meta in glob.glob(os.path.join(base, "**", "out", "*", "meta.json"), recursive=True):
        src = os.path.dirname(meta)
        name = os.path.basename(src)
        done = name == "select" or os.path.exists(os.path.join(src, "judgments.jsonl"))
        dst = os.path.join(out, name)
        if done and not os.path.exists(dst):
            shutil.copytree(src, dst)
            copied.append(name)
    for lj in glob.glob(os.path.join(base, "**", "out", "layers.json"), recursive=True):
        if not os.path.exists(os.path.join(out, "layers.json")):
            shutil.copy(lj, os.path.join(out, "layers.json"))
    print("resumed from previous run:", copied, flush=True)
    return copied


def _free():
    import gc

    import torch
    gc.collect()
    torch.cuda.empty_cache()


def main(run):
    """run: dict with configs, out, code_dir and options (limit, arc_limit, judge, gen_hours)."""
    t0 = time.time()
    status = {"started": time.strftime("%Y-%m-%d %H:%M:%S")}
    status.update(setup(any(c["kind"] == "gguf" for c in run["configs"])))

    sys.path.insert(0, run["code_dir"])
    from .data import find_input_dir, load_arc, load_sets
    from .models import find_model_dir

    # find_input_dir returns the folder holding the marker file (the OR-Bench folder);
    # its parent is the dataset root that also holds the walledai__* folders.
    prompts_dir = os.path.dirname(find_input_dir("or-bench-hard-1k.csv"))
    prompts, select_harmful = load_sets(prompts_dir)
    out = run["out"]
    os.makedirs(out, exist_ok=True)
    status["resumed"] = resume_previous(out) if run.get("resume") else []
    configs = sorted(run["configs"], key=lambda c: c["kind"] != "select")  # selection first
    gen_hours = float(run.get("gen_hours", 8.5))
    full = {**run, "configs": configs, "prompts": prompts, "select_harmful": select_harmful,
            "arc": load_arc(), "model_dir": find_model_dir(),
            "deadline_epoch": t0 + gen_hours * 3600}
    path = os.path.join(out, "run.json")
    json.dump(full, open(path, "w"), indent=1)
    status.update(n_prompts=len(prompts), model_dir=full["model_dir"])
    print(json.dumps(status, indent=1), flush=True)

    import torch
    n_gpu = max(1, torch.cuda.device_count())
    env = dict(os.environ, PYTHONPATH=run["code_dir"])
    libs = cuda12_lib_dirs()
    if libs:
        env["LD_LIBRARY_PATH"] = ":".join(libs + [env.get("LD_LIBRARY_PATH", "")]).rstrip(":")
    hard_stop = t0 + (gen_hours + 1.0) * 3600  # a config started just before the deadline may finish

    def remaining():
        claims = set(os.listdir(os.path.join(out, "_claims"))) if os.path.isdir(os.path.join(out, "_claims")) else set()
        return [c for c in configs if c["name"] not in claims
                and not os.path.exists(os.path.join(out, c["name"], "meta.json"))]

    def gpu_loop(g):
        # One fresh process per configuration: whatever a library leaks dies with it.
        e = dict(env, CUDA_VISIBLE_DEVICES=str(g))
        with open(os.path.join(out, f"worker{g}.log"), "a") as logf:
            while remaining() and time.time() < full["deadline_epoch"]:
                p = subprocess.Popen([sys.executable, "-m", "aqs.worker", path, "--one"], env=e,
                                     stdout=logf, stderr=subprocess.STDOUT)
                try:
                    p.wait(timeout=max(1, hard_stop - time.time()))
                except subprocess.TimeoutExpired:
                    p.kill()
                    print(f"gpu{g} worker killed at hard stop", flush=True)
                    return

    import threading
    threads = []
    for g in range(n_gpu):
        t = threading.Thread(target=gpu_loop, args=(g,))
        t.start()
        threads.append(t)
        time.sleep(20)  # stagger model loading
    for t in threads:
        t.join()
    for g in range(n_gpu):
        print(open(os.path.join(out, f"worker{g}.log")).read()[-8000:], flush=True)

    if run.get("judge", True):
        try:
            judge_all(out, deadline=t0 + (KAGGLE_LIMIT_H - 0.3) * 3600)
        except Exception:
            status["judge_error"] = traceback.format_exc()[-3000:]
            print(status["judge_error"], flush=True)

    names = [n for n in os.listdir(out) if os.path.isdir(os.path.join(out, n)) and not n.startswith("_")]
    status["minutes"] = round((time.time() - t0) / 60, 1)
    status["done"] = sorted(n for n in names if os.path.exists(os.path.join(out, n, "meta.json")))
    status["judged"] = sorted(n for n in names if os.path.exists(os.path.join(out, n, "judgments.jsonl")))
    status["failed"] = sorted(n for n in names if os.path.exists(os.path.join(out, n, "error.txt")))
    claims = os.listdir(os.path.join(out, "_claims")) if os.path.isdir(os.path.join(out, "_claims")) else []
    status["orphaned"] = sorted(c for c in claims if c not in status["done"] and c not in status["failed"])
    status["not_started"] = sorted(c["name"] for c in configs if c["name"] not in claims)
    json.dump(status, open(os.path.join(out, "status.json"), "w"), indent=1)
    print(json.dumps(status, indent=1), flush=True)
