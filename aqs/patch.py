"""Weight-space safety patch for a compressed model.

The information lost to quantization is dW = W_fp16 - W_quantized. A patch restores dW
for a small set of MLP neurons only (one neuron = row i of gate_proj and up_proj plus
column i of down_proj). Neurons are chosen by SafeTune safety scores, by quantization
error, or at random, always at the same size, so the choice of neurons is the only
difference between patches. No training is involved.
"""

import glob
import json
import os
import random
import time
import traceback

import torch

from . import config as C

MLP_ROWS = ("gate_proj", "up_proj")  # neuron i = row i
MLP_COL = "down_proj"                # neuron i = column i
SCORE_TOPK = 2048                    # neurons scored per layer before global ranking


class FP16Weights:
    """Reads original fp16 tensors straight from the checkpoint's safetensors files."""

    def __init__(self, model_dir):
        from safetensors import safe_open

        self.files = {}
        for f in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
            h = safe_open(f, framework="pt")
            for k in h.keys():
                self.files[k] = h

    def get(self, layer, proj):
        key = f"model.layers.{layer}.mlp.{proj}.weight"
        return self.files[key].get_tensor(key).to(torch.float16)


def _mlp(model, layer):
    return model.model.layers[layer].mlp


def _global_top(triples, n):
    triples = sorted(triples, key=lambda t: t[2], reverse=True)[:n]
    picks = {}
    for layer, idx, _ in triples:
        picks.setdefault(int(layer), []).append(int(idx))
    return {l: sorted(v) for l, v in picks.items()}


def safety_activation_neurons(model, tok, harmful, harmless):
    from safetune.instrumentation.interpret.safety_neurons import (
        SafetyNeuronConfig, identify_safety_neurons,
    )

    cfg = SafetyNeuronConfig(method="activation", activation_score="tstat",
                             top_k_per_layer=SCORE_TOPK, activation_max_tokens=64)
    rep = identify_safety_neurons(model, {}, cfg, tokenizer=tok,
                                  harmful_prompts=harmful, harmless_prompts=harmless)
    return [(l, i, abs(float(s))) for l, items in rep.per_layer.items() for i, s in items]


def safety_direction_neurons(model, tok, harmful, harmless):
    from safetune.instrumentation.interpret.safety_neurons import (
        SafetyNeuronConfig, identify_safety_neurons,
    )
    from safetune.interventions.steer.refusal_direction import (
        RefusalDirectionConfig, extract_refusal_direction,
    )

    _, _, per_layer = extract_refusal_direction(
        model, tok, harmful, harmless, RefusalDirectionConfig(select_directions=False))
    rep = identify_safety_neurons(model, per_layer,
                                  SafetyNeuronConfig(method="weight", top_k_per_layer=SCORE_TOPK))
    return [(l, i, abs(float(s))) for l, items in rep.per_layer.items() for i, s in items]


@torch.no_grad()
def error_neurons(model, fp):
    """Quantization error per neuron: squared norm of dW over its 3 weight vectors."""
    out = []
    for l in range(C.N_LAYERS):
        m = _mlp(model, l)
        err = torch.zeros(m.gate_proj.weight.shape[0], device="cpu")
        for p in MLP_ROWS:
            d = fp.get(l, p).float() - getattr(m, p).weight.detach().float().cpu()
            err += d.pow(2).sum(1)
        d = fp.get(l, MLP_COL).float() - m.down_proj.weight.detach().float().cpu()
        err += d.pow(2).sum(0)
        top = torch.topk(err, SCORE_TOPK)
        out += [(l, int(i), float(v)) for v, i in zip(top.values, top.indices)]
    return out


def random_neurons(n, seed):
    rng = random.Random(10_000 + seed)
    width = C.MLP_WIDTH
    flat = rng.sample(range(C.N_LAYERS * width), n)
    picks = {}
    for f in flat:
        picks.setdefault(f // width, []).append(f % width)
    return {l: sorted(v) for l, v in picks.items()}


@torch.no_grad()
def apply(model, fp, picks):
    """Write fp16 values into the picked neurons. Returns what is needed to undo it."""
    undo = {}
    for l, idx in picks.items():
        m = _mlp(model, l)
        ix = torch.tensor(idx, dtype=torch.long)
        saved = {}
        for p in MLP_ROWS:
            w = getattr(m, p).weight
            saved[p] = w.data[ix.to(w.device)].clone()
            w.data[ix.to(w.device)] = fp.get(l, p)[ix].to(w.device, w.dtype)
        w = m.down_proj.weight
        saved[MLP_COL] = w.data[:, ix.to(w.device)].clone()
        w.data[:, ix.to(w.device)] = fp.get(l, MLP_COL)[:, ix].to(w.device, w.dtype)
        undo[l] = (ix, saved)
    return undo


@torch.no_grad()
def revert(model, undo):
    for l, (ix, saved) in undo.items():
        m = _mlp(model, l)
        for p in MLP_ROWS:
            w = getattr(m, p).weight
            w.data[ix.to(w.device)] = saved[p]
        w = m.down_proj.weight
        w.data[:, ix.to(w.device)] = saved[MLP_COL]


def save_patch(path, fp, picks, info):
    """The shippable artifact: neuron indices plus their fp16 weights, per layer."""
    from safetensors.torch import save_file

    tensors = {}
    for l, idx in picks.items():
        ix = torch.tensor(idx, dtype=torch.long)
        tensors[f"layer{l}.index"] = ix.to(torch.int32)
        tensors[f"layer{l}.gate_proj"] = fp.get(l, "gate_proj")[ix].contiguous()
        tensors[f"layer{l}.up_proj"] = fp.get(l, "up_proj")[ix].contiguous()
        tensors[f"layer{l}.down_proj"] = fp.get(l, MLP_COL)[:, ix].contiguous()
    save_file(tensors, path, metadata={k: str(v) for k, v in info.items()})
    return os.path.getsize(path) / 1e6


def patch_mb(picks, hidden=2048):
    n = sum(len(v) for v in picks.values())
    return n * 3 * hidden * 2 / 1e6


def run_patchset(run, spec, out, evaluate, log):
    """Quantize the target once, then evaluate it unpatched and under every patch."""
    from . import models as M
    from .data import load_chat_calibration, load_harmless_select

    t0 = time.time()
    tok = M.load_tokenizer(run["model_dir"])
    model = M.load_fp16(run["model_dir"], "cuda:0")
    fp = FP16Weights(run["model_dir"])
    harmful, harmless = run["select_harmful"], load_harmless_select()

    scores = {}
    for key, fn in [("safety_act", safety_activation_neurons), ("safety_dir", safety_direction_neurons)]:
        t = time.time()
        try:
            scores[key] = fn(model, tok, harmful, harmless)
            log(f"{spec['name']}: scored {key} ({time.time() - t:.0f}s)")
        except Exception:
            log(f"{spec['name']}: {key} scoring FAILED\n{traceback.format_exc()[-2000:]}")

    calib_texts = load_chat_calibration(tok) if spec.get("calib", "chat") == "chat" else None
    M.apply_gptq(model, tok, spec["bits"], None, calibration_texts=calib_texts)
    baked = M.bake_weight_quantization(model, tok)
    scores["largest_error"] = error_neurons(model, fp)
    log(f"{spec['name']}: quantized + baked {baked} ({time.time() - t0:.0f}s)")

    base = {"target": spec["name"], "bits": spec["bits"], "calibration": spec.get("calib", "chat"),
            "size_gb_unpatched": round(M.size_gb_from_config(run["model_dir"], "gptq", spec["bits"]), 3)}
    jobs = [("none", 0, None)]
    for per_layer in spec["budgets"]:
        n = per_layer * C.N_LAYERS
        for key in ("safety_act", "safety_dir", "largest_error"):
            if key in scores:
                jobs.append((key, per_layer, _global_top(scores[key], n)))
        for s in range(spec.get("random_seeds", 3)):
            jobs.append((f"random_{s}", per_layer, random_neurons(n, s)))

    patch_dir = os.path.join(out, "_patches")
    os.makedirs(patch_dir, exist_ok=True)
    for method, per_layer, picks in jobs:
        name = f"{spec['name']}_patch_{method}_{per_layer}" if picks else f"{spec['name']}_unpatched"
        if os.path.exists(os.path.join(out, name, "meta.json")):
            continue
        undo = apply(model, fp, picks) if picks else None
        extra = {**base, "patch_method": method, "neurons_per_layer_avg": per_layer,
                 "patch_neurons": sum(len(v) for v in picks.values()) if picks else 0,
                 "patch_mb": round(patch_mb(picks), 1) if picks else 0.0,
                 "layers_touched": len(picks) if picks else 0,
                 "neurons_by_layer": {int(l): len(v) for l, v in sorted(picks.items())} if picks else {}}
        if picks and method in ("safety_act", "safety_dir", "largest_error"):
            extra["patch_file_mb"] = round(save_patch(os.path.join(patch_dir, name + ".safetensors"),
                                                      fp, picks, extra), 1)
        extra["size_gb"] = round(base["size_gb_unpatched"] + extra["patch_mb"] / 1e3, 3)
        evaluate(name, model, tok, {"spec": {"name": name, "kind": "patch", **extra}, **extra})
        if undo:
            revert(model, undo)
    json.dump({k: sorted(v, key=lambda t: t[2], reverse=True)[:40000] for k, v in scores.items()},
              open(os.path.join(out, f"{spec['name']}_neuron_scores.json"), "w"))
