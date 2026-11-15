"""Apply (or remove) a weight-space safety patch on a compressed Tiny Aya.

A patch file stores, for a few MLP neurons per layer, their original fp16 weights:
  layer{L}.index      int32 [n]          neuron indices
  layer{L}.gate_proj  fp16  [n, hidden]  rows of gate_proj
  layer{L}.up_proj    fp16  [n, hidden]  rows of up_proj
  layer{L}.down_proj  fp16  [hidden, n]  columns of down_proj

Usage:
    model = AutoModelForCausalLM.from_pretrained(compressed_repo, ...)  # dense weights
    undo = apply_patch(model, "safety_patch.safetensors")
    ...                                   # patched model
    remove_patch(model, undo)             # back to the compressed model
"""

import torch
from safetensors.torch import load_file


@torch.no_grad()
def apply_patch(model, path):
    patch = load_file(path)
    layers = sorted({int(k.split(".")[0][5:]) for k in patch})
    undo = {}
    for l in layers:
        mlp = model.model.layers[l].mlp
        dev = mlp.gate_proj.weight.device
        ix = patch[f"layer{l}.index"].long().to(dev)
        undo[l] = (ix, mlp.gate_proj.weight[ix].clone(), mlp.up_proj.weight[ix].clone(),
                   mlp.down_proj.weight[:, ix].clone())
        dtype = mlp.gate_proj.weight.dtype
        mlp.gate_proj.weight[ix] = patch[f"layer{l}.gate_proj"].to(dev, dtype)
        mlp.up_proj.weight[ix] = patch[f"layer{l}.up_proj"].to(dev, dtype)
        mlp.down_proj.weight[:, ix] = patch[f"layer{l}.down_proj"].to(dev, dtype)
        # If compressed-tensors still quantizes weights on the fly, it would round the
        # restored rows again. The loaded weights are already dequantized, so turning that
        # off changes nothing else (the evaluation pipeline did the same).
        for proj in (mlp.gate_proj, mlp.up_proj, mlp.down_proj):
            if hasattr(proj, "quantization_enabled"):
                proj.quantization_enabled = False
    return undo


@torch.no_grad()
def remove_patch(model, undo):
    for l, (ix, gate, up, down) in undo.items():
        mlp = model.model.layers[l].mlp
        mlp.gate_proj.weight[ix] = gate
        mlp.up_proj.weight[ix] = up
        mlp.down_proj.weight[:, ix] = down
