"""Rank Tiny Aya's decoder layers; the top PROTECT_K stay at native precision.

S1 safety_act:   SafeTune identify_safety_neurons, activation mode (t-statistic of MLP
                 neurons, harmful vs harmless prompts), summed per layer.
S2 safety_dir:   SafeTune identify_safety_neurons, weight mode (alignment of each MLP
                 output column with that layer's refusal direction from
                 extract_refusal_direction), summed per layer.
M  quant_error:  relative 4-bit round-to-nearest error of each layer's weights. A
                 safety-agnostic baseline: protect the layers quantization hurts most.
R  random:       uniformly random layers (several seeds).
"""

import random

import torch

from . import config as C

TOP_NEURONS = 256  # neurons per layer whose scores are summed into a layer score


def _layer_totals(report):
    totals = {l: 0.0 for l in range(C.N_LAYERS)}
    for layer, items in report.per_layer.items():
        totals[int(layer)] = float(sum(abs(float(s)) for _, s in items))
    return totals


def safety_activation_scores(model, tok, harmful, harmless):
    from safetune.instrumentation.interpret.safety_neurons import (
        SafetyNeuronConfig, identify_safety_neurons,
    )

    cfg = SafetyNeuronConfig(method="activation", activation_score="tstat",
                             top_k_per_layer=TOP_NEURONS, activation_max_tokens=64)
    rep = identify_safety_neurons(model, {}, cfg, tokenizer=tok,
                                  harmful_prompts=harmful, harmless_prompts=harmless)
    return _layer_totals(rep)


def safety_direction_scores(model, tok, harmful, harmless):
    from safetune.instrumentation.interpret.safety_neurons import (
        SafetyNeuronConfig, identify_safety_neurons,
    )
    from safetune.interventions.steer.refusal_direction import (
        RefusalDirectionConfig, extract_refusal_direction,
    )

    _, best_layer, per_layer = extract_refusal_direction(
        model, tok, harmful, harmless, RefusalDirectionConfig(select_directions=False))
    cfg = SafetyNeuronConfig(method="weight", top_k_per_layer=TOP_NEURONS)
    rep = identify_safety_neurons(model, per_layer, cfg)
    return _layer_totals(rep), int(best_layer)


@torch.no_grad()
def quant_error_scores(model, bits=4, group=128):
    totals = {l: [0.0, 0.0] for l in range(C.N_LAYERS)}
    for name, p in model.named_parameters():
        parts = name.split(".")
        if "layers" not in parts or p.dim() != 2 or not name.endswith(".weight"):
            continue
        layer = int(parts[parts.index("layers") + 1])
        w = p.detach().float()
        rows, cols = w.shape
        if cols % group:
            continue
        g = w.reshape(rows, cols // group, group)
        lo, hi = g.amin(-1, keepdim=True), g.amax(-1, keepdim=True)
        scale = (hi - lo).clamp(min=1e-8) / (2 ** bits - 1)
        q = ((g - lo) / scale).round().clamp(0, 2 ** bits - 1) * scale + lo
        totals[layer][0] += float((q - g).pow(2).sum())
        totals[layer][1] += float(g.pow(2).sum())
    return {l: e / max(n, 1e-12) for l, (e, n) in totals.items()}


def top_k(scores, k=C.PROTECT_K):
    return sorted(sorted(scores, key=lambda l: scores[l], reverse=True)[:k])


def random_layers(seed, k=C.PROTECT_K):
    return sorted(random.Random(1000 + seed).sample(range(C.N_LAYERS), k))
