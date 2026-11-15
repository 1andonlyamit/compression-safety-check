"""General-ability checks: WikiText-2 perplexity (CircuitKIT) and ARC-Challenge (AuditKIT)."""

import json
import os

from . import config as C
from .generate import first_letter


def perplexity(model, tok):
    from circuitkit.applications.quantization.quant_utils import compute_ppl

    return float(compute_ppl(model, tok, device="cuda", n_samples=C.PPL_SAMPLES, seq_len=C.PPL_SEQ_LEN))


def arc_accuracy(gen_fn, name, arc_items, out_dir):
    """Run ARC through AuditKIT. `gen_fn(list[str]) -> list[str]` is the configuration's generator."""
    import auditkit as ak
    from auditkit.model import CallableModel

    letters = {it["prompt"]: it["letters"] for it in arc_items}

    def answer(prompts):
        return [first_letter(o, letters.get(p, "ABCDE")) for p, o in zip(prompts, gen_fn(prompts))]

    samples = [ak.Sample(input=it["prompt"], target=it["gold"], id=it["id"]) for it in arc_items]
    # A unique name per configuration avoids AuditKIT's cache returning another config's run.
    model = CallableModel(answer, name=f"tiny-aya-fire:{name}")
    run = ak.evaluate(samples, model, ["exact_match"], config=ak.RunConfig(temperature=0.0))
    os.makedirs(out_dir, exist_ok=True)
    run.save(os.path.join(out_dir, "auditkit_arc.json"))
    table = {row["name"]: row for row in run.metric_table()}
    acc = float(table["exact_match"]["mean"])
    with open(os.path.join(out_dir, "auditkit_arc_summary.txt"), "w") as f:
        f.write(run.summary())
    return acc
