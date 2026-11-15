"""Build the public, redacted data bundle for the Hugging Face Space and dataset.

Usage: python scripts/build_space_data.py <results_roots> <report_dirs> <out_dir> [model] [out_file]

Safety rule: for harmful prompts, only refusals and nonsense replies the HarmBench judge
marked harmless are published. Nonsense is withheld too when it lists numbered steps, comes
from AdvBench (no topic labels), or is in the chemical/biological or cybercrime categories. The text of every other reply to a harmful prompt (judged
harmful, or a coherent answer) is withheld; its prompt and labels are kept.
"""

import glob
import json
import os
import re
import shutil
import sys

STEPS = re.compile(r"(?mi)^\s*(\d+\.|step \d)")  # nonsense that still lists steps is withheld
HIGH_RISK = {"chemical_biological", "cybercrime_intrusion"}  # nonsense withheld here too
REDACTED = "[withheld: answer to a harmful prompt (judged harmful or a coherent answer); labels kept]"


def main(results_roots, report_dirs, out_dir, model="tiny-aya-fire", out_file="responses_redacted.jsonl"):
    """results_roots and report_dirs may each be several paths joined with commas."""
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    metas = []
    for root in results_roots.split(","):
        metas += sorted(glob.glob(os.path.join(root, "**", "meta.json"), recursive=True))
    for meta_path in metas:
        d = os.path.dirname(meta_path)
        g, j = os.path.join(d, "generations.jsonl"), os.path.join(d, "judgments.jsonl")
        if not (os.path.exists(g) and os.path.exists(j)):
            continue
        name = json.load(open(meta_path))["spec"]["name"]
        jud = {x["id"]: x for x in map(json.loads, open(j, encoding="utf-8"))}
        for line in open(g, encoding="utf-8"):
            r = json.loads(line)
            k = jud.get(r["id"], {})
            harmful_prompt = r["expected"] == "refuse"
            harm, label = k.get("harmful"), k.get("refusal_label")
            # Show refusals and judged-harmless nonsense (the evidence of collapse); withhold
            # anything judged harmful and every coherent answer to a harmful prompt.
            safe_to_show = (not harmful_prompt) or (harm != "yes" and label == "refuse") or                 (harm == "no" and label == "nonsense" and r["source"] == "harmbench"
                 and r.get("category") not in HIGH_RISK and not STEPS.search(r["response"]))
            rows.append({
                "model": model, "config": name, "id": r["id"], "source": r["source"], "category": r.get("category", ""), "expected": r["expected"],
                "prompt": r["prompt"],
                "response": r["response"] if safe_to_show else REDACTED,
                "refusal_label": k.get("refusal_label"), "harmful": k.get("harmful"),
            })
    with open(os.path.join(out_dir, out_file), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for report_dir in [d for d in report_dirs.split(",") if d]:
        for name in ["summary.csv", "summary.md", "headline_harmful_compliance.png",
                     "headline_over_refusal.png", "all_metrics.png", "layer_scores.png",
                     "patch_summary.csv", "patch_curves.png", "patch_layers.png"]:
            p = os.path.join(report_dir, name)
            if os.path.exists(p):
                shutil.copy(p, out_dir)
    layers = glob.glob(os.path.join(results_roots.split(",")[0], "**", "layers.json"), recursive=True)
    if layers and model == "tiny-aya-fire":
        lj = json.load(open(layers[0]))
        json.dump({"protect": lj["protect"], "scores": lj["scores"], "notes": lj["notes"]},
                  open(os.path.join(out_dir, "layers.json"), "w"), indent=1)
    print(f"{len(rows)} rows, {sum(r['response'] == REDACTED for r in rows)} withheld -> {out_dir}")


if __name__ == "__main__":
    main(*sys.argv[1:6])
