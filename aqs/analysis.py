"""Turn Kaggle outputs into tables and charts (runs on a laptop; numpy, pandas, matplotlib only).

Usage: python -m aqs.analysis <results_root> <report_dir>
<results_root> holds one or more downloaded kernel outputs; every folder that contains
generations.jsonl + judgments.jsonl + meta.json counts as one configuration.
"""

import glob
import json
import math
import os
import sys

import numpy as np
import pandas as pd

from . import config as C

REFUSED = {"refuse", "partial"}
B = 10_000
RNG = np.random.default_rng(C.SEED)

FAMILY_ORDER = ["fp16", "uniform (GPTQ)", "uniform (GPTQ, WikiText calib)", "bitsandbytes NF4",
                "official GGUF", "uniform (RTN)",
                "protect: safety (activation)", "protect: safety (direction)",
                "protect: quant error", "protect: random"]
# Evidence-notebook palette: forest, terracotta, ochre, sage on warm paper.
COLORS = {"fp16": "#2B2B26", "uniform (RTN)": "#5f7f9a", "uniform (GPTQ)": "#254D3E",
          "uniform (GPTQ, WikiText calib)": "#B96550", "bitsandbytes NF4": "#5f7f9a",
          "official GGUF": "#C18A3A", "protect: safety (activation)": "#8c3f2f",
          "protect: safety (direction)": "#d9a58f", "protect: quant error": "#7c9a5a",
          "protect: random": "#A9BBAA"}


def notebook_style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": "#F7F4EC", "axes.facecolor": "#F7F4EC", "savefig.facecolor": "#F7F4EC",
        "axes.edgecolor": "#8a8a7a", "axes.labelcolor": "#2B2B26", "xtick.color": "#2B2B26",
        "ytick.color": "#2B2B26", "text.color": "#2B2B26", "axes.spines.top": False,
        "axes.spines.right": False, "grid.color": "#e3ddcf"})


PATCH_FAMILIES = {"none": "unpatched", "safety_act": "patch: safety (activation)",
                  "safety_dir": "patch: safety (direction)", "largest_error": "patch: largest error"}
FAMILY_ORDER += ["unpatched", "patch: safety (activation)", "patch: safety (direction)",
                 "patch: largest error", "patch: random"]
COLORS.update({"unpatched": "#76b7b2", "patch: safety (activation)": "#e15759",
               "patch: safety (direction)": "#f28e2b", "patch: largest error": "#59a14f",
               "patch: random": "#9c9c9c"})


def family(spec):
    k = spec.get("kind")
    if k == "patch":
        m = spec.get("patch_method", "none")
        return "patch: random" if m.startswith("random") else PATCH_FAMILIES.get(m, m)
    src = spec.get("protect_from", "")
    if k == "fp16":
        return "fp16"
    if k == "gguf":
        return "official GGUF"
    if k == "bnb4" and not src:
        return "bitsandbytes NF4"
    if k == "gptq" and spec.get("calib") == "wikitext" and not src:
        return "uniform (GPTQ, WikiText calib)"
    if src == "safety_act":
        return "protect: safety (activation)"
    if src == "safety_dir":
        return "protect: safety (direction)"
    if src == "quant_error":
        return "protect: quant error"
    if src.startswith("random"):
        return "protect: random"
    return "uniform (GPTQ)" if k == "gptq" else "uniform (RTN)"


def load(results_root):
    configs = {}
    for meta_path in glob.glob(os.path.join(results_root, "**", "meta.json"), recursive=True):
        d = os.path.dirname(meta_path)
        g, j = os.path.join(d, "generations.jsonl"), os.path.join(d, "judgments.jsonl")
        if not (os.path.exists(g) and os.path.exists(j)):
            continue
        meta = json.load(open(meta_path))
        gen = pd.read_json(g, lines=True)
        jud = pd.read_json(j, lines=True)
        df = gen.merge(jud[["id", "refusal_label", "harmful", "keyword_refusal"]], on="id")
        df["refused"] = df["refusal_label"].isin(REFUSED)
        df["nonsense"] = df["refusal_label"].eq("nonsense")
        df["harmful_yes"] = df["harmful"].eq("yes")
        configs[meta["spec"]["name"]] = {"meta": meta, "df": df}
    layers = None
    for p in glob.glob(os.path.join(results_root, "**", "layers.json"), recursive=True):
        layers = json.load(open(p))
    return configs, layers


def boot_ci(x):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return (math.nan, math.nan)
    means = x[RNG.integers(0, len(x), size=(B, len(x)))].mean(1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def paired(a, b):
    """Paired difference b - a (same prompts): mean, 95% bootstrap CI, exact McNemar p."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) == 0:
        return math.nan, math.nan, math.nan, math.nan
    d = b - a
    idx = RNG.integers(0, len(d), size=(B, len(d)))
    lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
    n01, n10 = int(((a == 0) & (b == 1)).sum()), int(((a == 1) & (b == 0)).sum())
    n = n01 + n10
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, k) for k in range(min(n01, n10) + 1)) / 2 ** n)
    return float(d.mean()), float(lo), float(hi), p


def summarize(configs):
    rows = []
    base = configs.get("fp16")
    for name, c in configs.items():
        df, meta = c["df"], c["meta"]
        harm, benign = df[df.expected == "refuse"], df[df.expected == "comply"]
        r = {"config": name, "family": family(meta["spec"]),
             "size_gb": meta.get("size_gb"), "arc_acc": meta.get("arc_acc"), "ppl": meta.get("ppl"),
             "protect": ",".join(map(str, meta.get("protect", []))) if meta.get("protect") else ""}
        for key, sub, col in [("harmful_compliance", harm, "harmful_yes"),
                              ("refusal_rate", harm, "refused"),
                              ("over_refusal", benign, "refused"),
                              ("nonsense_harmful", harm, "nonsense"),
                              ("nonsense_benign", benign, "nonsense"),
                              ("keyword_refusal_harmful", harm, "keyword_refusal")]:
            v = sub[col].astype(float)
            r[key] = v.mean()
            r[key + "_lo"], r[key + "_hi"] = boot_ci(v)
        for src in ["harmbench", "advbench", "xstest_safe", "orbench_hard"]:
            s = df[df.source == src]
            col = "harmful_yes" if src in ("harmbench", "advbench") else "refused"
            r[f"{src}_{'hc' if col == 'harmful_yes' else 'orr'}"] = s[col].astype(float).mean()
        if base is not None and name != "fp16":
            bdf = base["df"].set_index("id")
            for key, exp, col in [("d_harmful_compliance", "refuse", "harmful_yes"),
                                  ("d_refusal_rate", "refuse", "refused"),
                                  ("d_over_refusal", "comply", "refused")]:
                cur = df[df.expected == exp].set_index("id")
                ids = cur.index.intersection(bdf.index)
                m, lo, hi, p = paired(bdf.loc[ids, col].astype(float), cur.loc[ids, col].astype(float))
                r[key], r[key + "_lo"], r[key + "_hi"], r[key + "_p"] = m, lo, hi, p
        rows.append(r)
    out = pd.DataFrame(rows)
    out["family_rank"] = out["family"].map({f: i for i, f in enumerate(FAMILY_ORDER)})
    return out.sort_values(["family_rank", "size_gb"], ascending=[True, False]).drop(columns="family_rank")


def plot_headline(summary, path, metric="harmful_compliance", label="Harmful compliance (HarmBench judge)"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    notebook_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    for fam in FAMILY_ORDER:
        s = summary[summary.family == fam]
        if s.empty:
            continue
        y = s[metric] * 100
        err = np.vstack([y - s[metric + "_lo"] * 100, s[metric + "_hi"] * 100 - y])
        ax.errorbar(s["size_gb"], y, yerr=err, fmt="o", ms=7, capsize=3, color=COLORS[fam], label=fam)
        for _, r in s.iterrows():
            ax.annotate(r["config"], (r["size_gb"], r[metric] * 100), fontsize=7,
                        xytext=(4, 4), textcoords="offset points", color="#444")
    ax.set_xlabel("Model weight size (GB)")
    ax.set_ylabel(label + " (%)")
    ax.set_title("Tiny Aya Fire: safety vs size under quantization")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_bars(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = [("harmful_compliance", "Harmful compliance ↓"), ("refusal_rate", "Refusal of harmful ↑"),
            ("over_refusal", "Over-refusal of harmless ↓"), ("arc_acc", "ARC-Challenge accuracy ↑")]
    fig, axes = plt.subplots(1, 4, figsize=(16, 0.45 * len(summary) + 1.5), sharey=True)
    ylab = summary["config"].tolist()[::-1]
    for ax, (col, title) in zip(axes, cols):
        s = summary[::-1]
        vals = s[col].astype(float) * 100
        ax.barh(ylab, vals, color=[COLORS[f] for f in s["family"]])
        if col + "_lo" in s:
            ax.errorbar(vals, ylab, xerr=np.vstack([vals - s[col + "_lo"] * 100, s[col + "_hi"] * 100 - vals]),
                        fmt="none", ecolor="#333", capsize=2, lw=0.8)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_layers(layers, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not layers or not layers.get("scores"):
        return
    keys = list(layers["scores"])
    fig, axes = plt.subplots(len(keys), 1, figsize=(9, 2.2 * len(keys)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, k in zip(axes, keys):
        sc = {int(l): v for l, v in layers["scores"][k].items()}
        xs = list(range(C.N_LAYERS))
        ax.bar(xs, [sc.get(l, 0) for l in xs],
               color=["#e15759" if l in layers["protect"].get(k, []) else "#bbb" for l in xs])
        ax.set_ylabel(k, fontsize=9)
    axes[-1].set_xlabel("Decoder layer (red = kept at native precision)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def label_sheet(configs, path, n=150, seed=C.SEED):
    """Stratified sample for hand labelling, with judge labels hidden."""
    rng = np.random.default_rng(seed)
    pools = []
    for name, c in configs.items():
        for exp in ("refuse", "comply"):
            s = c["df"][c["df"].expected == exp]
            if len(s):
                pools.append(s.assign(config=name))
    per = max(1, n // max(1, len(pools)))
    picks = pd.concat([p.iloc[rng.permutation(len(p))[:per]] for p in pools])
    picks = picks.iloc[rng.permutation(len(picks))[:n]]
    sheet = picks[["config", "id", "source", "prompt", "response"]].copy()
    sheet.insert(0, "item", range(1, len(sheet) + 1))
    sheet["your_label (refuse/partial/comply/nonsense)"] = ""
    sheet["harmful_content (yes/no, harmful prompts only)"] = ""
    sheet.to_csv(path, index=False, encoding="utf-8-sig")
    key = picks[["config", "id", "refusal_label", "harmful"]].copy()
    key.insert(0, "item", range(1, len(key) + 1))
    key.to_csv(path.replace(".csv", "_judge_key.csv"), index=False)


def kappa(a, b):
    a, b = list(a), list(b)
    cats = sorted(set(a) | set(b))
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def main(results_root, report_dir):
    os.makedirs(report_dir, exist_ok=True)
    configs, layers = load(results_root)
    if not configs:
        print("no judged configurations found under", results_root)
        return
    summary = summarize(configs)
    summary.to_csv(os.path.join(report_dir, "summary.csv"), index=False)
    show = ["config", "family", "size_gb", "harmful_compliance", "refusal_rate", "over_refusal",
            "nonsense_harmful", "arc_acc", "ppl", "d_harmful_compliance", "d_harmful_compliance_p",
            "d_over_refusal", "protect"]
    with open(os.path.join(report_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write(summary[[c for c in show if c in summary]].round(4).to_markdown(index=False))
    plot_headline(summary, os.path.join(report_dir, "headline_harmful_compliance.png"))
    plot_headline(summary, os.path.join(report_dir, "headline_over_refusal.png"), "over_refusal",
                  "Over-refusal of harmless prompts")
    plot_bars(summary, os.path.join(report_dir, "all_metrics.png"))
    plot_layers(layers, os.path.join(report_dir, "layer_scores.png"))
    label_sheet(configs, os.path.join(report_dir, "hand_labels.csv"))
    print(summary[[c for c in show if c in summary]].round(3).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
