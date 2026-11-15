"""Charts and tables for the weight-space patch experiment (laptop side).

Usage: python -m aqs.patch_report <patch_results> <main_results> <report_dir>
"""

import json
import os
import sys

import numpy as np
import pandas as pd

from .analysis import load, summarize

METHODS = [("largest_error", "Most-damaged neurons", "#254D3E"),
           ("safety_act", "SafeTune safety neurons (activation)", "#B96550"),
           ("safety_dir", "SafeTune safety neurons (refusal direction)", "#C18A3A"),
           ("random", "Random neurons (3 seeds)", "#A9BBAA")]
TARGETS = {"gptq4_wikitext": "4-bit GPTQ, WikiText calibration (broken)",
           "gptq3": "3-bit GPTQ, chat calibration"}
METRICS = [("harmful_compliance", "Harmful compliance (%) ↓"), ("refusal_rate", "Refuses harmful (%) ↑"),
           ("nonsense_harmful", "Nonsense on harmful (%) ↓"), ("arc_acc", "ARC-Challenge (%) ↑")]


def table(patch_root, main_root):
    P, _ = load(patch_root)
    M, _ = load(main_root)
    P["fp16"] = M["fp16"]
    s = summarize(P)
    rows = []
    for name, c in P.items():
        sp = c["meta"]["spec"]
        if sp.get("kind") == "patch":
            rows.append({"config": name, "target": sp["target"], "method": sp["patch_method"],
                         "budget": sp["neurons_per_layer_avg"], "patch_mb": sp.get("patch_mb", 0.0)})
    return s.merge(pd.DataFrame(rows), on="config", how="left"), s[s.config == "fp16"].iloc[0]


def plot(t, fp16, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .analysis import notebook_style
    notebook_style()
    fig, axes = plt.subplots(len(TARGETS), len(METRICS), figsize=(17, 7.5), sharex=True)
    for r, (tgt, title) in enumerate(TARGETS.items()):
        x = t[t.target == tgt]
        base = x[x.method == "none"].iloc[0]
        for c, (col, lab) in enumerate(METRICS):
            ax = axes[r, c]
            ax.axhline(fp16[col] * 100, color="#1b1b1b", ls="--", lw=1, label="full size (fp16)")
            ax.axhline(base[col] * 100, color="#76b7b2", ls=":", lw=1.5, label="compressed, no patch")
            budgets = sorted(x.budget.dropna().unique())
            budgets = [b for b in budgets if b > 0]
            for key, name, color in METHODS:
                if key == "random":
                    sub = x[x.method.str.startswith("random")]
                    g = sub.groupby("budget")[col]
                    m, lo, hi = g.mean() * 100, g.min() * 100, g.max() * 100
                    ax.fill_between(m.index, lo, hi, color=color, alpha=0.25)
                    ax.plot(m.index, m, "o-", color=color, label=name)
                else:
                    sub = x[x.method == key].sort_values("budget")
                    ax.plot(sub.budget, sub[col] * 100, "o-", color=color, label=name, lw=2)
            ax.set_xscale("log", base=2)
            ax.set_xticks(budgets)
            ax.set_xticklabels([f"{int(b)}\n({int(round(b * 36 * 6144 * 2 / 1e6))} MB)" for b in budgets], fontsize=7)
            if r == 0:
                ax.set_title(lab, fontsize=10)
            if c == 0:
                ax.set_ylabel(title, fontsize=9)
            ax.grid(alpha=0.3)
    axes[-1, 0].set_xlabel("Neurons restored per layer (patch size)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_layers(patch_root, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 3.2))
    width = 0.27
    for k, (name, color, label) in enumerate([
            ("gptq4_wikitext_patch_largest_error_64", "#59a14f", "largest-error neurons"),
            ("gptq4_wikitext_patch_safety_act_64", "#e15759", "safety neurons (activation)"),
            ("gptq4_wikitext_patch_safety_dir_64", "#f28e2b", "safety neurons (direction)")]):
        meta = json.load(open(os.path.join(patch_root, name, "meta.json")))
        nbl = {int(a): b for a, b in meta["neurons_by_layer"].items()}
        xs = np.arange(36)
        ax.bar(xs + (k - 1) * width, [nbl.get(i, 0) for i in xs], width, color=color, label=label)
    ax.set_xlabel("Decoder layer")
    ax.set_ylabel("Neurons restored")
    ax.set_title("Where each 28 MB patch puts its 2,304 neurons (4-bit, WikiText calibration)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main(patch_root, main_root, report_dir):
    os.makedirs(report_dir, exist_ok=True)
    t, fp16 = table(patch_root, main_root)
    t.to_csv(os.path.join(report_dir, "patch_summary.csv"), index=False)
    plot(t, fp16, os.path.join(report_dir, "patch_curves.png"))
    plot_layers(patch_root, os.path.join(report_dir, "patch_layers.png"))
    print("written to", report_dir)


if __name__ == "__main__":
    main(*sys.argv[1:4])
