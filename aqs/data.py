"""Prompt sets: evaluation prompts and disjoint layer-selection prompts."""

import glob
import os
import random

import pandas as pd

from . import config as C


def find_one(root, pattern):
    hits = sorted(glob.glob(os.path.join(root, "**", pattern), recursive=True))
    if not hits:
        raise FileNotFoundError(f"{pattern} not found under {root}")
    return hits[0]


def find_input_dir(marker, base="/kaggle/input"):
    """Return the directory under `base` that contains `marker` (a relative path)."""
    hits = sorted(glob.glob(os.path.join(base, "**", marker), recursive=True))
    if not hits:
        raise FileNotFoundError(f"{marker} not found under {base}")
    return hits[0][: -len(marker)].rstrip("/")


def _sample(df, n, seed=C.SEED):
    if n >= len(df):
        return df.reset_index(drop=True), df.iloc[0:0]
    idx = list(range(len(df)))
    random.Random(seed).shuffle(idx)
    return df.iloc[idx[:n]].reset_index(drop=True), df.iloc[idx[n:]].reset_index(drop=True)


def _rows(df, source, expected, category_col=None, extra=None):
    out = []
    for i, r in df.iterrows():
        out.append({
            "id": f"{source}-{i:04d}",
            "source": source,
            "expected": expected,  # "refuse" for harmful prompts, "comply" for harmless ones
            "category": str(r[category_col]) if category_col and category_col in r else "",
            "prompt": str(r["prompt"]),
            **({k: str(r[k]) for k in extra if k in r} if extra else {}),
        })
    return out


def load_sets(root):
    """Return (eval_prompts, select_harmful, advbench_pool_left)."""
    hb = pd.read_parquet(find_one(root, os.path.join("standard", "*.parquet")))
    adv = pd.read_parquet(find_one(root, os.path.join("walledai__AdvBench", "**", "*.parquet")))
    xs = pd.read_parquet(find_one(root, os.path.join("walledai__XSTest", "**", "*.parquet")))
    orb = pd.read_csv(find_one(root, "or-bench-hard-1k.csv"))

    hb, _ = _sample(hb, C.N_HARMBENCH)
    adv_eval, adv_rest = _sample(adv, C.N_ADVBENCH)
    xs_safe = xs[xs["label"].astype(str).str.lower() == "safe"]
    xs_safe, _ = _sample(xs_safe, C.N_XSTEST_SAFE)
    orb, _ = _sample(orb, C.N_ORBENCH_HARD)

    prompts = (
        _rows(hb, "harmbench", "refuse", "category")
        + _rows(adv_eval, "advbench", "refuse")
        + _rows(xs_safe, "xstest_safe", "comply", "type", extra=["focus"])
        + _rows(orb, "orbench_hard", "comply", "category")
    )
    # Harmful prompts for locating safety layers come from a different benchmark
    # (OR-Bench toxic) so they cannot near-duplicate any evaluation prompt.
    toxic = pd.read_csv(find_one(root, "or-bench-toxic.csv"))
    select_harmful, _ = _sample(toxic, C.N_SELECT, seed=C.SEED + 1)
    return prompts, [str(p) for p in select_harmful["prompt"]]


def load_harmless_select(n=C.N_SELECT, seed=C.SEED + 1):
    """Plain harmless instructions (Alpaca), used only to locate safety-relevant layers."""
    from datasets import load_dataset

    ds = load_dataset("tatsu-lab/alpaca", split="train")
    rows = [r["instruction"] for r in ds if not r["input"]]
    random.Random(seed).shuffle(rows)
    return rows[:n]


def load_chat_calibration(tok, n=128, seed=C.SEED + 2):
    """GPTQ calibration texts in Tiny Aya's chat format (Alpaca instruction + answer).

    Calibrating on plain WikiText never shows the quantizer the chat special tokens, and
    the quantized model then stops emitting its response-start token.
    """
    from datasets import load_dataset

    rows = [r for r in load_dataset("tatsu-lab/alpaca", split="train") if r["output"]]
    random.Random(seed).shuffle(rows)
    texts = []
    for r in rows[:n]:
        user = r["instruction"] + (f"\n\n{r['input']}" if r["input"] else "")
        texts.append(tok.apply_chat_template([{"role": "user", "content": user},
                                              {"role": "assistant", "content": r["output"]}],
                                             tokenize=False))
    return texts


def load_arc(n=C.N_ARC, seed=C.SEED):
    """ARC-Challenge test questions as (prompt, gold_letter) pairs."""
    from datasets import load_dataset

    ds = list(load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test"))
    random.Random(seed).shuffle(ds)
    out = []
    for r in ds:
        labels, texts = r["choices"]["label"], r["choices"]["text"]
        if r["answerKey"] not in labels or len(labels) > 5:
            continue
        letters = "ABCDE"[: len(labels)]
        gold = letters[labels.index(r["answerKey"])]
        opts = "\n".join(f"{l}. {t}" for l, t in zip(letters, texts))
        prompt = (f"{r['question']}\n\n{opts}\n\n"
                  f"Answer with the letter of the correct option only.")
        out.append({"id": r["id"], "prompt": prompt, "gold": gold, "letters": letters})
        if len(out) >= n:
            break
    return out
