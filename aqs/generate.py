"""Greedy generation through Tiny Aya's own chat template (which adds its safety preamble).

The HF and GGUF paths use the same rendered prompt (from the HF tokenizer's chat
template) and stop on the same token ids, so every configuration is generated alike.
"""

import re

import torch

from . import config as C

SPECIAL = re.compile(r"<\|[A-Z_]+\|>|<BOS_TOKEN>|<EOS_TOKEN>")


def chat_texts(tok, prompts):
    return [tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True,
                                    tokenize=False) for p in prompts]


@torch.no_grad()
def generate_hf(model, tok, prompts, max_new_tokens=C.MAX_NEW_TOKENS, batch=C.GEN_BATCH):
    texts = chat_texts(tok, prompts)
    # Batch similar lengths together to waste less padding, then restore order.
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out = [None] * len(texts)
    device = next(model.parameters()).device
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             eos_token_id=C.EOS_TOKEN_IDS, pad_token_id=tok.pad_token_id)
        dec = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for i, d in zip(idx, dec):
            out[i] = d.strip()
    return out


def generate_gguf(llm, tok, prompts, max_new_tokens=C.MAX_NEW_TOKENS):
    """Greedy llama.cpp generation from the HF-rendered prompt, stopping on Tiny Aya's turn tokens."""
    from llama_cpp import StoppingCriteriaList

    stop_ids = set(C.EOS_TOKEN_IDS)
    stop = StoppingCriteriaList([lambda ids, logits: len(ids) > 0 and int(ids[-1]) in stop_ids])
    out = []
    for text in chat_texts(tok, prompts):
        ids = llm.tokenize(text.encode("utf-8"), add_bos=False, special=True)
        r = llm.create_completion(ids, max_tokens=max_new_tokens, temperature=0.0, top_k=1,
                                  stopping_criteria=stop)
        out.append(SPECIAL.sub("", r["choices"][0]["text"] or "").strip())
    return out


ANCHORED = [re.compile(p) for p in (
    r"(?i:answer(?: is)?)\s*[:\-]?\s*[\*\(\[]*([A-E])\b",   # "The answer is D"
    r"^\s*[\*\(\[]*([A-E])(?:[\*\)\]\.:,]|\s*$)",          # "B", "(B)", "**B**", "B."
    r"\b([A-E])[\).:]",                                     # "... B)" "B." "B:"
    r"\b([A-E])\b",                                         # last resort
)]


def first_letter(text, letters="ABCDE"):
    """Extract the chosen option letter, case-sensitively (so the article 'a' is not 'A')."""
    for k, pat in enumerate(ANCHORED):
        hits = [m.group(1) for m in pat.finditer(text or "") if m.group(1) in letters]
        if hits:
            # A bare capital letter is ambiguous ("A good choice is B"); the last one wins.
            return hits[-1] if k == len(ANCHORED) - 1 else hits[0]
    return ""
