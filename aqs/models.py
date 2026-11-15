"""Load Tiny Aya Fire and build each precision configuration."""

import gc
import glob
import json
import os

# quanto compiles a small CUDA extension on first use; on T4 (sm75) the build
# fails unless the arch list names the actual GPU (see CircuitKIT's
# examples/quantization/quantize-llama.py).
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.5")

import torch

from . import config as C


def find_model_dir(base="/kaggle/input"):
    for cfg in sorted(glob.glob(os.path.join(base, "**", "config.json"), recursive=True)):
        try:
            with open(cfg) as f:
                if json.load(f).get("model_type") == "cohere2":
                    return os.path.dirname(cfg)
        except Exception:
            continue
    raise FileNotFoundError("Tiny Aya (cohere2) config.json not found under " + base)


def load_tokenizer(model_dir):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_fp16(model_dir, device="cuda:0"):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float16, device_map=device)
    return model.eval()


def load_bnb4(model_dir, protect=None, device="cuda:0"):
    """bitsandbytes NF4 4-bit loading (the usual `load_in_4bit` path); `protect` layers stay fp16."""
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    skip = ["lm_head"] + [f"model.layers.{i}.{m}" for i in (protect or [])
                          for m in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                                    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")]
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                           bnb_4bit_compute_dtype=torch.float16, llm_int8_skip_modules=skip)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float16, device_map=device,
                                                 quantization_config=q)
    return model.eval()


def apply_quanto(model, bits, protect=None):
    """Round-to-nearest weight quantization of every decoder layer except `protect`."""
    from optimum.quanto import qint2, qint4, qint8
    from circuitkit.applications.quantization.quant_utils import circuit_quantize

    qtype = {8: qint8, 4: qint4, 2: qint2}[bits]
    return circuit_quantize(model, {}, {}, C.N_LAYERS, low_weights=qtype, high_weights=None,
                            high_fraction=0.0, protect_layers=list(protect) if protect else None)


def apply_gptq(model, tok, bits, protect=None, n_calib=128, calibration_texts=None):
    """GPTQ weight quantization (llm-compressor) of every decoder layer except `protect`.

    calibration_texts=None uses CircuitKIT's default (WikiText-2 train).
    """
    from circuitkit.applications.quantization.llmcompressor_quantize import (
        llmcompressor_circuit_quantize,
    )

    try:
        return llmcompressor_circuit_quantize(model, tok, {}, {}, C.N_LAYERS, bits=bits,
                                              high_fraction=0.0,
                                              protect_layers=list(protect) if protect else None,
                                              n_calibration_samples=n_calib, max_seq_length=512,
                                              calibration_texts=calibration_texts)
    finally:
        # llm-compressor keeps the model in a global session after oneshot();
        # without a reset the next configuration on this GPU loads a second copy.
        try:
            from llmcompressor.core import reset_session
            reset_session()
        except Exception:
            pass
        gc.collect()
        torch.cuda.empty_cache()


@torch.no_grad()
def bake_weight_quantization(model, tok):
    """Apply llm-compressor's weight quantize-dequantize once instead of on every forward.

    After oneshot() the model runs quantize-dequantize (QDQ) on each Linear weight in
    every forward pass. For weight-only schemes the result is a fixed tensor, so writing
    it into the weight and switching QDQ off gives the same outputs at fp16 speed.
    Returns (modules baked, max |logit difference| on a probe prompt).
    """
    from compressed_tensors.quantization.lifecycle.forward import forward_quantize

    probe = tok("The capital of France is", return_tensors="pt").to(next(model.parameters()).device)
    before = model(**probe).logits.float()
    n = 0
    for m in model.modules():
        scheme = getattr(m, "quantization_scheme", None)
        if scheme is None or getattr(scheme, "weights", None) is None or not hasattr(m, "weight_scale"):
            continue
        if getattr(scheme, "input_activations", None) is not None:
            continue  # activation quantization cannot be baked into the weight
        m.weight.data = forward_quantize(m, m.weight.data, "weight", scheme.weights)
        m.quantization_enabled = False
        n += 1
    after = model(**probe).logits.float()
    return n, float((after - before).abs().max())


# Storage cost per quantized weight, in bits, including quantization metadata:
#   quanto qint4/qint2: group 128, fp16 scale + fp16 zero point -> +32/128
#   quanto qint8: per output channel fp16 scale -> about +0 bits per weight
#   GPTQ (symmetric, group 128): fp16 scale only -> +16/128
#   bitsandbytes NF4: block 64, fp32 absmax -> +32/64
OVERHEAD_BITS = {("quanto", 4): 0.25, ("quanto", 2): 0.25, ("quanto", 8): 0.0,
                 ("gptq", 3): 0.125, ("gptq", 4): 0.125, ("gptq", 8): 0.125, ("bnb4", 4): 0.5}
SKIP_SUFFIXES = ("_scale", "_zero_point", "_g_idx", "weight_shape")


def linear_size_gb(model, kind="fp16", bits=None, protect=()):
    """Weight size in GB, computed the same way for every HF configuration.

    Quantized decoder-layer Linear weights cost `bits` + metadata overhead; all
    other weights (embeddings, norms, protected layers) count at 16 bits.
    Quantization-metadata tensors themselves are skipped (already in the overhead).
    """
    protect = set(protect or ())
    extra = OVERHEAD_BITS.get((kind, bits), 0.0)
    total_bits = 0.0
    for name, p in model.named_parameters():
        if name.endswith(SKIP_SUFFIXES):
            continue
        n = p.numel()
        layer = _layer_index(name)
        quantized = (bits is not None and layer is not None and layer not in protect
                     and name.endswith(".weight") and p.dim() == 2)
        total_bits += n * ((bits + extra) if quantized else 16)
    return total_bits / 8 / 1e9


def size_gb_from_config(model_dir, kind="fp16", bits=None, protect=()):
    """Weight size in GB from the architecture in config.json, identical for every HF arm.

    (bitsandbytes and llm-compressor store weights differently in memory, so counting
    the loaded tensors would not be comparable across arms.)
    """
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    h, inter = cfg["hidden_size"], cfg["intermediate_size"]
    heads, kv = cfg["num_attention_heads"], cfg.get("num_key_value_heads", cfg["num_attention_heads"])
    hd = cfg.get("head_dim") or h // heads
    n_layers = cfg["num_hidden_layers"]
    per_layer = h * heads * hd * 2 + h * kv * hd * 2 + 3 * h * inter
    quant_bits = (bits + OVERHEAD_BITS.get((kind, bits), 0.0)) if bits else 16
    protect = set(protect or ())
    total = sum(per_layer * (16 if (l in protect or not bits) else quant_bits) for l in range(n_layers))
    embed = cfg["vocab_size"] * h * (1 if cfg.get("tie_word_embeddings", True) else 2)
    norms = (n_layers + 1) * h
    return (total + (embed + norms) * 16) / 8 / 1e9


def _layer_index(name):
    parts = name.split(".")
    for i, part in enumerate(parts[:-1]):
        if part == "layers" and parts[i + 1].isdigit():
            return int(parts[i + 1])
    return None


def gpu_model_gb():
    gc.collect()
    torch.cuda.empty_cache()
    return torch.cuda.memory_allocated() / 1e9


def clear_gpu():
    """Call after the caller has dropped its own references to the model."""
    gc.collect()
    torch.cuda.empty_cache()


def load_gguf(filename):
    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama

    path = hf_hub_download(C.GGUF_REPO, filename)
    llm = Llama(model_path=path, n_gpu_layers=-1, n_ctx=2048, seed=C.SEED, verbose=False)
    return llm, os.path.getsize(path) / 1e9
