# Compression Safety Check

**Is your compressed model still safe? Check it, and repair it with a 28 MB patch.**

Most people run small LLMs compressed to 8, 4 or 3 bits, but safety is tested only on the full-size model. This project measures how compression changes the safety of Cohere's **Tiny Aya**, and repairs a broken compressed model by editing its weights directly, with no retraining.

**[Try the interactive tool](https://huggingface.co/spaces/1andonlyamit/compression-safety-check)** · **[Read the full write-up](WRITEUP.md)**

## Key results

**1. Some compressed versions quietly lose safety, and the usual quality check misses it.**

| Tiny Aya Fire | Size | Helps with harmful requests | Refuses harmful requests | Perplexity |
|---|---|---|---|---|
| Original | 6.7 GB | 2.0% | 97.7% | 18.5 |
| 8-bit, or 4-bit NF4 | 2.7–3.9 GB | ~2% (no change) | ~97% | 18.5 |
| Official GGUF Q4_0 | 2.0 GB | 4.0% | 95.3% | – |
| 4-bit GPTQ, plain-text calibration | 2.5 GB | **13.0%** | **56.3%** | **20.8 (looks normal)** |

**2. A 28 MB patch repairs the broken model.** We write the original weights back into the 0.6% of neurons that compression damaged most:

| | Helps with harmful | Refuses harmful | Nonsense replies | Quiz score (ARC) |
|---|---|---|---|---|
| Broken 4-bit model | 15.0% | 49.0% | 42.7% | 31.0% |
| **+ 28 MB repair patch** | **4.0%** | **95.7%** | **1.0%** | **70.7%** |

**3. "Safety neurons" are not where the damage is.** Patches of the same size on safety neurons, or on random neurons, do nothing.

**4. It replicates on a second model (Tiny Aya Earth) at 4-bit.** At 3-bit the patch brings back coherent replies but not safety, so always re-check safety after a repair.

![Safety vs size](figures/headline_harmful_compliance.png)

## Use the repair patch

The broken model and the patch are on [Hugging Face](https://huggingface.co/1andonlyamit/tiny-aya-fire-gptq-w4-wikitextcalib).

```python
from transformers import AutoModelForCausalLM
from transformers.utils.quantization_config import CompressedTensorsConfig
from patch_tool.apply_patch import apply_patch, remove_patch

model = AutoModelForCausalLM.from_pretrained(
    "1andonlyamit/tiny-aya-fire-gptq-w4-wikitextcalib", device_map="cuda", dtype="float16",
    quantization_config=CompressedTensorsConfig(run_compressed=False))
undo = apply_patch(model, "repair_patch.safetensors")   # repaired
remove_patch(model, undo)                               # back to broken
```

## How it works

1. **Compress** Tiny Aya 23 ways: GPTQ 8/4/3-bit, bitsandbytes NF4, and the official GGUF files.
2. **Test** each version on 700 prompts: 300 harmful (it should refuse) and 400 harmless but scary-sounding (it should answer). Two judge models label every reply, and paired statistics separate real changes from noise.
3. **Repair** by restoring the original fp16 weights of the most damaged neurons, and compare against patches of the same size on safety neurons and random neurons.

Built with Lexsi Labs' **SafeTune** (safety neurons, refusal direction), **CircuitKIT** (GPTQ compression) and **AuditKIT** (capability evaluation).

## Repository

| Folder | What |
|---|---|
| `aqs/` | The pipeline: compression, generation, judges, patches, analysis |
| `patch_tool/` | Apply or remove a repair patch |
| `webapp/` | The interactive tool (static HTML/JS) |
| `scripts/`, `runs/` | Kaggle run scripts and the exact run settings |

**Reproduce:** GPU runs use Kaggle (2× T4); analysis runs on a laptop.

```bash
pip install -r requirements.txt
export KAGGLE_USER=<your-kaggle-username>
python scripts/kaggle_push.py aya-qs-main runs/main2.json
bash scripts/kaggle_wait.sh aya-qs-main results/main
python -m aqs.analysis results/main reports/main
```

The write-up's [reproduction section](WRITEUP.md#documentation-and-reproduction) has the details.

## Links

- Interactive tool: [huggingface.co/spaces/1andonlyamit/compression-safety-check](https://huggingface.co/spaces/1andonlyamit/compression-safety-check)
- Model + repair patch: [huggingface.co/1andonlyamit/tiny-aya-fire-gptq-w4-wikitextcalib](https://huggingface.co/1andonlyamit/tiny-aya-fire-gptq-w4-wikitextcalib)
- Results data: [huggingface.co/datasets/1andonlyamit/compression-safety-check-results](https://huggingface.co/datasets/1andonlyamit/compression-safety-check-results)

## Licence

Code: MIT. Models, patches and model outputs derived from Tiny Aya: CC-BY-NC-4.0 (non-commercial).

Made by Amit Suthar for the Lexsi Labs AI Safety and Alignment Hackathon (Track 2b).
