"""Experiment constants. Every run reads these, so results stay comparable."""

SEED = 0

# Tiny Aya Fire: 36 decoder layers, hidden 2048, MLP 11008.
N_LAYERS = 36
MLP_WIDTH = 11008
PROTECT_K = 9  # 25% of layers kept at native precision in the mixed-precision arms

# Generation. Tiny Aya ends replies with <|END_RESPONSE|> (261001) and turns with
# <|END_OF_TURN_TOKEN|> (6); generation_config.json only lists <EOS_TOKEN> (3).
EOS_TOKEN_IDS = [3, 6, 261001]
GGUF_STOP = ["<|END_RESPONSE|>", "<|END_OF_TURN_TOKEN|>"]
MAX_NEW_TOKENS = 160
GEN_BATCH = 16

# Evaluation set sizes (sampled with SEED where a subset is used).
N_HARMBENCH = 200       # walledai/HarmBench "standard": all rows
N_ADVBENCH = 100        # walledai/AdvBench: random subset of 520
N_XSTEST_SAFE = 250     # walledai/XSTest safe prompts: all rows
N_ORBENCH_HARD = 150    # bench-llm/or-bench hard-1k: random subset

# Layer-selection contrast sets, disjoint from the evaluation prompts.
N_SELECT = 64

# Capability.
N_ARC = 300
PPL_SAMPLES = 40
PPL_SEQ_LEN = 512

GGUF_REPO = "CohereLabs/tiny-aya-fire-GGUF"
HARMBENCH_JUDGE = "cais/HarmBench-Mistral-7b-val-cls"
REFUSAL_JUDGE = "Qwen/Qwen2.5-7B-Instruct"
