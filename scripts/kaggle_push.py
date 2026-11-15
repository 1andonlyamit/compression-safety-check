"""Write a Kaggle script kernel for one run and push it.

Usage: python scripts/kaggle_push.py <slug> <run.json>
The aqs package source is embedded in the script, so the kernel needs no code dataset.
"""

import glob
import json
import os
import subprocess
import sys

USER = os.environ.get("KAGGLE_USER", "amitmakar")  # your Kaggle username
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
STUB = '''import os, sys
for root, dirs, files in os.walk("/kaggle/input"):
    depth = root.count(os.sep) - 2
    if depth <= 4:
        print(root, "|", ", ".join(sorted(files)[:6]))
    if depth >= 4:
        dirs[:] = []
SOURCES = {sources}
code_dir = "/kaggle/working/code"
os.makedirs(os.path.join(code_dir, "aqs"), exist_ok=True)
for name, text in SOURCES.items():
    with open(os.path.join(code_dir, "aqs", name), "w") as f:
        f.write(text)
sys.path.insert(0, code_dir)
from aqs.runner import main
RUN = {run}
RUN["code_dir"] = code_dir
main(RUN)
'''


def push(slug, run):
    sources = {os.path.basename(p): open(p, encoding="utf-8").read()
               for p in sorted(glob.glob(os.path.join(ROOT, "aqs", "*.py")))}
    folder = os.path.join(ROOT, "..", "kaggle", "kernels", slug)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "run.py"), "w", encoding="utf-8") as f:
        f.write(STUB.format(sources=repr(sources), run=repr(run)))
    meta = {
        "id": f"{USER}/{slug}", "title": slug, "code_file": "run.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [f"{USER}/aya-qs-prompts"],
        "kernel_sources": run.get("kernel_sources", []),
        "model_sources": [run.get("model_source", "cohere-labs/tiny-aya/transformers/fire/1")],
    }
    with open(os.path.join(folder, "kernel-metadata.json"), "w") as f:
        json.dump(meta, f, indent=1)
    # `python -m kaggle` instead of kaggle.exe: Windows Smart App Control may block the .exe launcher.
    r = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "push", "-p", folder],
                       capture_output=True, text=True)
    print(r.stdout.strip()[-500:], r.stderr.strip()[-500:])
    return r.returncode


if __name__ == "__main__":
    sys.exit(push(sys.argv[1], json.load(open(sys.argv[2]))))
