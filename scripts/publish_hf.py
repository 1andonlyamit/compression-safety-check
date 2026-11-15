"""Publish the project to Hugging Face (run only after review).

Usage:
  python scripts/publish_hf.py upload [--public]   # create repos (private by default) and upload
  python scripts/publish_hf.py public              # flip the repos and both collections to public
"""

import os
import sys

from huggingface_hub import HfApi

USER = "1andonlyamit"
SPACE = f"{USER}/compression-safety-check"
MODEL = f"{USER}/tiny-aya-fire-gptq-w4-wikitextcalib"
DATASET = f"{USER}/compression-safety-check-results"
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
STAGING = os.path.join(ROOT, "publish", "staging")
SPACE_DIR = os.path.join(ROOT, "repo", "webapp")  # static HTML/JS app (free to host)

api = HfApi()


def upload(private=True):
    api.create_repo(MODEL, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(repo_id=MODEL, repo_type="model", folder_path=os.path.join(STAGING, "model"),
                      commit_message="Add broken 4-bit Tiny Aya and 28 MB repair patch")
    api.create_repo(DATASET, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(repo_id=DATASET, repo_type="dataset", folder_path=os.path.join(STAGING, "dataset"),
                      commit_message="Add evaluation results, scores, patches and code")
    api.create_repo(SPACE, repo_type="space", space_sdk="static", private=private, exist_ok=True)
    api.upload_folder(repo_id=SPACE, repo_type="space", folder_path=SPACE_DIR,
                      ignore_patterns=["__pycache__/*", "*.pyc", "flagged/*"],
                      commit_message="Add interactive demo")
    # The hackathon asks for two public, un-gated collections: one for models, one for data.
    colls = {}
    for key, title, desc, items in [
        ("models", "Compression Safety Check: models and repair patch",
         "Lexsi AI Safety Hackathon (Track 2b): a broken 4-bit Tiny Aya Fire, its 28 MB weight-space "
         "repair patch, and the interactive Compression Safety Check.",
         [(MODEL, "model"), (SPACE, "space")]),
        ("data", "Compression Safety Check: safety evaluation data",
         "Lexsi AI Safety Hackathon (Track 2b): judged replies, summaries, neuron scores, patches and "
         "run logs for compressed Tiny Aya Fire and Earth.",
         [(DATASET, "dataset")]),
    ]:
        coll = api.create_collection(title=title, description=desc, private=private, exists_ok=True)
        for item, kind in items:
            try:
                api.add_collection_item(coll.slug, item_id=item, item_type=kind, exists_ok=True)
            except Exception as e:  # gated or missing items should not stop the rest
                print("collection item skipped:", item, e)
        colls[key] = coll.slug
    print("Space:     https://huggingface.co/spaces/" + SPACE)
    print("Model:     https://huggingface.co/" + MODEL)
    print("Dataset:   https://huggingface.co/datasets/" + DATASET)
    for key, slug in colls.items():
        print(f"Collection ({key}): https://huggingface.co/collections/" + slug)


def make_public():
    for rid, kind in [(MODEL, "model"), (DATASET, "dataset"), (SPACE, "space")]:
        api.update_repo_settings(rid, repo_type=kind, private=False)
        print("public:", rid)
    for c in api.list_collections(owner=USER, limit=50):
        if c.title.startswith(("Compression Safety Check", "Compressed Tiny Aya")):
            api.update_collection_metadata(c.slug, private=False)
            print("public: collection", c.slug)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "upload":
        upload(private="--public" not in sys.argv)
    elif cmd == "public":
        make_public()
