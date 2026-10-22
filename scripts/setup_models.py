"""Download the pretrained models the two apps use, at the exact versions of the first PC (SETUP.bat runs this).

  Grounding DINO base, SAM 2.1 large, OpenCLIP ViT-B-32 (laion2B), MiniLM-L6 -> <model_cache>/huggingface/hub
     (only the files the PC has; refs/main points to the pinned revision because the apps load them offline)
  DINOv2 ViT-L/14 weights                                                  -> <model_cache>/torch/hub/checkpoints
  (the DINOv2 code folder comes with the data download: cg2_model_dinov2_code.tar)
<model_cache> = CG2_MODEL_CACHE, default <CG2_ROOT>/model_cache (see .env.example). Models already there are skipped.
Needs internet once (~3.7 GB); afterwards both apps run offline.
"""
from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from glazeprep.config import MODEL_CACHE  # noqa: E402  (sets HF_HOME / TORCH_HOME before huggingface_hub is imported)

from huggingface_hub import snapshot_download  # noqa: E402

HUB = MODEL_CACHE / "huggingface" / "hub"
MODELS = {   # repo: (revision used on the first PC, files)
    "IDEA-Research/grounding-dino-base": ("12bdfa3120f3e7ec7b434d90674b3396eccf88eb",
        ["config.json", "model.safetensors", "preprocessor_config.json", "special_tokens_map.json", "tokenizer.json",
         "tokenizer_config.json", "vocab.txt"]),
    "facebook/sam2.1-hiera-large": ("665f8e2ad61cf5f53d65644ff27c8ee525124610",
        ["config.json", "model.safetensors", "preprocessor_config.json", "processor_config.json", "video_preprocessor_config.json"]),
    "laion/CLIP-ViT-B-32-laion2B-s34B-b79K": ("1a25a446712ba5ee05982a381eed697ef9b435cf", ["open_clip_model.safetensors"]),
    "sentence-transformers/all-MiniLM-L6-v2": ("1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        ["1_Pooling/config.json", "README.md", "config.json", "config_sentence_transformers.json", "model.safetensors",
         "modules.json", "sentence_bert_config.json", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt"]),
}
DINOV2 = ("https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth", 1217586395)


def have(repo: str, rev: str, files: list) -> bool:
    snap = HUB / ("models--" + repo.replace("/", "--")) / "snapshots" / rev
    return all((snap / f).exists() for f in files)


def main() -> int:
    print(f"Model folder: {MODEL_CACHE}", flush=True)
    for repo, (rev, files) in MODELS.items():
        if have(repo, rev, files):
            print(f"  ok       {repo}", flush=True)
            continue
        print(f"  download {repo} @ {rev[:8]} ...", flush=True)
        snapshot_download(repo, revision=rev, cache_dir=HUB, allow_patterns=files)
        ref = HUB / ("models--" + repo.replace("/", "--")) / "refs" / "main"
        ref.parent.mkdir(parents=True, exist_ok=True)
        ref.write_text(rev)
    ck = MODEL_CACHE / "torch" / "hub" / "checkpoints" / "dinov2_vitl14_pretrain.pth"
    if ck.exists() and ck.stat().st_size == DINOV2[1]:
        print("  ok       DINOv2 ViT-L/14 weights", flush=True)
    else:
        print("  download DINOv2 ViT-L/14 weights (1.2 GB) ...", flush=True)
        ck.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(str(ck) + ".part")
        urllib.request.urlretrieve(DINOV2[0], tmp)
        if tmp.stat().st_size != DINOV2[1]:
            print(f"ERROR: DINOv2 download is {tmp.stat().st_size} bytes, expected {DINOV2[1]} - run SETUP.bat again", flush=True)
            return 1
        os.replace(tmp, ck)
    print("Models ready.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
