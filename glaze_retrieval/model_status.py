"""Check locally cached Hugging Face snapshots without downloading weights."""
from __future__ import annotations
from pathlib import Path
from glazeprep import config as C

MODELS = {
    "Grounding DINO": C.GROUNDING_DINO_ID,
    "SAM 2.1": C.SAM2_ID,
    "MiniLM retrieval encoder": "sentence-transformers/all-MiniLM-L6-v2",
}


def _snapshot_ready(repo_id: str) -> bool:
    """A complete snapshot must have a config and at least one weight file."""
    from huggingface_hub import scan_cache_dir
    roots = [Path(C.MODEL_CACHE) / "huggingface" / "hub", Path.home() / ".cache/huggingface/hub"]
    for root in roots:
        if not root.is_dir():
            continue
        try:
            for repo in scan_cache_dir(cache_dir=root).repos:
                if repo.repo_id != repo_id:
                    continue
                for revision in repo.revisions:
                    names = {f.file_name for f in revision.files}
                    if "config.json" in names and any(n.endswith((".safetensors", ".bin")) for n in names):
                        return True
        except Exception:
            continue
    return False


def statuses() -> dict[str, bool | None]:
    result = {label: _snapshot_ready(repo) for label, repo in MODELS.items()}
    # OpenCLIP uses its own checkpoint naming and may have been installed in an
    # alternate cache; do not mislabel it as installed without loading it.
    result["Ceramic CLIP (OpenCLIP)"] = None
    return result
