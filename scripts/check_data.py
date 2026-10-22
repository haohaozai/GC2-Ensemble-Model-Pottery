"""Check that everything Glaze Search needs is in place (CHECK_DATA.bat runs this).

Reads the "checks" of data_manifest.json (folder, file pattern, expected number of files) and the pretrained models of
scripts/setup_models.py, and prints what is missing and how to fix it.
Exit code 0 = all present, 1 = something is missing.
usage: python scripts/check_data.py [--quick]   (--quick: count files only in folders, skip the model files)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from glazeprep.config import CG2_DATA, MODEL_CACHE  # noqa: E402

ROOTS = {"data": CG2_DATA, "project": CODE, "models": MODEL_CACHE}


def count(folder: Path, pattern: str) -> int:
    return sum(1 for _ in folder.glob(pattern)) if folder.is_dir() else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    man = json.loads((CODE / "data_manifest.json").read_text(encoding="utf-8"))
    print(f"Checking {man['dataset']} {man['version']}")
    for k, v in ROOTS.items():
        print(f"  {k:8s} {v}")
    missing = []
    for c in man["checks"]:
        p = ROOTS[c["root"]] / c["path"]
        if c.get("pattern"):
            n = count(p, c["pattern"])
            ok = n >= c["count"]
            line = f"{n:>7,} / {c['count']:<7,} {c['root']}\\{c['path']}\\{c['pattern']}"
        else:
            ok = p.is_file()
            line = f"{'found' if ok else 'MISSING':>17} {c['root']}\\{c['path']}"
        print(f"  {'OK ' if ok else 'NO '} [{c['needed_by']:^6}] {line}")
        if not ok:
            missing.append((c, line))
    if not args.quick:
        sys.path.insert(0, str(CODE / "scripts"))
        import setup_models as SM
        for repo, (rev, files) in SM.MODELS.items():
            ok = SM.have(repo, rev, files)
            print(f"  {'OK ' if ok else 'NO '} [search] model {repo}")
            if not ok:
                missing.append(({"fix": "SETUP.bat"}, f"model {repo}"))
        ck = MODEL_CACHE / "torch" / "hub" / "checkpoints" / "dinov2_vitl14_pretrain.pth"
        ok = ck.exists() and ck.stat().st_size == SM.DINOV2[1]
        print(f"  {'OK ' if ok else 'NO '} [search] model DINOv2 ViT-L/14 weights")
        if not ok:
            missing.append(({"fix": "SETUP.bat"}, "DINOv2 weights"))
    py = CODE / ".venv" / "Scripts" / "python.exe"
    if not py.exists():
        missing.append(({"fix": "SETUP.bat"}, "Python environment .venv"))
    print()
    if not missing:
        print("All data and models are in place: start RUN_GLAZE_SEARCH.bat.")
        return 0
    print(f"MISSING: {len(missing)} item(s)")
    for c, line in missing:
        print(f"  - {line.strip()}  ->  fix: {c.get('fix', 'DOWNLOAD_DATA_ORIGINAL_AND_SEGMENTATION.bat')}")
    print("If your data is in another folder, copy .env.example to .env and set CG2_DATA / CG2_MODEL_CACHE.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
