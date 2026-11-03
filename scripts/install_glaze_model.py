"""Install (or update) Claire's glaze retrieval model for the glaze search page (INSTALL_GLAZE_MODEL.bat runs this).

Copies the three runtime files out of her ZIP (test_similarity.zip) into the model folder of configs/glaze_retrieval.json
(default multimodal_experiments/models/glaze_retrieval, not in git), after the same checks the app makes at start-up:
catalog columns, catalog and vectors in the same recipe order, checkpoint readable with weights_only=True (no code is run).
The old model stays in place when a check fails. Writes manifest.json (source, SHA-256 of every file, model version).
The ZIP's notebook, training_data.csv and glaze_chemistry_models.joblib are not needed by the app and are not copied.
usage: python scripts/install_glaze_model.py ["path\\to\\test_similarity.zip"]
       without a path: the newest test_similarity*.zip in the project folder or the folder above it
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "multimodal_experiments"))
from mm.glaze_retrieval import FILES, load_artifacts, model_version, read_config, resolve  # noqa: E402


def find_zip() -> Path | None:
    hits = [p for d in (CODE, CODE.parent) for p in d.glob("test_similarity*.zip")]
    return max(hits, key=lambda p: p.stat().st_mtime) if hits else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("zip", nargs="?", help="Claire's ZIP (default: newest test_similarity*.zip next to the project)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    src = Path(args.zip) if args.zip else find_zip()
    if src is None or not src.is_file():
        print("No ZIP found. Drag Claire's ZIP onto INSTALL_GLAZE_MODEL.bat, or give its path.")
        return 1
    dest = resolve(read_config()["model_dir"])
    print(f"ZIP:    {src}\nfolder: {dest}")
    with zipfile.ZipFile(src) as z:
        names = [n for n in z.namelist() if not n.startswith("__MACOSX/") and not n.endswith("/")]
        member = {}
        for key, fname in FILES.items():
            hits = [n for n in names if n.rsplit("/", 1)[-1] == fname]
            if len(hits) != 1:
                print(f"ERROR: the ZIP has {len(hits)} copies of {fname} (needs exactly one)")
                return 1
            member[key] = hits[0]
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix="glaze_model_", dir=dest.parent))
        try:
            for key, fname in FILES.items():                # read by name, written under our own file name (no paths from the ZIP)
                with z.open(member[key]) as fi, open(tmp / fname, "wb") as fo:
                    shutil.copyfileobj(fi, fo)
            catalog, vectors, ckpt = load_artifacts(tmp)
            ver = model_version(tmp)
            manifest = {"source_zip": src.name, "installed": time.strftime("%Y-%m-%d %H:%M:%S"), "model_version": ver,
                        "recipes": len(catalog), "vector_dim": int(vectors.shape[1]), "kmeans_clusters": int(catalog["kmeans_cluster"].nunique()),
                        "text_model": ckpt.get("model_name"), "files": {f: hashlib.sha256((tmp / f).read_bytes()).hexdigest() for f in FILES.values()}}
            (tmp / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
            if dest.exists():
                shutil.rmtree(dest)
            tmp.rename(dest)
        except Exception as exc:  # noqa: BLE001
            shutil.rmtree(tmp, ignore_errors=True)
            print(f"ERROR: {type(exc).__name__}: {exc}\nNothing was changed.")
            return 1
    print(f"Installed model {ver}: {manifest['recipes']:,} recipes, {manifest['vector_dim']}-d vectors, "
          f"{manifest['kmeans_clusters']} K-Means clusters.\nRestart RUN_GLAZE_SEARCH.bat to use it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
