"""Download and unpack the data of both apps (DOWNLOAD_DATA.bat runs this), then rebuild the Tile cut-outs.

The archives are listed in data_manifest.json (GitHub release data-v1 of nghialeduc268-a11y/nghia_ceramic): name,
URL, size, SHA-256 and where each one is unpacked:
  data     -> CG2_DATA       (default <CG2_ROOT>/new image data: photos, crops, previews, masks, review files)
  project  -> this project   (multimodal_experiments/work + cache: search index, trained models, recipe thumbnails)
  models   -> CG2_MODEL_CACHE (default <CG2_ROOT>/model_cache: the DINOv2 code folder)
Interrupted downloads continue where they stopped; every archive is checked (size + SHA-256) before unpacking;
an archive already unpacked (marker file .cg2_data/<archive>.done) is skipped. Existing files are never deleted.

usage: python scripts/download_data.py [--from-folder DIR] [--downloads DIR] [--keep-archives] [--only NAME ...]
  --from-folder DIR  use archives already downloaded by hand (browser, Google Drive copy, USB) instead of the internet
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from glazeprep.config import CG2_DATA, CG2_ROOT, MODEL_CACHE  # noqa: E402  (env / .env / defaults)

ROOTS = {"data": CG2_DATA, "project": CODE, "models": MODEL_CACHE}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for b in iter(lambda: fh.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def download(url: str, dst: Path, size: int, tries: int = 8) -> None:
    """HTTP download with resume (Range) and retries"""
    part = Path(str(dst) + ".part")
    for k in range(1, tries + 1):
        have = part.stat().st_size if part.exists() else 0
        if have == size:
            break
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                mode = "ab" if have and r.status == 206 else "wb"
                done, t0, last = (have if mode == "ab" else 0), time.time(), 0.0
                with part.open(mode) as fh:
                    while True:
                        b = r.read(4 << 20)
                        if not b:
                            break
                        fh.write(b)
                        done += len(b)
                        if time.time() - last > 5:
                            last = time.time()
                            print(f"    {done / 1e9:5.2f} / {size / 1e9:.2f} GB  ({done / max(1, size) * 100:4.1f} %)", flush=True)
        except Exception as exc:  # noqa: BLE001
            wait = min(300, 15 * k)
            print(f"    download interrupted ({type(exc).__name__}: {str(exc)[:120]}); retry {k}/{tries} in {wait} s", flush=True)
            time.sleep(wait)
    if not part.exists() or part.stat().st_size != size:
        raise SystemExit(f"could not download {url} - run DOWNLOAD_DATA.bat again (it continues where it stopped)")
    os.replace(part, dst)


def safe_extract(tar: Path, root: Path) -> int:
    """unpack only regular files and folders that stay inside root"""
    root = root.resolve()
    n = 0
    with tarfile.open(tar) as t:
        for m in t:
            target = (root / m.name).resolve()
            if root != target and root not in target.parents:
                raise SystemExit(f"{tar.name}: unsafe path {m.name!r}")
            if m.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif m.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                with t.extractfile(m) as src, open(target, "wb") as out:
                    while True:
                        b = src.read(8 << 20)
                        if not b:
                            break
                        out.write(b)
                os.utime(target, (m.mtime, m.mtime))
                n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-folder", type=Path, default=None)
    ap.add_argument("--downloads", type=Path, default=CG2_ROOT / "_downloads")
    ap.add_argument("--keep-archives", action="store_true", help="keep the .tar files after unpacking")
    ap.add_argument("--only", nargs="*", default=None, help="archive names to process (default: all)")
    ap.add_argument("--no-rebuild", action="store_true", help="skip rebuilding the Tile cut-outs")
    args = ap.parse_args()
    man = json.loads((CODE / "data_manifest.json").read_text(encoding="utf-8"))
    archives = [a for a in man["archives"] if not args.only or a["name"] in args.only]
    total = sum(a["size"] for a in archives)
    print(f"Data {man['dataset']} {man['version']}: {len(archives)} archives, {total / 1e9:.1f} GB", flush=True)
    for k, v in ROOTS.items():
        print(f"  {k:8s} -> {v}", flush=True)
    args.downloads.mkdir(parents=True, exist_ok=True)
    for i, a in enumerate(archives, 1):
        root = ROOTS[a["extract_to"]]
        marker = root / ".cg2_data" / f"{a['name']}.done"
        if marker.exists() and marker.read_text().strip() == a["sha256"]:
            print(f"[{i}/{len(archives)}] {a['name']}: already unpacked", flush=True)
            continue
        src = args.from_folder / a["name"] if args.from_folder else args.downloads / a["name"]
        if not (src.exists() and src.stat().st_size == a["size"]):
            if args.from_folder:
                raise SystemExit(f"{src} is missing or has the wrong size (expected {a['size']:,} bytes)")
            print(f"[{i}/{len(archives)}] downloading {a['name']} ({a['size'] / 1e9:.2f} GB) ...", flush=True)
            download(a["url"], src, a["size"])
        print(f"[{i}/{len(archives)}] checking {a['name']} ...", flush=True)
        if sha256(src) != a["sha256"]:
            if not args.from_folder:
                src.unlink()
            raise SystemExit(f"{a['name']}: SHA-256 does not match the manifest (damaged file removed) - run again")
        print(f"[{i}/{len(archives)}] unpacking {a['name']} -> {root} ...", flush=True)
        n = safe_extract(src, root)
        if n != a["files"]:
            raise SystemExit(f"{a['name']}: unpacked {n} files, the manifest says {a['files']}")
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(a["sha256"])
        if not args.from_folder and not args.keep_archives:
            src.unlink()
    if not args.no_rebuild:
        print("Rebuilding the Tile cut-outs (smaller_object_rgba_raw, ~12 GB, about 2 - 10 min) ...", flush=True)
        r = subprocess.run([sys.executable, str(CODE / "scripts" / "rebuild_tile_cutouts.py")])
        if r.returncode:
            return r.returncode
    print("Data ready. Next: CHECK_DATA.bat, then OPEN_REVIEW_APP.bat / RUN_GLAZE_SEARCH.bat", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
