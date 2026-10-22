"""Shared paths, cache redirection, logging, disk / GPU guards and small IO helpers.

Import this module before torch / transformers: it points every model cache (Hugging Face, torch hub, pip,
TEMP) at D: through glazeprep.config, so nothing large is written to C:.
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

CODE = Path(__file__).resolve().parents[2]                       # ...\code_project
EXP = CODE / "multimodal_experiments"
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(EXP / "vendor" / "gc2_text_branch"))
import glazeprep.config  # noqa: E402,F401  (HF_HOME, TORCH_HOME, TEMP -> <CG2_ROOT>\model_cache)

os.environ.setdefault("PIP_CACHE_DIR", str(Path(os.environ["HF_HOME"]).parent / "pip"))

# folder that holds the project and the image data (env CG2_ROOT / CG2_DATA or .env; see glazeprep/config.py)
CG2 = glazeprep.config.CG2_ROOT
BASE = glazeprep.config.CG2_DATA
SEG = BASE / "full_segmentation_claude_process"                 # existing segmentation (read-only here)
PHOTOS = BASE / "7k_matched_images" / "7k_matched_images"
CSV7K = BASE / "csv 7k data" / "7k_matched_all_data.csv"
ING7K = BASE / "csv 7k data" / "7k_matched_ingredients.csv"
REVIEW = SEG / "review" / "object_review.csv"                   # master review record (after round 2)
GLAZY_EXPORT = Path(os.environ.get("CG2_GLAZY_EXPORT", r"D:\Downloads\glazy_20260531.csv"))   # read-only
CACHE = EXP / "cache"                                           # features shared by smoke and full runs
NOT_NEEDED = "does_not_needed"
MIN_FREE_D_GB = 20.0
MIN_FREE_C_GB = 5.0
SEED = 20261017


def work_dir(smoke: bool) -> Path:
    return EXP / ("work_smoke" if smoke else "work")


def setup_logging(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("mm")
    if not log.handlers:
        log.setLevel(logging.INFO)
        log.propagate = False            # some libraries add a root handler; avoid printing every line twice
        fmt = logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S")
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        fh = logging.FileHandler(path, encoding="utf-8")
        fh.setFormatter(fmt)
        log.addHandler(sh)
        log.addHandler(fh)
    return log


def log() -> logging.Logger:
    return logging.getLogger("mm")


def free_gb(drive: str) -> float:
    return shutil.disk_usage(drive).free / 1e9


class DiskGuardError(RuntimeError):
    pass


def disk_guard(where: str = "") -> dict:
    d, c = free_gb(CG2.anchor or str(CG2)), free_gb(os.environ.get("SystemDrive", "C:") + "\\")   # data drive, system drive
    if d < MIN_FREE_D_GB or c < MIN_FREE_C_GB:
        raise DiskGuardError(f"stopping {where}: free space D: {d:.1f} GB (min {MIN_FREE_D_GB}), "
                             f"C: {c:.1f} GB (min {MIN_FREE_C_GB})")
    return {"free_D_GB": round(d, 1), "free_C_GB": round(c, 1)}


def dir_size_gb(p: Path) -> float:
    return sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file()) / 1e9 if Path(p).exists() else 0.0


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(p: Path) -> list[dict]:
    with open(p, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(p: Path, rows: list[dict], cols: list[str] | None = None):
    p.parent.mkdir(parents=True, exist_ok=True)
    cols = cols or (list(rows[0].keys()) if rows else [])
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, p)


def write_json(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(tmp, p)


def read_json(p: Path, default=None):
    return json.loads(Path(p).read_text(encoding="utf-8")) if Path(p).exists() else default


def obj_file(sub: str, name: str, root: Path = SEG) -> Path:
    p = root / sub / name
    return p if p.exists() else root / NOT_NEEDED / sub / name


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def __call__(self) -> float:
        return round(time.time() - self.t0, 1)


def gpu_peak_reset():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:  # noqa: BLE001
        pass


def gpu_peak_gb() -> float:
    try:
        import torch
        return round(torch.cuda.max_memory_allocated() / 1e9, 3) if torch.cuda.is_available() else 0.0
    except Exception:  # noqa: BLE001
        return 0.0


def other_gpu_jobs() -> list[str]:
    """Python processes of the segmentation pipeline that would compete for the GPU (never run two GPU jobs)."""
    try:
        import subprocess
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | % CommandLine"],
                             capture_output=True, text=True, timeout=60).stdout
        keys = ("10_claude_ceramic_pipeline", "19_manual_corrections", "16_retag_attributes")
        return [ln.strip()[-90:] for ln in out.splitlines() if any(k in ln for k in keys)]
    except Exception:  # noqa: BLE001
        return []
