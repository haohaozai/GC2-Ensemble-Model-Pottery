"""Glaze Search launcher in Python: the same steps as the .bat (Windows) and .command (macOS) files, for when those cannot
be started (blocked scripts, no double-click, Linux, or an IDE). Needs only a standard Python 3.8+ to start.

    python glaze_search.py                 do what is still missing, then start the app and open http://127.0.0.1:8770
    python glaze_search.py run             only start the app and open the page        (= RUN_GLAZE_SEARCH)
    python glaze_search.py setup           Python 3.11 environment + packages + models   (= SETUP)
    python glaze_search.py download [--from-folder FOLDER]   photos and segmentation data (= DOWNLOAD_DATA_ORIGINAL_AND_SEGMENTATION)
    python glaze_search.py check           check data and models                         (= CHECK_DATA)
    python glaze_search.py install-model [model.zip]   colour match + smart search models    (= INSTALL_GLAZE_MODEL)
Options: --port 8770 (another port if 8770 is busy) · --no-browser (do not open the page).
On macOS and Linux use python3. Stop the app with Ctrl+C in this window."""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WIN = os.name == "nt"
VENV_PY = ROOT / ".venv" / ("Scripts/python.exe" if WIN else "bin/python")
MODELS = ROOT / "multimodal_experiments" / "models"
GLAZE_FILES = [MODELS / "glaze_retrieval" / f for f in ("glaze_retrieval_catalog.csv", "glaze_retrieval_vectors.npz", "shared_space_heads.pt")]
JOINT_FILES = [MODELS / "glaze_joint" / f for f in ("heads.pt", "gallery.npz", "meta.json")]
ENV = dict(os.environ, PYTHONIOENCODING="utf-8")
ENV.setdefault("UV_CACHE_DIR", str(ROOT.parent / "model_cache" / "uv"))


def say(msg: str) -> None:
    print(f"\n=== {msg}", flush=True)


def run(cmd: list, cwd: Path = ROOT) -> int:
    print("  >", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.call([str(c) for c in cmd], cwd=str(cwd), env=ENV)


def uv() -> list:
    """the uv command: on PATH, in the places the uv installer and Homebrew use, or installed with this Python's pip"""
    home = Path.home()
    for c in [shutil.which("uv"), home / ".local/bin/uv", home / ".cargo/bin/uv", Path("/opt/homebrew/bin/uv"),
              home / ".local/bin/uv.exe", home / ".cargo/bin/uv.exe"]:
        if c and Path(c).is_file():
            return [str(c)]
    print("  uv (the installer this project uses) was not found: installing it with pip for this Python ...", flush=True)
    if subprocess.call([sys.executable, "-m", "pip", "install", "--user", "uv"]) == 0:
        return [sys.executable, "-m", "uv"]
    sys.exit("uv could not be installed. Install it once, then run this file again:\n"
             '  Windows (PowerShell): powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"\n'
             "  macOS / Linux:        curl -LsSf https://astral.sh/uv/install.sh | sh")


def need_venv() -> None:
    if not VENV_PY.is_file():
        sys.exit("The Python environment .venv is missing: run  python glaze_search.py setup  first.")


def step_setup() -> int:
    say("SETUP: Python 3.11 environment, packages, pretrained models")
    if platform.system() == "Darwin" and platform.machine() != "arm64":
        sys.exit("This Mac has an Intel processor: PyTorch 2.11 needs Apple Silicon (M1 or newer). Use the online version instead.")
    u = uv()
    if not VENV_PY.is_file() and run(u + ["venv", ".venv", "--python", "3.11"]):
        return 1
    torch = ["torch==2.11.0", "torchvision==0.26.0"]
    if platform.system() != "Darwin":                                   # same index as SETUP.bat: CUDA when an NVIDIA GPU is there
        torch += ["--index-url", "https://download.pytorch.org/whl/cu128" if shutil.which("nvidia-smi") else "https://download.pytorch.org/whl/cpu"]
    if run(u + ["pip", "install", "--python", VENV_PY] + torch) or run(u + ["pip", "install", "--python", VENV_PY, "-r", "requirements.txt"]):
        return 1
    return run([VENV_PY, "-u", "scripts/setup_models.py"])


def step_download(extra: list | None = None) -> int:
    need_venv()
    say("DOWNLOAD: original photos and segmentation data (continues where it stopped)")
    return run([VENV_PY, "-u", "scripts/download_data.py"] + (extra or []))


def step_check() -> int:
    need_venv()
    say("CHECK: data and models")
    return run([VENV_PY, "-u", "scripts/check_data.py"])


def step_install_model(zip_path: str | None = None) -> int:
    need_venv()
    say("INSTALL MODELS: colour match (from the model ZIP) and smart search (trained on this PC)")
    rc = run([VENV_PY, "-u", "scripts/install_glaze_model.py"] + ([zip_path] if zip_path else []))
    return run([VENV_PY, "-u", "scripts/train_glaze_joint.py"]) or rc


def answers(url: str) -> bool:
    try:
        with urllib.request.urlopen(url + "api/glaze/options", timeout=3) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def step_run(port: int, browser: bool) -> int:
    need_venv()
    url = f"http://127.0.0.1:{port}/"
    if answers(url):
        print(f"\nGlaze Search is already running: {url}", flush=True)
        if browser:
            webbrowser.open(url)
        return 0
    say(f"RUN: starting Glaze Search on {url} (the first start loads the models: about 30-60 s)")
    srv = subprocess.Popen([str(VENV_PY), "-u", "app/server.py", "--port", str(port), "--no-browser"],
                           cwd=str(ROOT / "multimodal_experiments"), env=ENV)
    t0 = time.time()
    while srv.poll() is None and not answers(url):
        if time.time() - t0 > 1800:
            print("The app did not answer within 30 minutes: see multimodal_experiments/work/logs/app.log", flush=True)
            break
        time.sleep(2)
    if srv.poll() is not None:
        print(f"The app stopped (exit code {srv.returncode}): see multimodal_experiments/work/logs/app.log", flush=True)
        return srv.returncode or 1
    print(f"\nGlaze Search is ready: {url}\nKeep this window open while you use the page; Ctrl+C stops the app.", flush=True)
    if browser:
        webbrowser.open(url)
    try:
        return srv.wait()
    except KeyboardInterrupt:
        srv.terminate()
        return 0


def step_all(port: int, browser: bool) -> int:
    if not VENV_PY.is_file() and step_setup():
        return 1
    if step_check():                     # something is missing: pretrained models, then the data (both skip what is there)
        say("Getting what is missing")
        run([VENV_PY, "-u", "scripts/setup_models.py"])
        step_download()
        if step_check():
            print("\nSome data or models are still missing (the NO lines above); the app starts with what is there.", flush=True)
    if not all(f.is_file() for f in GLAZE_FILES + JOINT_FILES):
        if step_install_model():
            print("\nThe glaze recipe models are not installed; the app still starts (Glaze recipes needs them).", flush=True)
    return step_run(port, browser)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("step", nargs="?", default="all", choices=["all", "run", "setup", "download", "check", "install-model"])
    ap.add_argument("zip", nargs="?", help="install-model: the model ZIP (default: the newest test_similarity*.zip "
                                           "in this folder, the folder above it or Downloads)")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--no-browser", action="store_true")
    a, extra = ap.parse_known_args()                # download: other options go to download_data.py (e.g. --from-folder)
    if extra and a.step != "download":
        ap.error(f"unknown options: {' '.join(extra)}")
    os.chdir(ROOT)
    steps = {"all": lambda: step_all(a.port, not a.no_browser), "run": lambda: step_run(a.port, not a.no_browser),
             "setup": step_setup, "download": lambda: step_download(extra), "check": step_check,
             "install-model": lambda: step_install_model(a.zip)}
    return steps[a.step]()


if __name__ == "__main__":
    sys.exit(main())
