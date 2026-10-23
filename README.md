# CG2 Glaze Search

The glaze search of the CG2 pottery project is built on **7,063 Glazy glaze recipes** whose photos were segmented into ceramic objects (Grounding DINO + SAM 2.1 + CLIP). Drop or paste a glaze photo: the app finds objects, describes colour regions and six attributes, and searches **12,309 objects** by shape, glaze/surface and colour.

This branch contains **only Glaze Search**, with original photos and segmentation data required by the search. The Review app is on branch [`nghia_mike_ceramic`](https://github.com/haohaozai/GC2-Ensemble-Model-Pottery/tree/nghia_mike_ceramic).

> **macOS status:** The `.command` launchers are converted from the provided Windows launchers, but have **not been verified on a Mac**. The data download step still needs a macOS-compatible downloader. Some ML dependencies may require additional Apple Silicon or Intel Mac adjustments.

---

## 1. Download the project from GitHub

**Use branch `nghia_mike_ceramic_search`, not `main`.**

Repository: [GC2-Ensemble-Model-Pottery](https://github.com/haohaozai/GC2-Ensemble-Model-Pottery/tree/nghia_mike_ceramic_search)

### Windows

Open CMD and choose a location:

```cmd
mkdir D:\CG2
cd /d D:\CG2
git clone -b nghia_mike_ceramic_search --single-branch https://github.com/haohaozai/GC2-Ensemble-Model-Pottery.git
cd GC2-Ensemble-Model-Pottery
```

### macOS

Open **Terminal** and run:

```bash
mkdir -p ~/CG2
cd ~/CG2
git clone -b nghia_mike_ceramic_search --single-branch https://github.com/haohaozai/GC2-Ensemble-Model-Pottery.git
cd GC2-Ensemble-Model-Pottery
```

Alternatively, select **Code → Download ZIP** on the branch page and extract the ZIP. The `.command` files must be present in the project root beside `requirements.txt`; if the repository does not yet contain them, copy the provided `.command` files there.

---

## 2. Installation — in order

### Windows — Step 1: `SETUP.bat`

Install [uv](https://docs.astral.sh/uv/) once in PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Open a new CMD window and double-click `SETUP.bat` (or run it in CMD). It creates a Python 3.11 `.venv`, installs PyTorch and pinned packages, and downloads pretrained models (~3.7 GB).

### macOS — Step 1: `SETUP.command`

Install [uv](https://docs.astral.sh/uv/) once in Terminal:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Close and reopen Terminal so `uv` is available. In Terminal, navigate to the repository root and grant permission **once**:

```bash
cd ~/CG2/GC2-Ensemble-Model-Pottery
chmod +x SETUP.command CHECK_DATA.command RUN_GLAZE_SEARCH.command
```

**Then double-click `SETUP.command` in Finder.** It opens Terminal and runs the setup. Keep Terminal open until it finishes; the launcher waits for Return before closing. You can also run `./SETUP.command` in Terminal.

If Finder reports that the file cannot be opened, verify the executable permission with `ls -l *.command`. If macOS blocks a downloaded script, inspect the script first and use **System Settings → Privacy & Security → Open Anyway** only if you trust it. Do not bypass security checks for unknown scripts.

**macOS notes:** The launcher uses `.venv/bin/python` and macOS-compatible PyTorch installation instead of Windows CUDA wheels. Apple Silicon/Intel dependencies have not been validated. Windows `.venv` directories cannot be copied to macOS; run setup on each computer.

### Windows — Step 2: `DOWNLOAD_DATA_ORIGINAL_AND_SEGMENTATION.bat`

Double-click to download **8 archives (~6.15 GB)** from [data-v2](https://github.com/nghialeduc268-a11y/nghia_ceramic/releases/tag/data-v2). The script checks sizes and SHA-256 hashes and unpacks data into the expected locations. It can resume interrupted downloads. See [docs/DATA_DOWNLOAD.md](docs/DATA_DOWNLOAD.md).

| Contents                                          | Archives                                                     | Size     |
| ------------------------------------------------- | ------------------------------------------------------------ | -------- |
| Original photos (7,063)                           | `cg2_photos_7k_part1of2.tar`, `cg2_photos_7k_part2of2.tar`   | 2.59 GB  |
| Segmented objects (11,930 crops)                  | `cg2_object_crops_part1of2.tar`, `cg2_object_crops_part2of2.tar` | 2.58 GB  |
| Segmentation labels, review records, boxes, masks | `cg2_segmentation_labels.tar`                                | 0.46 GB  |
| 125 extra photos, objects and labels              | `cg2_bonus_set.tar`                                          | 0.22 GB  |
| Search index, labels, models, thumbnails          | `cg2_search_artifacts.tar`                                   | 0.31 GB  |
| DINOv2 code                                       | `cg2_model_dinov2_code.tar`                                  | 0.004 GB |

### macOS — Step 2: data download (not yet automated)

**Do not double-click the Windows `.bat` downloader on macOS.** A macOS equivalent of `DOWNLOAD_DATA_ORIGINAL_AND_SEGMENTATION.bat` is not included in this package because its source was not provided for conversion.

You may download the eight archives from the [data-v2 release](https://github.com/nghialeduc268-a11y/nghia_ceramic/releases/tag/data-v2), but **do not guess the extraction destinations**: follow the data layout documented in `docs/DATA_DOWNLOAD.md` and the downloader source. Check the SHA-256 values against the project's manifest before extracting. For a fully automated macOS installation, convert and verify that downloader as `DOWNLOAD_DATA_ORIGINAL_AND_SEGMENTATION.command` first.

### Windows — Step 3: `CHECK_DATA.bat`

Double-click `CHECK_DATA.bat`. Each `OK` means the item was found; `NO` indicates missing data and how to fix it.

### macOS — Step 3: `CHECK_DATA.command`

Double-click `CHECK_DATA.command` in Finder after data is installed, or run:

```bash
./CHECK_DATA.command
```

All required checks should say `OK`. If checks fail, fix the missing data or models before launching the app.

---

## 3. Start Glaze Search

**Windows:** Double-click `RUN_GLAZE_SEARCH.bat`.

**macOS:** Double-click `RUN_GLAZE_SEARCH.command` in Finder, or run:

```bash
./RUN_GLAZE_SEARCH.command
```

The local page should be available at **http://127.0.0.1:8770** once the server starts. Keep the Terminal/CMD window open while using the app; press Ctrl+C in Terminal to stop it.

- Drop, paste (Ctrl+V on Windows, Command+V on Mac) or link a photo; objects are described by colour regions and six attributes (object form, surface finish, transparency, surface texture, glaze effects, possible defects).
- Search by shape, glaze/surface and colour using priority sliders, optional text and formula, and required-feature filters.
- Search metrics include Hit@K, Precision@K, Recall@K and nDCG.

Guide: [docs/SEARCH_GUIDE.md](docs/SEARCH_GUIDE.md).

Uploads and searches: `multimodal_experiments/work/queries/`; session log: `multimodal_experiments/work/queries/session_log.json`; app log: `multimodal_experiments/work/logs/app.log`.

### Search labels

| Label                                         | Source                                                       | Behavior                                        |
| --------------------------------------------- | ------------------------------------------------------------ | ----------------------------------------------- |
| Colour (three colours per object, name/hex/%) | `new image data/full_segmentation_claude_process/review/object_review.csv`, `full_auto_tags.csv`, `tag_json/` | Read at startup                                 |
| Six attributes                                | `multimodal_experiments/work/attributes_v3/` (plus v1/v2), `review/tile_review.json` | Human > Claude > Grounding DINO > earlier model |
| Review decision                               | `review/object_review.csv` (`review_status`)                 | Objects marked *not needed* excluded            |

---

## 4. System requirements

| Item             | Requirement                                                  |
| ---------------- | ------------------------------------------------------------ |
| Operating system | Windows 10/11; macOS launcher support is **experimental/unverified** |
| Disk space       | Approximately **20 GB** free; original project estimate ~15 GB retained, plus caches |
| Downloads        | Approximately **13 GB** once (6.15 GB data, ~3.7 GB models, ~3 GB packages); exact macOS package sizes may differ |
| GPU              | Windows: NVIDIA GPU with ≥6 GB VRAM recommended. CPU mode is slower. macOS: no CUDA; Apple Metal/MPS acceleration is **not verified** for this pipeline |
| Python           | 3.11, provisioned by `uv` through the setup launcher         |

---

## 5. Folder layout

```text
CG2/                                      # parent folder, any location
├── GC2-Ensemble-Model-Pottery/           # repository root
│   ├── SETUP.bat                          # Windows
│   ├── CHECK_DATA.bat
│   ├── RUN_GLAZE_SEARCH.bat
│   ├── DOWNLOAD_DATA_ORIGINAL_AND_SEGMENTATION.bat
│   ├── SETUP.command                      # macOS (add converted files)
│   ├── CHECK_DATA.command
│   ├── RUN_GLAZE_SEARCH.command
│   ├── multimodal_experiments/            # server, search engine, work, cache
│   ├── glazeprep/                         # shared paths, segmentation, CLIP
│   ├── reference_library/
│   ├── third_party/color_lab_v3/
│   ├── scripts/  configs/  docs/
│   └── data_manifest.json  requirements.txt  .env.example
├── new image data/                        # data downloader output
│   ├── 7k_matched_images/7k_matched_images/
│   ├── full_segmentation_claude_process/
│   ├── bonus_image_claude_process/
│   └── others_image/
└── model_cache/                           # pretrained models
```

**Data elsewhere?** Copy `.env.example` to `.env` and set `CG2_DATA`, `CG2_ROOT` and/or `CG2_MODEL_CACHE` as appropriate. Use absolute macOS paths such as `/Users/yourname/CG2/new image data` on Mac.

---

## 6. Online version

[Glaze Search on Hugging Face](https://huggingface.co/spaces/FBIWarninggg/glaze_search) (CPU, slower for new photos).

---

## 7. Troubleshooting

| Problem                                        | Fix                                                          |
| ---------------------------------------------- | ------------------------------------------------------------ |
| `git` not found                                | Install Git, reopen Terminal/CMD                             |
| `uv` not found                                 | Install `uv` using the platform-specific command above, reopen Terminal/CMD |
| macOS says permission denied                   | In repository root: `chmod +x *.command`                     |
| `.command` opens in editor instead of Terminal | Finder → Get Info → Open with → Terminal; ensure executable bit is set |
| macOS security warning                         | Inspect and trust the script before allowing it via Privacy & Security |
| Python environment missing                     | Run `SETUP.bat` (Windows) or `SETUP.command` (Mac)           |
| `CHECK_DATA` reports `NO`                      | Restore missing archives/models and verify `.env` paths      |
| Download stops or SHA-256 fails                | Windows: rerun downloader. Mac: use verified archives and follow data guide |
| Models missing / wants internet                | Rerun platform-specific SETUP launcher                       |
| CUDA out of memory (Windows)                   | Close other GPU jobs; CPU mode in CMD: `set CUDA_VISIBLE_DEVICES=-1` then `RUN_GLAZE_SEARCH.bat` |
| App already running                            | Open http://127.0.0.1:8770                                   |
| Port 8770 busy                                 | Windows: `RUN_GLAZE_SEARCH.bat --port 8771`; Mac: `./RUN_GLAZE_SEARCH.command --port 8771` |
| Mac dependency installation fails              | Check Apple Silicon/Intel architecture and Python package compatibility; macOS pipeline is unverified |

---

## 8. Manuals

PDF manuals with screenshots in [docs/manuals](docs/manuals): Glaze Search beginner guide, manual v4 (search section), flow mind map, and attribute report.

---

## 9. What is (not) on this branch

Included: Glaze Search server/page/search engine, shared modules, configuration, colour vocabulary, reference library, setup/download/check scripts, guides and manuals. Data comes from release data-v2; pretrained models from `scripts/setup_models.py` (pinned versions).

Not included: Review app and review-only data (see [`nghia_mike_ceramic`](https://github.com/haohaozai/GC2-Ensemble-Model-Pottery/tree/nghia_mike_ceramic) and data-v1), segmentation/training pipelines, Python environment, personal settings and uploaded photos.

**Windows quick start:** `SETUP.bat` → `DOWNLOAD_DATA_ORIGINAL_AND_SEGMENTATION.bat` → `CHECK_DATA.bat` → `RUN_GLAZE_SEARCH.bat`.

**macOS quick start (after data downloader is converted or data is correctly installed):** `SETUP.command` → data installation → `CHECK_DATA.command` → `RUN_GLAZE_SEARCH.command`.
