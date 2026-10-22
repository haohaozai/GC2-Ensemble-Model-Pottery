# CG2 Glaze Search

The glaze search of the CG2 pottery project, built on **7,063 Glazy glaze recipes** whose photos were segmented into
ceramic objects (Grounding DINO + SAM 2.1 + CLIP). Drop or paste a photo of a glaze: the app finds the objects,
describes them (colour regions + six attributes) and searches **12,309 objects** by shape, glaze / surface and colour.

This branch contains **only Glaze Search** and only the data it needs: the **original photos** and the
**segmentation** (object crops, colour and attribute labels, masks). The review of the objects is finished and its
results are included. The Review app is on branch
[`nghia_mike_ceramic`](https://github.com/haohaozai/GC2-Ensemble-Model-Pottery/tree/nghia_mike_ceramic).

---

## 1. Download the project from GitHub

**Important: use the `nghia_mike_ceramic_search` branch, not `main`.**

Repository: [GC2-Ensemble-Model-Pottery](https://github.com/haohaozai/GC2-Ensemble-Model-Pottery/tree/nghia_mike_ceramic_search)

### Step 1 — Choose a folder

Any drive works, for example `D:\CG2` or `C:\CG2`. A short folder path is recommended. Open **Windows CMD**:

```cmd
mkdir D:\CG2
cd /d D:\CG2
```

### Step 2 — Clone the branch

```cmd
git clone -b nghia_mike_ceramic_search --single-branch https://github.com/haohaozai/GC2-Ensemble-Model-Pottery.git
cd GC2-Ensemble-Model-Pottery
```

**Alternative:** open the branch link above, click **Code → Download ZIP** and extract it into your folder.

---

## 2. Installation — 3 steps, in this order

### STEP 1 — `SETUP.bat`

Double-click `SETUP.bat` (or run it in CMD). It creates the Python environment (`.venv`, Python 3.11), installs
PyTorch (GPU or CPU build) and the pinned packages, and downloads the pretrained models (~3.7 GB, fixed versions).

**Before running:** install [uv](https://docs.astral.sh/uv/) once, in PowerShell, then open a new CMD window:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### STEP 2 — `DOWNLOAD_DATA_ORIGINAL_AND_SEGMENTATION.bat`

Double-click it. It downloads **8 archives (6.15 GB)** from the release
[data-v2](https://github.com/nghialeduc268-a11y/nghia_ceramic/releases/tag/data-v2), checks each one (size + SHA-256)
and unpacks it into the right folder:

| what | archives | size |
|---|---|---|
| original photos (7,063) | `cg2_photos_7k_part1of2.tar`, `cg2_photos_7k_part2of2.tar` | 2.59 GB |
| segmented objects (11,930 crops) | `cg2_object_crops_part1of2.tar`, `cg2_object_crops_part2of2.tar` | 2.58 GB |
| segmentation labels: colours, attributes, review records, boxes, masks | `cg2_segmentation_labels.tar` | 0.46 GB |
| 125 extra photos with their objects and labels | `cg2_bonus_set.tar` | 0.22 GB |
| search index, attribute labels and models, recipe thumbnails | `cg2_search_artifacts.tar` | 0.31 GB |
| DINOv2 code | `cg2_model_dinov2_code.tar` | 0.004 GB |

Interrupted? Run it again: it continues and skips what is finished. Details:
[docs/DATA_DOWNLOAD.md](docs/DATA_DOWNLOAD.md).

### STEP 3 — `CHECK_DATA.bat`

Every line should show `OK`. A line with `NO` says what is missing and which step fixes it.

---

## 3. Start Glaze Search

Double-click **`RUN_GLAZE_SEARCH.bat`**; the page opens at http://127.0.0.1:8770. Keep the black window open while
you use the page; close it to stop.

* Drop, paste (Ctrl+V) or link a photo: the objects are found and described (colour regions + six attributes:
  object form, surface finish, transparency, surface texture, glaze effects, possible defects).
* Search the database by shape, glaze / surface and colour with priority sliders, optionally with text and a glaze
  formula; tick the features that must match.
* Every search is evaluated (Hit@K, Precision@K, Recall@K, nDCG).

Guide: [docs/SEARCH_GUIDE.md](docs/SEARCH_GUIDE.md). Uploads and searches are saved in
`multimodal_experiments\work\queries\` (log: `work\queries\session_log.json`, `work\logs\app.log`).

### The labels the search uses

| label | file | |
|---|---|---|
| colour (3 main colours per object: name, hex, %) | `new image data\full_segmentation_claude_process\review\object_review.csv` (same values in `full_auto_tags.csv` and `tag_json\`) | read at every start |
| six attributes | `multimodal_experiments\work\attributes_v3\…` (+ v1, v2); human-checked: `review\tile_review.json` | best label per object: human > Claude > Grounding DINO > earlier model |
| review decision | `review\object_review.csv` (`review_status`) | objects marked *not needed* are not searched |

---

## 4. System requirements

| | |
|---|---|
| operating system | Windows 10 or 11 |
| free disk | about **20 GB**; about 15 GB stay (measured: original photos + segmentation 5.6 GB, search files 0.3 GB, models 3.8 GB, Python environment 5.0 GB); while downloading, one archive (≤ 1.3 GB) is kept until it is unpacked, plus the package cache of `uv` |
| download | about **13 GB** once: data 6.15 GB, models 3.7 GB, PyTorch and packages ~3 GB (e.g. ~22 min at 10 MB/s, ~1.8 h at 2 MB/s); afterwards everything runs offline |
| GPU | NVIDIA GPU with ≥ 6 GB memory recommended (the app uses ~4 GB). Without one it runs on the CPU: a new photo takes 1 - 3 minutes, searching stays fast |
| Python | 3.11, installed by `SETUP.bat` through uv |

---

## 5. Folder layout

```text
D:\CG2\                               <- any folder (the "CG2 root")
├── GC2-Ensemble-Model-Pottery\       <- this branch (the folder name does not matter)
│   ├── SETUP.bat  DOWNLOAD_DATA_ORIGINAL_AND_SEGMENTATION.bat  CHECK_DATA.bat  RUN_GLAZE_SEARCH.bat
│   ├── multimodal_experiments\       Glaze Search: app\ (server + page), mm\ (search engine), vendor\ (text branch)
│   │                                 work\ + cache\ = search index and trained files (from the download)
│   ├── glazeprep\                    shared code: paths, segmentation, CLIP
│   ├── reference_library\            example images + definitions behind the (i) buttons
│   ├── third_party\color_lab_v3\     colour vocabulary (Color Search Lab V3)
│   ├── scripts\  configs\  docs\
│   └── data_manifest.json  requirements.txt  .env.example
├── new image data\                   <- made by DOWNLOAD_DATA_ORIGINAL_AND_SEGMENTATION.bat
│   ├── 7k_matched_images\7k_matched_images\    original photos
│   ├── full_segmentation_claude_process\       objects (smaller_object), labels, masks, review\
│   ├── bonus_image_claude_process\             the 125 extra photos' objects, same structure
│   └── others_image\                           their original photos
└── model_cache\                      <- made by SETUP.bat (pretrained models)
```

**Data somewhere else?** Copy `.env.example` to `.env` and set `CG2_DATA` (and / or `CG2_ROOT`, `CG2_MODEL_CACHE`).
Nothing else needs to be edited.

---

## 6. Online version

[Glaze Search on Hugging Face](https://huggingface.co/spaces/FBIWarninggg/glaze_search) (CPU, slower for new photos).

---

## 7. Troubleshooting

| problem | fix |
|---|---|
| `git` is not recognized | install Git for Windows, open a new CMD |
| `uv was not found` | install uv (command above), open a **new** window, run `SETUP.bat` again |
| `ERROR: Python environment not found` | run `SETUP.bat` |
| a `CHECK_DATA.bat` line says `NO` | run `DOWNLOAD_DATA_ORIGINAL_AND_SEGMENTATION.bat` again (it only fetches what is missing); data in another folder → `.env` |
| download stops / SHA-256 does not match | run `DOWNLOAD_DATA_ORIGINAL_AND_SEGMENTATION.bat` again (continues, damaged files are fetched again) |
| GitHub download is slow or blocked | download the 8 `.tar` files in the browser from the release page, then `DOWNLOAD_DATA_ORIGINAL_AND_SEGMENTATION.bat --from-folder "<folder>"` |
| model not found / wants internet | run `SETUP.bat` again (models missing) |
| `CUDA out of memory` or `Another GPU job is running` | close other GPU programs; or CPU mode: `set CUDA_VISIBLE_DEVICES=-1`, then `RUN_GLAZE_SEARCH.bat` in the same window |
| `already running` | the page is open already: http://127.0.0.1:8770 |
| port in use | `RUN_GLAZE_SEARCH.bat --port 8771` |

---

## 8. Manuals

PDF manuals with screenshots in [docs/manuals](docs/manuals): Glaze Search beginner guide, manual v4 (its search
part), flow mind map, attribute report.

---

## 9. What is (not) on this branch

Included: everything Glaze Search runs (server, page, search engine, shared modules, configuration, colour vocabulary,
reference library), the setup / download / check scripts, guides and manuals. The data comes from release data-v2,
the models from their original publishers (`scripts/setup_models.py`, pinned versions).

Not included: the Review app and the data only it reads (previews, rejected objects, Tile cut-outs) — see branch
[`nghia_mike_ceramic`](https://github.com/haohaozai/GC2-Ensemble-Model-Pottery/tree/nghia_mike_ceramic) and release
data-v1; the segmentation / training pipelines; the Python environment; personal settings and uploaded photos.

**Quick start: SETUP → DOWNLOAD_DATA_ORIGINAL_AND_SEGMENTATION → CHECK_DATA → RUN_GLAZE_SEARCH.**
