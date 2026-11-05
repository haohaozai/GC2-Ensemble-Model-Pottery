# CG2 Glaze Search

The glaze search of the CG2 pottery project is built on **7,063 Glazy glaze recipes** whose photos were segmented into ceramic objects (Grounding DINO + SAM 2.1 + CLIP). Drop or paste a glaze photo: the app finds objects, describes colour regions and six attributes, and searches **12,309 objects** by shape, glaze/surface and colour.

This branch contains **only Glaze Search**, with original photos and segmentation data required by the search. The Review app is on branch [`nghia_mike_ceramic`](https://github.com/haohaozai/GC2-Ensemble-Model-Pottery/tree/nghia_mike_ceramic).

> **macOS status:** The `.command` launchers are converted from the provided Windows launchers, but have **not been verified on a Mac**. The data download step still needs a macOS-compatible downloader. Some ML dependencies may require additional Apple Silicon or Intel Mac adjustments.

### Glaze Search documents at a glance

| You want to see | Open | What is in it |
|---|---|---|
| The full process of a search | [Process, models and parameters (PDF, 8 pages)](docs/manuals/CG2_Glaze_Search_Process_Models_MindMap.pdf) · [mind map page](https://claude.ai/artifact/B1zswhT3kHGYnZYYvYgaa5) | Data → preprocessing → training → evaluation, and the 10 steps of a search with their settings |
| Which pretrained model does what, with diagrams | [Pretrained models, illustrated (PDF, 16 pages)](docs/manuals/CG2_Glaze_Search_Pretrained_Models.pdf) | Grounding DINO, SAM 2.1, OpenCLIP, DINOv2 and MiniLM: zero-shot, frozen or with a trained head; an activity diagram and 13 figures from one real photo and sentence |
| Metrics and evaluation results | [Evaluation (PDF, 14 pages)](docs/manuals/CG2_Glaze_Search_Evaluation.pdf) · [evaluation page](https://claude.ai/artifact/BVYzi6cWhbLpf6x1N8qQTh) | Every metric (R@K, P@10, MAD@K, accuracy, AUROC …) stage by stage, from the data and self-made labels to the final ranking, and the limits to state |
| How to use the app | [Manual (HTML)](docs/manuals/CG2_Glaze_Search_Manual.html) / [PDF](docs/manuals/CG2_Glaze_Search_Manual.pdf) · [manual page](https://claude.ai/artifact/7rmiKzN17dk2Fhb4AGeJhz) · [step-by-step guide (PDF)](docs/manuals/CG2_Glaze_Search_Step_by_Step_Guide.pdf) | The pages of the app and six worked examples (photo, words with typos, a formula) |
| Try it without installing | [Glaze Search on Hugging Face](https://huggingface.co/spaces/FBIWarninggg/glaze_search) | The same page and models, on Hugging Face's free GPU |

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

### STEP 4 — Install the glaze recipe models (for card G of Glaze Search)

Drag the model ZIP (`test_similarity.zip`) onto `INSTALL_GLAZE_MODEL.bat`, or run:

```cmd
INSTALL_GLAZE_MODEL.bat "D:\path\to\test_similarity.zip"
```

Without a path it uses the newest `test_similarity*.zip` in the project folder or the folder above it.

**What happens (about 2 minutes with an NVIDIA GPU):**

1. **Colour match model:** three files (about 9 MB) are checked and copied to `multimodal_experiments\models\glaze_retrieval\`. If a check fails, the old model stays.
2. **Smart search model:** the joint text + photo + formula model is trained on this PC from the downloaded data (STEP 2) and tested on held-out recipes. It goes to `multimodal_experiments\models\glaze_joint\`.

Both folders are not in git. Run the BAT again when a new model ZIP arrives: a smart search model that is already trained is kept (its test numbers stay valid). To train it again: `.venv\Scripts\python.exe scripts\train_glaze_joint.py --retrain`.

Card G also reads `7k_matched_all_data.csv` and `7k_matched_ingredients.csv` (recipe names, original cone values, ingredients) from the folder above the project (`D:\NUS study\CG2`), from `new image data\csv 7k data\`, or from the folder in the `CG2_7K_CSV_DIR` setting.

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
- **Card G · Glaze recipes:** find Glazy recipes from your own words, a colour, a photo object or a glaze formula (needs STEP 4); see [How to use card G](#how-to-use-card-g-glaze-recipes).

Guide: [docs/SEARCH_GUIDE.md](docs/SEARCH_GUIDE.md).

Uploads and searches: `multimodal_experiments/work/queries/`; session log: `multimodal_experiments/work/queries/session_log.json`; app log: `multimodal_experiments/work/logs/app.log`.

### Search labels

| Label                                         | Source                                                       | Behavior                                        |
| --------------------------------------------- | ------------------------------------------------------------ | ----------------------------------------------- |
| Colour (three colours per object, name/hex/%) | `new image data/full_segmentation_claude_process/review/object_review.csv`, `full_auto_tags.csv`, `tag_json/` | Read at startup                                 |
| Six attributes                                | `multimodal_experiments/work/attributes_v3/` (plus v1/v2), `review/tile_review.json` | Human > Claude > Grounding DINO > earlier model |
| Review decision                               | `review/object_review.csv` (`review_status`)                 | Objects marked *not needed* excluded            |

#### How to use card G (glaze recipes)

| Step | What you do | What happens |
|---|---|---|
| 1 | Card 1: drop or paste a photo (optional) | SAM 2.1 outlines every ceramic object |
| 2 | Click the object you mean | It becomes the active object (yellow outline) |
| 3 | **Pick colour (HEX)** → click the glaze → **Use for glaze search**, or press **glaze** next to a colour region in card 1b | The colour (median of 13 × 13 pixels inside the object, highlights and shadows left out) goes to card G; correct the HEX by hand if needed |
| 4 | Card G · **Describe it**: type freely, e.g. `shinyy cobolt blu ^6 ox, kinda runny` → **Read it** | Keywords and values are read and shown as chips: colour → HEX, Orton cone (`04` ≠ `4`), surface, transparency, atmosphere, effects, colourant %, oxides; typos are fixed; suggestions come from Glazy descriptions |
| 5 | Choose **Smart search** (text + colour + photo object + formula + cone + surface) or **Colour match** (colour + cone + surface), then **Find recipes** | Every recipe is scored, about 0.1 s |
| 6 | Read **Best matches** and **Explore formulations** (≤ 2 per family) | Each card: Glazy photo, score parts, ingredients, **View on Glazy** |

The page has a nav bar: 🏠 Start · 📷 My photo · 🧪 Glaze recipes · 🏺 Similar pottery · 📊 Scores · 📖 How to use. Manual with worked examples: [CG2_Glaze_Search_Manual.html](docs/manuals/CG2_Glaze_Search_Manual.html) / [PDF](docs/manuals/CG2_Glaze_Search_Manual.pdf) (also the app's 📖 How to use page).

Step-by-step guide with screenshots: [CG2_Glaze_Search_Step_by_Step_Guide.pdf](docs/manuals/CG2_Glaze_Search_Step_by_Step_Guide.pdf) · process, models and parameters: [CG2_Glaze_Search_Process_Models_MindMap.pdf](docs/manuals/CG2_Glaze_Search_Process_Models_MindMap.pdf) · [usage mind map](docs/glaze_search_usage_mindmap.png).

The whole pipeline, data → preprocessing → training → evaluation: [glaze_search_ml_pipeline.png](docs/glaze_search_ml_pipeline.png). No pretrained model is fine-tuned: MiniLM and DINOv2 stay frozen and three small heads are trained on top of them (InfoNCE); "Explore formulations" uses K-Means families (k-means++ initialisation) because density clustering (DBSCAN / HDBSCAN) found no useful groups.

Held-out test results of the smart search model (recipes it never saw): a noisy sentence finds the right recipe in the top 10 for 74 % of the test recipes (pretrained text model alone: 62 %); a photo object for 45 %; a formula for 33 % (random: 1 %). Read values are right for 93 % (colour) to 100 % (atmosphere) of noisy sentences.

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
│   ├── INSTALL_GLAZE_MODEL.bat             # card G models (STEP 4)
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

- [Glaze Search on Hugging Face](https://huggingface.co/spaces/FBIWarninggg/glaze_search) (Hugging Face's free GPU)
- Web pages: [Glaze Search Mind Map](https://claude.ai/artifact/B1zswhT3kHGYnZYYvYgaa5) · [Glaze Search Evaluation](https://claude.ai/artifact/BVYzi6cWhbLpf6x1N8qQTh) · [Glaze Search Manual](https://claude.ai/artifact/7rmiKzN17dk2Fhb4AGeJhz)

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

- [Glaze Search manual](docs/manuals/CG2_Glaze_Search_Manual.html) ([PDF](docs/manuals/CG2_Glaze_Search_Manual.pdf)) — the pages, six worked examples and what every number means.
- [Glaze Search Guide](docs/SEARCH_GUIDE.md) — Search inputs, priority sliders, results, and evaluation.
- [Glaze recipe search (card G)](docs/glaze_integration_architecture.md) — smart search and colour match: architecture, API, cone handling, tests, and how to update the models ([system diagram](docs/glaze_search_full_flow.png)).
- [Step-by-step guide (PDF)](docs/manuals/CG2_Glaze_Search_Step_by_Step_Guide.pdf) and [process, models and parameters (PDF)](docs/manuals/CG2_Glaze_Search_Process_Models_MindMap.pdf) for card G.
- [Pretrained models, illustrated (PDF)](docs/manuals/CG2_Glaze_Search_Pretrained_Models.pdf) — how each pretrained model works on a real example, with an activity diagram.
- [Evaluation (PDF)](docs/manuals/CG2_Glaze_Search_Evaluation.pdf) — every metric and test result, stage by stage.
- [Data Download Guide](docs/DATA_DOWNLOAD.md) — Data archives, manual downloads, and dataset relationships.
- [Manuals and Screenshots](docs/manuals) — Detailed PDF manuals, workflow diagrams, and attribute documentation.

---

## 9. What is (not) on this branch

Included: Glaze Search server/page/search engine, shared modules, configuration, colour vocabulary, reference library, setup/download/check scripts, guides and manuals. Data comes from release data-v2; pretrained models from `scripts/setup_models.py` (pinned versions); the card G recipe models from `INSTALL_GLAZE_MODEL.bat` (STEP 4).

Not included: Review app and review-only data (see [`nghia_mike_ceramic`](https://github.com/haohaozai/GC2-Ensemble-Model-Pottery/tree/nghia_mike_ceramic) and data-v1), segmentation/training pipelines, Python environment, personal settings and uploaded photos.

**Windows quick start:** `SETUP.bat` → `DOWNLOAD_DATA_ORIGINAL_AND_SEGMENTATION.bat` → `CHECK_DATA.bat` → `INSTALL_GLAZE_MODEL.bat` (card G) → `RUN_GLAZE_SEARCH.bat`.

**macOS quick start (after data downloader is converted or data is correctly installed):** `SETUP.command` → data installation → `CHECK_DATA.command` → `RUN_GLAZE_SEARCH.command`.
