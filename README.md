# CG2 Ceramic Review and Glaze Search

A ceramic image review and glaze search system developed for the CG2 pottery project.

Built using **7,063 Glazy glaze recipes** and **12,309 segmented ceramic objects**, processed with Grounding DINO, SAM 2.1, and CLIP.

This project provides two applications:

- **Ceramic Review:** Browse ceramic images, inspect segmented objects, review attributes, and save review decisions.
- **Glaze Search:** Upload a ceramic photo and search for visually similar objects based on shape, glaze, surface, and colour, or find glaze recipes from a photo, words or a chemical formula (card G).

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

**Important: Use the `nghia_mike_ceramic` branch, not `main`.**

Repository: [GC2-Ensemble-Model-Pottery](https://github.com/haohaozai/GC2-Ensemble-Model-Pottery/tree/nghia_mike_ceramic)

### Step 1 — Choose a folder

You can install the project on any drive, for example:

- `D:\CG2`
- `C:\CG2`
- `D:\NUS study\CG2`

A short folder path is recommended.

Open **Windows CMD** and navigate to your chosen folder.

Example:

```cmd
mkdir D:\CG2
cd /d D:\CG2
```

### Step 2 — Clone the correct branch

Copy this command into CMD:

```cmd
git clone -b nghia_mike_ceramic --single-branch https://github.com/haohaozai/GC2-Ensemble-Model-Pottery.git
```

Once completed, open the downloaded project folder:

```cmd
cd GC2-Ensemble-Model-Pottery
```

You should now be inside the folder containing the project files.

**Alternative:** Open the GitHub branch link, click **Code → Download ZIP**, then extract the ZIP into your chosen folder.

---

## 2. Installation — Complete these 3 steps in order

**Do not launch the applications before completing all three steps.**

### STEP 1 — Run SETUP.bat

From inside the downloaded project folder, double-click:

`SETUP.bat`

Or run it using CMD:

```cmd
SETUP.bat
```

**What happens:**

- Creates the Python environment (`.venv`).
- Installs Python 3.11 and the required packages.
- Installs PyTorch for GPU or CPU execution.
- Downloads pretrained AI models (approximately 3.7 GB).
- Prepares the applications for execution.

**Before running:** Install [uv](https://docs.astral.sh/uv/) if it is not already installed. In PowerShell, run:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then reopen CMD.

**Wait until SETUP.bat finishes successfully before continuing.**

### STEP 2 — Run DOWNLOAD_DATA.bat

After setup completes, double-click:

`DOWNLOAD_DATA.bat`

Or run:

```cmd
DOWNLOAD_DATA.bat
```

**What happens:**

- Downloads the ceramic datasets from the GitHub release.
- Retrieves approximately 10.3 GB of compressed data across 10 archives.
- Extracts the 7,063 original glaze photos.
- Restores the segmented ceramic objects, annotations, masks, and review data.
- Rebuilds Tile cut-out images.
- Places everything in the expected project folders.

The download source is:

[GitHub Data Release — data-v1](https://github.com/nghialeduc268-a11y/nghia_ceramic/releases/tag/data-v1)

**Important:** This is the largest data download step. Do not close the CMD window while it is running.

If the download is interrupted, run `DOWNLOAD_DATA.bat` again. The script is designed to continue and skip successfully completed downloads.

### STEP 3 — Run CHECK_DATA.bat

After downloading, double-click:

`CHECK_DATA.bat`

Or run:

```cmd
CHECK_DATA.bat
```

**What happens:**

- Verifies the required folders and datasets.
- Checks whether the project files are available.
- Reports missing data.
- Confirms whether the installation is ready.

**Expected result:** All required checks should display `OK`.

If a check displays `NO`, read its message and rerun `DOWNLOAD_DATA.bat` or `SETUP.bat` as appropriate.

**Once all required checks pass, the installation is complete.**

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

## 3. Launch the applications

After completing the installation, you can use either application.

### Application A — Ceramic Review

Double-click:

`OPEN_REVIEW_APP.bat`

Or run:

```cmd
OPEN_REVIEW_APP.bat
```

Open in your browser:

http://127.0.0.1:8765

**Features:**

- Browse original ceramic photos and segmented objects.
- Inspect colours and six ceramic attributes.
- Mark objects as *Not needed* or *Needs review*.
- Review individual objects in the Tile panel.
- Save review decisions to local CSV and JSON files.

### Application B — Glaze Search

Double-click:

`RUN_GLAZE_SEARCH.bat`

Or run:

```cmd
RUN_GLAZE_SEARCH.bat
```

Open in your browser:

http://127.0.0.1:8770

**Features:**

- Upload, drag, or paste ceramic photos.
- Detect and describe ceramic objects.
- Search by shape, glaze, surface, and colour.
- Adjust search priorities using sliders.
- Optionally search with text and glaze formulas.
- Evaluate results using Hit@K, Precision@K, Recall@K, and nDCG.
- **Card G · Glaze recipes:** find Glazy recipes from your own words, a colour, a photo object or a glaze formula. Needs STEP 4. See [How to use card G](#how-to-use-card-g-glaze-recipes) below.

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

**Important:** Keep the CMD window open while using either application. Closing it stops the corresponding local server.

---

## 4. System requirements

| Requirement | Details |
|---|---|
| Operating system | Windows 10 or Windows 11 |
| Storage | Approximately 35 GB free |
| Initial download | Approximately 17 GB |
| Python | Python 3.11, installed through uv |
| GPU | NVIDIA GPU with at least 6 GB VRAM recommended for Glaze Search |
| CPU support | Both apps can run without an NVIDIA GPU |
| Internet | Required for initial installation and downloads |

Approximate download time:

| Internet speed | Estimated time for 17 GB |
|---|---|
| 2 MB/s | 2.5 hours |
| 5 MB/s | 1 hour |
| 10 MB/s | 30 minutes |

After setup, the local applications can operate offline.

On CPU, processing a new photo for Glaze Search may take 1–3 minutes.

---

## 5. Project structure

```text
CG2/
├── GC2-Ensemble-Model-Pottery/
│   ├── SETUP.bat
│   ├── DOWNLOAD_DATA.bat
│   ├── CHECK_DATA.bat
│   ├── OPEN_REVIEW_APP.bat
│   ├── RUN_GLAZE_SEARCH.bat
│   ├── INSTALL_GLAZE_MODEL.bat
│   ├── SYNC_ONLINE_REVIEWS.bat
│   ├── review_app/
│   ├── multimodal_experiments/
│   ├── glazeprep/
│   ├── reference_library/
│   ├── third_party/
│   ├── scripts/
│   ├── configs/
│   ├── web/
│   ├── docs/
│   └── data_manifest.json
│
├── new image data/
│   ├── 7k_matched_images/
│   ├── full_segmentation_claude_process/
│   ├── bonus_image_claude_process/
│   └── others_image/
│
└── model_cache/
```

The `new image data` and `model_cache` folders are created or populated during installation.

If your data is stored elsewhere, copy `.env.example` to `.env` and configure `CG2_DATA`, `CG2_ROOT`, or `CG2_MODEL_CACHE`.

---

## 6. Online versions

Online copies are also available:

- [Ceramic Review Page](https://fbiwarninggg-ceramic-review.static.hf.space)
- [Glaze Search on Hugging Face](https://huggingface.co/spaces/FBIWarninggg/glaze_search)
- Web pages: [Glaze Search Mind Map](https://claude.ai/artifact/B1zswhT3kHGYnZYYvYgaa5) · [Glaze Search Evaluation](https://claude.ai/artifact/BVYzi6cWhbLpf6x1N8qQTh) · [Glaze Search Manual](https://claude.ai/artifact/7rmiKzN17dk2Fhb4AGeJhz)

The online Review page can use Supabase for shared review decisions. The local Review app saves data on your computer.

For online review setup, refer to the Supabase SQL files in `web/`. Never commit secret keys or service-role credentials to GitHub.

---

## 7. Troubleshooting

| Problem | Solution |
|---|---|
| `git` is not recognized | Install Git for Windows and reopen CMD |
| `uv was not found` | Install uv and reopen CMD |
| Python environment not found | Run `SETUP.bat` |
| Required data missing | Run `DOWNLOAD_DATA.bat` again |
| Download interrupted | Rerun `DOWNLOAD_DATA.bat` |
| SHA-256 verification fails | Rerun the download to replace damaged files |
| AI models are missing | Run `SETUP.bat` again |
| CUDA out of memory | Close other GPU applications or use CPU mode |
| Review app port is occupied | Check the CMD window for the selected port |
| Glaze Search port is occupied | Run `RUN_GLAZE_SEARCH.bat --port 8771` |
| Tile cut-outs missing | Run the Tile rebuilding script |

For CPU-only Glaze Search, use CMD:

```cmd
set CUDA_VISIBLE_DEVICES=-1
RUN_GLAZE_SEARCH.bat
```

For manually downloaded archives:

```cmd
DOWNLOAD_DATA.bat --from-folder "D:\YourDownloadFolder"
```

To rebuild missing Tile cut-outs:

```cmd
.venv\Scripts\python.exe scripts\rebuild_tile_cutouts.py
```

---

## 8. Manuals and additional documentation

Detailed instructions are available in the repository:

- [Review Guide](docs/REVIEW_GUIDE.md) — Review interface, filters, keyboard shortcuts, Tile panel, and saving decisions.
- [Glaze Search manual](docs/manuals/CG2_Glaze_Search_Manual.html) ([PDF](docs/manuals/CG2_Glaze_Search_Manual.pdf)) — the pages, six worked examples and what every number means.
- [Glaze Search Guide](docs/SEARCH_GUIDE.md) — Search inputs, priority sliders, results, and evaluation.
- [Glaze recipe search (card G)](docs/glaze_integration_architecture.md) — smart search and colour match: architecture, API, cone handling, tests, and how to update the models ([system diagram](docs/glaze_search_full_flow.png)).
- [Step-by-step guide (PDF)](docs/manuals/CG2_Glaze_Search_Step_by_Step_Guide.pdf) and [process, models and parameters (PDF)](docs/manuals/CG2_Glaze_Search_Process_Models_MindMap.pdf) for card G.
- [Pretrained models, illustrated (PDF)](docs/manuals/CG2_Glaze_Search_Pretrained_Models.pdf) — how each pretrained model works on a real example, with an activity diagram.
- [Evaluation (PDF)](docs/manuals/CG2_Glaze_Search_Evaluation.pdf) — every metric and test result, stage by stage.
- [Data Download Guide](docs/DATA_DOWNLOAD.md) — Data archives, manual downloads, and dataset relationships.
- [Manuals and Screenshots](docs/manuals) — Detailed PDF manuals, workflow diagrams, and attribute documentation.

These documents are optional references. You do not need to read them before following the installation steps.

---

## 9. Quick-start checklist

- [ ] Clone the `nghia_mike_ceramic` branch.
- [ ] Install uv if needed.
- [ ] Run `SETUP.bat` and wait for completion.
- [ ] Run `DOWNLOAD_DATA.bat` and wait for completion.
- [ ] Run `CHECK_DATA.bat` and confirm all required checks pass.
- [ ] Run `INSTALL_GLAZE_MODEL.bat` with the model ZIP (for card G).
- [ ] Launch `OPEN_REVIEW_APP.bat` or `RUN_GLAZE_SEARCH.bat`.

**Installation order: SETUP → DOWNLOAD DATA → CHECK DATA → LAUNCH APP.**
