# CG2 Ceramic Review and Glaze Search

A ceramic image review and glaze search system developed for the CG2 pottery project.

Built using **7,063 Glazy glaze recipes** and **12,309 segmented ceramic objects**, processed with Grounding DINO, SAM 2.1, and CLIP.

This project provides two applications:

- **Ceramic Review:** Browse ceramic images, inspect segmented objects, review attributes, and save review decisions.
- **Glaze Search:** Upload a ceramic photo and search for visually similar objects based on shape, glaze, surface, and colour.

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
- [Glaze Search Guide](docs/SEARCH_GUIDE.md) — Search inputs, priority sliders, results, and evaluation.
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
- [ ] Launch `OPEN_REVIEW_APP.bat` or `RUN_GLAZE_SEARCH.bat`.

**Installation order: SETUP → DOWNLOAD DATA → CHECK DATA → LAUNCH APP.**
