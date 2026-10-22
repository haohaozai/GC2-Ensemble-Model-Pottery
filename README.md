# CG2 Ceramic Review and Glaze Search

Two local web apps of the CG2 pottery project, built on **7,063 Glazy glaze recipes** whose photos were segmented
into **12,309 ceramic objects** (Grounding DINO + SAM 2.1 + CLIP):

| app | start with | what it does |
|---|---|---|
| **Review** | `OPEN_REVIEW_APP.bat` → http://127.0.0.1:8765 | browse every photo with its segmented objects, colours and six attributes; mark objects *Not needed* / *Needs review*; review the attributes of single objects (Tile panel); everything is saved in CSV / JSON files on your computer |
| **Glaze Search** | `RUN_GLAZE_SEARCH.bat` → http://127.0.0.1:8770 | drop or paste a photo of a glaze: it finds the objects, describes them (colour regions + six attributes) and searches the database by shape, glaze / surface and colour (priority sliders), optionally with text and a glaze formula; every search is evaluated (Hit@K, Precision@K, Recall@K, nDCG) |

Online copies of the first PC: [review page](https://fbiwarninggg-ceramic-review.static.hf.space) ·
[Glaze Search](https://huggingface.co/spaces/FBIWarninggg/glaze_search).

---

## What you need

* Windows 10 or 11, internet for the setup, **about 45 GB free disk** (35 GB stay: data 10 GB + rebuilt cut-outs 12 GB
  + models 4 GB + Python environment 5 GB; the 10 GB of downloaded archives are deleted after unpacking).
* For Glaze Search an **NVIDIA GPU with ≥ 6 GB** memory is recommended (it uses ~4 GB). Without one it runs on the CPU:
  a new photo then takes 1 - 3 minutes, searching stays fast. The Review app needs no GPU.
* [uv](https://docs.astral.sh/uv/) (installs Python 3.11 for you). Install it once in PowerShell:
  `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`

## Install (once, ~30 - 60 min, mostly downloading)

1. Make a folder for the project, e.g. `D:\CG2`, and get this branch into it:
   ```
   cd /d D:\CG2
   git clone -b nghia_ceramic_part https://github.com/nghialeduc268-a11y/nghia_ceramic.git
   ```
   (or on GitHub: branch `nghia_ceramic_part` → **Code → Download ZIP**, and unzip it into `D:\CG2`.)
2. Double-click **`SETUP.bat`**: Python environment `.venv`, PyTorch (GPU or CPU build), the pinned packages and the
   pretrained models (~3.7 GB, fixed versions).
3. Double-click **`DOWNLOAD_DATA.bat`**: the data (10 archives, 10.3 GB) from the release
   [data-v1](https://github.com/nghialeduc268-a11y/nghia_ceramic/releases/tag/data-v1), checked and unpacked into the
   right folders, then the Tile cut-outs are rebuilt. Interrupted? Run it again — it continues.
4. Double-click **`CHECK_DATA.bat`**: every folder shows `OK` (or what is missing and how to fix it).
5. Start **`OPEN_REVIEW_APP.bat`** or **`RUN_GLAZE_SEARCH.bat`**. Keep the black window open while you use the page;
   close it to stop.

All `.bat` files can be run again safely; finished steps are skipped.

## Folder layout

```
D:\CG2\                          <- any folder (the "CG2 root")
├── nghia_ceramic\               <- this repository (the folder name does not matter)
│   ├── OPEN_REVIEW_APP.bat  RUN_GLAZE_SEARCH.bat  SETUP.bat  DOWNLOAD_DATA.bat  CHECK_DATA.bat  SYNC_ONLINE_REVIEWS.bat
│   ├── review_app\              Review page          scripts\       Review server, setup / data / check scripts
│   ├── multimodal_experiments\  Glaze Search: app\ (server + page), mm\ (search engine), vendor\ (text branch)
│   │                            work\ + cache\ = search index and trained files (from the download)
│   ├── glazeprep\               shared code: paths, segmentation, CLIP, Tile reviews
│   ├── reference_library\       example images + definitions behind the (i) buttons
│   ├── third_party\color_lab_v3\  colour vocabulary (Color Search Lab V3)
│   ├── configs\  web\           thresholds · Supabase SQL for the online review page
│   ├── docs\                    guides + manuals (PDF)
│   └── data_manifest.json  requirements.txt  .env.example
├── new image data\              <- made by DOWNLOAD_DATA.bat
│   ├── 7k_matched_images\7k_matched_images\    original photos
│   ├── full_segmentation_claude_process\       objects, previews, masks, annotations, review files
│   ├── bonus_image_claude_process\             the 125 extra photos, same structure
│   └── others_image\                           their original photos
└── model_cache\                 <- made by SETUP.bat (pretrained models)
```

**Data somewhere else?** Copy `.env.example` to `.env` and set `CG2_DATA` (and / or `CG2_ROOT`,
`CG2_MODEL_CACHE`). Nothing else needs to be edited: every path in the code is derived from these.

## Using the apps

* **Review** — [docs/REVIEW_GUIDE.md](docs/REVIEW_GUIDE.md): filters, keys (`N` not needed, `R` needs review,
  `I` ignore the rest, `T` review tile, `V` overview), the Tile panel, what *Save* changes.
* **Glaze Search** — [docs/SEARCH_GUIDE.md](docs/SEARCH_GUIDE.md): photo, text, formula and colour inputs, presets
  and priority sliders, results and evaluation.
* **Data** — [docs/DATA_DOWNLOAD.md](docs/DATA_DOWNLOAD.md): the archives, manual download, how photos, objects,
  boxes and recipes are linked.
* Manuals with screenshots (PDF): [docs/manuals](docs/manuals) — manual v4, Glaze Search beginner guide, flow mind
  map, attribute report.

### Where results are saved

| app | saved in |
|---|---|
| Review | `new image data\full_segmentation_claude_process\review\object_review.csv` (master record), `full_auto_tags.csv`, `review\tile_review.json`; *Not needed* objects are moved to `does_not_needed\` |
| Glaze Search | `multimodal_experiments\work\queries\` (uploads + searches), `work\queries\session_log.json`, `work\logs\app.log` |

Each computer keeps its own copy. To review together use the online review page (next section).

## Online review page (optional)

The review page can also run online with a free [Supabase](https://supabase.com) database (anyone views; editors sign
in with email + editor code). Setup: run `web/supabase_setup.sql` (put your email in the last lines first),
`web/supabase_editor_codes.sql` and `web/supabase_tile_reviews.sql` in Supabase → SQL Editor; then copy
`web/publish_settings.example.json` to `web/publish_settings.json` and fill in the project URL and the **anon /
publishable** key. `SYNC_ONLINE_REVIEWS.bat` brings the online decisions to this computer (it shows them first and
asks). **Never put the `service_role` / secret key in any file.**

## Problems

| problem | fix |
|---|---|
| `uv was not found` | install uv (command above), open a **new** window, run `SETUP.bat` again |
| `ERROR: Python environment not found` | run `SETUP.bat` |
| a `CHECK_DATA.bat` line says `NO` | run `DOWNLOAD_DATA.bat` again (it only fetches what is missing); data in another folder → `.env` |
| download stops / SHA-256 does not match | run `DOWNLOAD_DATA.bat` again (continues, damaged files are fetched again) |
| GitHub download is slow or blocked | download the 10 `.tar` files in the browser from the release page, then `DOWNLOAD_DATA.bat --from-folder "<folder>"` |
| Glaze Search: model not found / wants internet | run `SETUP.bat` again (models missing) |
| Glaze Search: `CUDA out of memory` or `Another GPU job is running` | close other GPU programs; or CPU mode: `set CUDA_VISIBLE_DEVICES=-1` then `RUN_GLAZE_SEARCH.bat` in the same window |
| Glaze Search: `already running` | the page is open already: http://127.0.0.1:8770 |
| Review: Tile panel without cut-out | `.venv\Scripts\python.exe scripts\rebuild_tile_cutouts.py` |
| port in use | Review picks the next free port (see the black window); Glaze Search: `RUN_GLAZE_SEARCH.bat --port 8771` |

## What is (not) in this repository

Included: everything both apps run (server code, pages, shared modules, configuration, colour vocabulary, reference
library, Supabase SQL), setup / download / check scripts, guides and the current manuals. The data comes from the
release; the models from their original publishers (`scripts/setup_models.py`, pinned versions).

Not included: the segmentation / training pipelines and their intermediate outputs (~26 GB), the Python environment,
personal settings and keys, and uploaded photos.
