# Data download

The images and trained files are too big for a git repository (GitHub recommends < 1 GB per repository). They are
attached as 10 archives (10.3 GB) to the GitHub release **data-v1**:
https://github.com/nghialeduc268-a11y/nghia_ceramic/releases/tag/data-v1

**Use `DOWNLOAD_DATA.bat`** (after `SETUP.bat`): it downloads every archive, continues interrupted downloads, checks
size and SHA-256 against `data_manifest.json`, unpacks each archive into the right folder, skips archives that are
already unpacked, and finally rebuilds the Tile cut-outs. Run it again whenever it stopped.

## The archives

| file | size | files | unpacked to | needed by | contents |
|---|---|---|---|---|---|
| `cg2_photos_7k_part1of2.tar` | 1.29 GB | 3,482 | data | both | `7k_matched_images/7k_matched_images` (original photos) |
| `cg2_photos_7k_part2of2.tar` | 1.29 GB | 3,581 | data | both | same, second half |
| `cg2_object_crops_part1of2.tar` | 1.29 GB | 6,451 | data | both | `full_segmentation_claude_process/smaller_object` (object crops) |
| `cg2_object_crops_part2of2.tar` | 1.29 GB | 5,479 | data | both | same, second half |
| `cg2_detailed_previews_part1of2.tar` | 1.42 GB | 7,305 | data | Review | `…/detailed_preview` (card view) |
| `cg2_detailed_previews_part2of2.tar` | 1.42 GB | 7,635 | data | Review | same + `…/does_not_needed/detailed_preview` |
| `cg2_review_core.tar` | 1.74 GB | 54,398 | data | Review | `preview`, `instance_masks`, `smaller_object_masks_raw`, `meta`, `tag_json`, `full_auto_tags.csv`, `review/`, previews before correction, `does_not_needed/` crops |
| `cg2_bonus_set.tar` | 0.22 GB | 2,147 | data | both | `bonus_image_claude_process` (125 extra photos) + `others_image` |
| `cg2_search_artifacts.tar` | 0.31 GB | 7,213 | project | Glaze Search | `multimodal_experiments/work` + `cache`: search index, trained models, recipe thumbnails |
| `cg2_model_dinov2_code.tar` | 0.002 GB | 203 | models | Glaze Search | DINOv2 code folder for `torch.hub` |

*data* = `CG2_DATA` (default `<CG2>\new image data`) · *project* = this repository folder · *models* = `CG2_MODEL_CACHE`
(default `<CG2>\model_cache`). Exact sizes and SHA-256: `data_manifest.json`.

## Made on your computer (not downloaded)

* **Tile cut-outs** `smaller_object_rgba_raw` (15,351 PNG, ~12 GB): each is the original photo cropped to the object's
  box (`meta/<photo>.json` → `segmentation.instances[index].box_xyxy`, x0 y0 x1 y1, end excluded) with the raw mask as
  transparency. `scripts/rebuild_tile_cutouts.py` (run by `DOWNLOAD_DATA.bat`) makes them; they are pixel-identical to
  the originals (checked for every object).
* **Review thumbnails** (`review_thumbs`): made by the Review app when needed.
* **Pretrained models** (~3.7 GB): `SETUP.bat` downloads them from Hugging Face and Meta at fixed versions.

## Downloading by hand (browser, Google Drive copy, USB)

Put all 10 `.tar` files in one folder and run

```
DOWNLOAD_DATA.bat --from-folder "D:\folder\with\the\tar\files"
```

Do not unpack them yourself: the archives go to three different places (see the table).

## How the data is linked

| what | rule |
|---|---|
| photo id | file name of the photo: `7k_matched_images/7k_matched_images/3032.jpg` → `3032` |
| object id | `<photo id>_<index>`, e.g. `3032_1`; `index` = pixel value of the object in `instance_masks/<photo>.png` |
| object box | `meta/<photo>.json` → `segmentation.instances[index].box_xyxy` = `[x0, y0, x1, y1]` in photo pixels (end excluded) |
| crop / mask / cut-out | `smaller_object/<object>.jpg`, `smaller_object_masks_raw/<object>.png`, `smaller_object_rgba_raw/<object>.png` (box size) |
| recipe | `full_auto_tags.csv` columns `recipe_id`, `recipe_name` (Glazy) per object |
| review decisions | `review/object_review.csv` (master record) |

## What is *not* included

The segmentation pipeline's intermediate outputs (`rgba`, `white_background`, `masks`, `diagnostics`, `evaluation`,
`logs`, correction sheets and backups except the previews, `*_v3.1_backup*`, `review/history`) — about 26 GB that
neither app reads. They are kept on the first PC.
