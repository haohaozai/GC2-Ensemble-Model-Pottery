# Data download (Glaze Search)

The photos and the segmentation are too big for a git repository. They are attached as 8 archives (6.15 GB) to the
GitHub release **data-v2**:
https://github.com/nghialeduc268-a11y/nghia_ceramic/releases/tag/data-v2

data-v2 holds only what `RUN_GLAZE_SEARCH.bat` needs: the **original photos** and the **segmentation** (object crops,
colour and attribute labels, masks, review records) plus the search index. The review is finished; the Review app and
the files only it reads (previews, rejected objects, Tile cut-outs) are on branch
[`nghia_mike_ceramic`](https://github.com/haohaozai/GC2-Ensemble-Model-Pottery/tree/nghia_mike_ceramic) with release
[data-v1](https://github.com/nghialeduc268-a11y/nghia_ceramic/releases/tag/data-v1).

**Use `DOWNLOAD_DATA_ORIGINAL_AND_SEGMENTATION.bat`** (after `SETUP.bat`): it downloads every archive, continues
interrupted downloads, checks size and SHA-256 against `data_manifest.json`, unpacks each archive into the right folder
and skips archives that are already unpacked. Run it again whenever it stopped.

## The archives

| file | size | files | unpacked to | contents |
|---|---|---|---|---|
| `cg2_photos_7k_part1of2.tar` | 1.29 GB | 3,482 | data | `7k_matched_images/7k_matched_images` (original photos) |
| `cg2_photos_7k_part2of2.tar` | 1.29 GB | 3,581 | data | same, second half |
| `cg2_object_crops_part1of2.tar` | 1.29 GB | 6,451 | data | `full_segmentation_claude_process/smaller_object` (segmented objects) |
| `cg2_object_crops_part2of2.tar` | 1.29 GB | 5,479 | data | same, second half |
| `cg2_segmentation_labels.tar` | 0.46 GB | 37,994 | data | `full_segmentation_claude_process`: `review/object_review.csv`, `review/tile_review.json`, `full_auto_tags.csv`, `tag_json/`, `meta/`, `instance_masks/`, `smaller_object_masks_raw/` |
| `cg2_bonus_set.tar` | 0.22 GB | 2,147 | data | `bonus_image_claude_process` (125 extra photos, same structure) + `others_image` (their originals) |
| `cg2_search_artifacts.tar` | 0.31 GB | 7,213 | project | `multimodal_experiments/work` + `cache`: search index, attribute labels and models, recipe thumbnails |
| `cg2_model_dinov2_code.tar` | 0.004 GB | 203 | models | DINOv2 code folder for `torch.hub` |

*data* = `CG2_DATA` (default `<CG2>\new image data`) · *project* = this repository folder · *models* = `CG2_MODEL_CACHE`
(default `<CG2>\model_cache`). Exact sizes and SHA-256: `data_manifest.json`. Seven archives are byte-identical to
the ones of data-v1; `cg2_segmentation_labels.tar` is new (the label and mask files of data-v1's `cg2_review_core.tar`,
without the Review app's previews and rejected objects).

## Where the labels are

| label | files | read by Glaze Search |
|---|---|---|
| colour: the 3 main colours of each object (name, hex, % of the object) | `review/object_review.csv` columns `color1_*` … `color3_*`; the same values are in `full_auto_tags.csv` and `tag_json/<object>.json` | yes, at start-up: the palette on every result and the colour check of the evaluation |
| the six attributes: object form, surface finish, transparency, surface texture, glaze effects, possible defects | `multimodal_experiments/work/attributes_v3/attribute_predictions_v3.json` (+ `_bonus`), `attributes_v2`, `attributes_v1/attribute_predictions.json`; human-checked ones in `review/tile_review.json`; the first tagger's values in `full_auto_tags.csv` / `tag_json` | yes: best available label per object (human > Claude > Grounding DINO > earlier model) |
| review decision | `review/object_review.csv` column `review_status` (objects marked *not needed* are not in the search) | yes |
| segmentation | `meta/<photo>.json` (boxes, scores), `instance_masks/<photo>.png`, `smaller_object_masks_raw/<object>.png`, crops `smaller_object/<object>.jpg` | crops yes; boxes and masks are included so the segmentation is complete and can be reused |
| recipe | `full_auto_tags.csv` / `tag_json` columns `recipe_id`, `recipe_name`, ingredients (Glazy) | through the search index |

## Downloading by hand (browser, Google Drive copy, USB)

Put all 8 `.tar` files in one folder and run

```
DOWNLOAD_DATA_ORIGINAL_AND_SEGMENTATION.bat --from-folder "D:\folder\with\the\tar\files"
```

Do not unpack them yourself: the archives go to three different places (see the table).

## How the data is linked

| what | rule |
|---|---|
| photo id | file name of the photo: `7k_matched_images/7k_matched_images/3032.jpg` → `3032` |
| object id | `<photo id>_<index>`, e.g. `3032_1`; `index` = pixel value of the object in `instance_masks/<photo>.png` |
| object box | `meta/<photo>.json` → `segmentation.instances[index].box_xyxy` = `[x0, y0, x1, y1]` in photo pixels (end excluded) |
| crop / mask | `smaller_object/<object>.jpg`, `smaller_object_masks_raw/<object>.png` (box size) |
| recipe | `full_auto_tags.csv` columns `recipe_id`, `recipe_name` (Glazy) per object |
| review decisions | `review/object_review.csv` (master record) |

## What is *not* included

Only the Review app reads these (they are in data-v1): `preview`, `detailed_preview`, `does_not_needed` (the 3,010
objects marked *not needed*), `manual_corrections_backup*`, the Tile cut-outs `smaller_object_rgba_raw` (~12 GB, rebuilt
on the computer) and `review_thumbs`. The segmentation pipeline's intermediate outputs (about 26 GB) are kept on the
first PC.
