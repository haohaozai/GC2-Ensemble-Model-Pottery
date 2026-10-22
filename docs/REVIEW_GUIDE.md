# Review app (`OPEN_REVIEW_APP.bat`)

The Review app shows every ceramic photo with the objects that the segmentation cut out of it, their colours and
attributes, and lets you mark each object. It runs only on your computer: http://127.0.0.1:8765 (the next free port if
8765 is taken). Keep the black window open while you review; close it to stop the app.

Details with screenshots: [manuals/CG2_Manual_v4_Review_and_Search.pdf](manuals/CG2_Manual_v4_Review_and_Search.pdf).

## What you see

* **Photo cards**: the original photo with the outline and number of every object (from `instance_masks`), then one
  card per object: the crop (`smaller_object`), its 3 main colours (ColorThief), surface, transparency, glaze effects,
  defects and the review status.
* **Overview** (key `V` or the ⤢ button): the whole photo, all its objects, the pipeline's 4-panel preview (`preview`) and,
  for photos that were corrected by hand, the preview from before the correction.
* **Tile ▸** (attribute review of one object): for each of the six attributes (form, surface finish, transparency,
  texture, glaze effect, defect) the Grounding DINO label (validated / uncertain) with its top-3 candidates, Claude's
  check, the recipe / CSV value and the earlier model. Click a candidate to see its box on the cut-out, **use** it,
  or pick any value; then **Save attribute review** or **Save & next**.

## Show filter

To review · Bonus images (125 extra photos) · Needs review · Unreviewed · Not needed · Completed (round 1) ·
Completed v2 (round 2) · Corrected segmentation · Review tile (category) · Attributes to check (most uncertain first) ·
All photos.

## Keys (mouse over a card)

| key | does |
|---|---|
| `N` | not needed (press again: restore) |
| `R` | needs review + note |
| `I` | ignore the rest of the photo's objects (again: undo) |
| `T` | add / remove the **review_tile** category |
| `←` `→` | previous / next object |
| `V` · `O` | overview of the photo · outlines on / off |
| `A` (Tile panel) | accept every validated Grounding DINO label |

## What "Save" changes (only on this computer)

| file / folder (in `new image data\full_segmentation_claude_process\`) | change |
|---|---|
| `review\object_review.csv` | rewritten: the master record of all decisions |
| `full_auto_tags.csv` | review_status / note / date / location columns updated |
| `does_not_needed\…` | objects marked *Not needed* are **moved** there with their 4 companion files (restored when unmarked) |
| `review\move_log.csv` | every move is logged |
| `review\tile_review.json` + `tile_review_history.jsonl` | Tile reviews and their history |
| `review_thumbs\` | thumbnail cache (made automatically, can be deleted) |

The bonus photos keep the same files in `bonus_image_claude_process\`. The page also downloads the review CSV.
Glaze Search reads these review files (the best available label of every object), so it sees your decisions after a
restart.

> Each computer keeps its own copy of these files. To review together, use the online review page (below) or send
> each other the `object_review.csv`.

## Online review page (optional, Supabase)

The same review page can run online (anyone views, editors sign in with email + editor code). Decisions are stored in a
free Supabase database:

1. Create a Supabase project, open *SQL Editor* and run, in this order: `web/supabase_setup.sql` (replace
   `YOUR_EMAIL@example.com` at the end with the admin's email first), `web/supabase_editor_codes.sql`,
   `web/supabase_tile_reviews.sql`.
2. Copy `web/publish_settings.example.json` to `web/publish_settings.json` and fill in the project URL and the
   **anon / publishable** key (*Project Settings > API*). Never use the `service_role` / secret key: this file is read
   by the sync script and must stay public-safe. `web/publish_settings.json` is ignored by git.
3. `SYNC_ONLINE_REVIEWS.bat` shows the online decisions and Tile reviews that differ from this computer, asks, then
   writes them into `review\object_review.csv` / `review\tile_review.json` (the newer one wins; history is kept).

The page of the first PC: https://fbiwarninggg-ceramic-review.static.hf.space

## Problems

| message | fix |
|---|---|
| `ERROR: Python environment not found` | run `SETUP.bat` |
| `full_auto_tags.csv` not found / empty page | run `DOWNLOAD_DATA.bat`, then `CHECK_DATA.bat`; if the data is elsewhere set `CG2_DATA` in `.env` |
| Tile panel shows no cut-out | the cut-outs were not rebuilt: `.venv\Scripts\python.exe scripts\rebuild_tile_cutouts.py` |
| the page does not open | open http://127.0.0.1:8765 yourself (or the port printed in the black window) |
