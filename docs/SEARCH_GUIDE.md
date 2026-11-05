# Glaze Search (`RUN_GLAZE_SEARCH.bat`)

Find glaze recipes and ceramic objects that look like your photo, match your words or your glaze formula. The search
runs on your computer: http://127.0.0.1:8770 (opens by itself). Keep the black window open while you search.

**Manual with worked examples** (what the app found for real photos, the top recipes and pots, every number explained):
[manuals/CG2_Glaze_Search_Manual.html](manuals/CG2_Glaze_Search_Manual.html) (opens offline in any browser) and
[manuals/CG2_Glaze_Search_Manual.pdf](manuals/CG2_Glaze_Search_Manual.pdf); in the app it is the **📖 How to use** page
(http://127.0.0.1:8770/manual/CG2_Glaze_Search_Manual.html). Step-by-step guide with screenshots of the earlier one-page layout:
[manuals/CG2_Glaze_Search_Beginner_Guide.pdf](manuals/CG2_Glaze_Search_Beginner_Guide.pdf); how everything fits together:
[manuals/CG2_Flow_MindMap.pdf](manuals/CG2_Flow_MindMap.pdf).

## The pages (nav bar at the top)

| Button | Holds | Address |
|---|---|---|
| 🏠 Start | five big buttons and four example photos (one click loads one) | `#home` |
| 📷 My photo | card 1 (photo), **What the app found** (glaze colour, shape, surface top 3 with %), card 1b | `#photo` |
| 🧪 Glaze recipes | card G (words, colour, cone and surface, engine, **Show 3 / 5 / 10 / 15 / 20**), optional formula, results | `#recipes` |
| 🏺 Similar pottery | the photo strip, card 2 (words), card 3 (formula), card 4 (colours), the search bar, results | `#pottery` |
| 📊 Scores | plain-words help and the session evaluation | `#scores` |
| 📖 How to use | quick start, the example photos and the manual | `#guide` |

All cards keep their values when you switch pages; the formula card moves to the page that uses it. Old links such as
`#glaze=3054A5&cone=6` still open the recipes page with that search. The cards are described below.

## What happens when it starts (about 30 - 60 s)

The server loads the retrieval model and the database of **7,063 Glazy recipes / 12,309 segmented objects**, the
segmentation models (Grounding DINO + SAM 2.1 + OpenCLIP), DINOv2 and the six attribute models. With an NVIDIA GPU
they use about 4 GB of graphics memory; without a GPU everything runs on the CPU (a new photo then takes 1 - 3 min,
searches stay fast). Everything works offline after `SETUP.bat`.

## The page

1. **Card 1 · Photo**: drop a photo, paste it (`Ctrl+V`), browse, or paste a link. Grounding DINO + SAM 2.1 cut out
   the ceramic objects (same settings as the 7k database photos). Hover an object to see it, **click** = use it
   (yellow outline = active), `Ctrl`+click = add more. **Pick colour (HEX)**: click inside an object, then
   **Add as search colour** (card 4) or **Use for glaze search** (card G). Or **type the colour** in the same box:
   `#` + 6 characters (`#1D325B`) is used as it is; a colour word (`red`, `sand`, `celadon`, `dark sand`,
   `reddish brown`) gives its colour plus **10 related colours** to click (nearest in CIELAB by CIEDE2000; the ColorThief
   names that *What the app found* uses, then Color Lab V3; a family word like `red` keeps its family first); a typo
   (`blu`, `brwn`, `dark blu`) gives **5 guesses**, and clicking one works like a colour word (`/api/colour_lookup`,
   `mm/colorlab.py`).
2. **Card 1b · Your object**: the colour regions of the active object and the six attributes Grounding DINO detects,
   each with its top-3 candidates; click the right one, **(i)** explains that value with example images
   (`reference_library`). Tick the attributes the results must share. **glaze** next to a colour region sends that
   colour to card G.
3. **Card G · Glaze recipes**: your words, a colour, a photo object or a formula → the closest Glazy recipes.
   **Describe it**: type freely (typos and shorthand like `^6`, `c10`, `ox` are fine) and press **Read it**: chips show
   the colour (→ HEX), Orton cone (`04` ≠ `4`), surface, transparency, atmosphere, effects, colourant % and oxides it
   understood, the typos it fixed and suggestions learnt from Glazy descriptions. *Search by colour*: a colour word, a HEX
   code or the colour picker. *Search by photo*: the colour picked on the photo (median of 13 × 13 pixels inside the
   object, reflections and shadows left out) or a colour region of card 1b; correct the HEX if the pick is off.
   **Smart search** (one joint model: text + photo object + formula, plus colour, cone and surface; only the parts you
   give count) or **Colour match** (colour + cone + surface: 0.40 latent + 0.40 CIEDE2000 + 0.10 surface + 0.10 cone).
   Two tabs from one search: **Best matches** and **Explore formulations** (the same ranking, at most 2 recipes per
   family). Each card shows the Glazy photo, the score parts and **View on Glazy**. Needs `INSTALL_GLAZE_MODEL.bat` once;
   step by step with screenshots: [manuals/CG2_Glaze_Search_Step_by_Step_Guide.pdf](manuals/CG2_Glaze_Search_Step_by_Step_Guide.pdf);
   details: [glaze_integration_architecture.md](glaze_integration_architecture.md).
4. **Card 2 · Text**: subtype, firing cone, surface, transparency, atmosphere, free notes.
5. **Card 3 · Formula**: oxides (weight % or UMF), optional ingredients.
6. **Card 4 · Colours**: colour words (Color Lab V3: celadon, crimson, eggshell …), HEX codes or phrases.
7. **Search bar**: search *objects* (the segmented database objects) or *recipes*; a preset or your own priority with
   the sliders (shape / glaze / colour, always adding up to 1); Top K; the true Glazy recipe ID if you know it.

| preset | use it for |
|---|---|
| Balanced | the same glaze on a similar object (default) |
| Shape | the same kind of object |
| Glaze / surface | the same glaze look, any colour shift |
| Colour | anything in these colours |

## Results and evaluation

Every search shows the ranked objects / recipes with photos, recipe data and contribution bars, plus an evaluation
that never changes the order: per criterion (shape, glaze / surface, colour, strict) **Hit@K**, **Precision@K**,
matching objects, **Recall@K** (true recipe) and a weighted **nDCG**. *Session evaluation* adds up all searches
(Hit Rate@K, Precision@K, real Recall@K) and can be downloaded as CSV.

## Files it writes

| where (in `multimodal_experiments\`) | what |
|---|---|
| `work\queries\<upload id>\` | your uploaded photo, its objects, overlays and every search |
| `work\queries\session_log.json` | the session evaluation |
| `work\queries\glaze_search_log.jsonl` | every card G search (query, model version, result IDs) |
| `work\logs\app.log` | the server log |

Your uploads stay on your computer (they are not part of the repository).

## Online version

The same page runs on Hugging Face's free GPU: https://huggingface.co/spaces/FBIWarninggg/glaze_search (log in to
Hugging Face for a larger daily GPU allowance).

## Problems

| message / symptom | fix |
|---|---|
| `Glaze search is already running` | it is open already: http://127.0.0.1:8770 (one server at a time: each one needs ~4 GB of GPU memory) |
| `Another GPU job is running` | close the other GPU program (only one GPU job at a time) |
| `ERROR: rebuilt embeddings differ …` / file not found in `work\` | data incomplete: `DOWNLOAD_DATA.bat`, then `CHECK_DATA.bat` |
| model not found / tries to go online | run `SETUP.bat` again (models are missing) |
| CUDA out of memory | close other GPU programs; or run on the CPU: `set CUDA_VISIBLE_DEVICES=-1` before `RUN_GLAZE_SEARCH.bat` |
| card G: `the glaze recipe model is not available` / `smart search is not available` | run `INSTALL_GLAZE_MODEL.bat` (drag the model ZIP onto it), then restart `RUN_GLAZE_SEARCH.bat` |
| card G: no recipe names, cones without the leading 0 | `7k_matched_all_data.csv` not found: put it in `<CG2>\` or `<CG2>\new image data\csv 7k data\`, or set `CG2_7K_CSV_DIR` |
| card G: grey "photo unavailable" boxes | the photos come from Glazy's server: needs internet (the search itself works offline) |
