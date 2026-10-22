# Glaze Search (`RUN_GLAZE_SEARCH.bat`)

Find glaze recipes and ceramic objects that look like your photo, match your words or your glaze formula. The search
runs on your computer: http://127.0.0.1:8770 (opens by itself). Keep the black window open while you search.

Step-by-step guide with screenshots: [manuals/CG2_Glaze_Search_Beginner_Guide.pdf](manuals/CG2_Glaze_Search_Beginner_Guide.pdf);
how everything fits together: [manuals/CG2_Flow_MindMap.pdf](manuals/CG2_Flow_MindMap.pdf).

## What happens when it starts (about 30 - 60 s)

The server loads the retrieval model and the database of **7,063 Glazy recipes / 12,309 segmented objects**, the
segmentation models (Grounding DINO + SAM 2.1 + OpenCLIP), DINOv2 and the six attribute models. With an NVIDIA GPU
they use about 4 GB of graphics memory; without a GPU everything runs on the CPU (a new photo then takes 1 - 3 min,
searches stay fast). Everything works offline after `SETUP.bat`.

## The page

1. **Card 1 · Photo**: drop a photo, paste it (`Ctrl+V`), browse, or paste a link. Grounding DINO + SAM 2.1 cut out
   the ceramic objects (same settings as the 7k database photos). Hover an object to see it, **click** = use it
   (yellow outline = active), `Ctrl`+click = add more. **Pick colour (HEX)**: click inside an object, then
   **Add as search colour**.
2. **Card 1b · Your object**: the colour regions of the active object and the six attributes Grounding DINO detects,
   each with its top-3 candidates; click the right one, **(i)** explains that value with example images
   (`reference_library`). Tick the attributes the results must share.
3. **Card 2 · Text**: subtype, firing cone, surface, transparency, atmosphere, free notes.
4. **Card 3 · Formula**: oxides (weight % or UMF), optional ingredients.
5. **Card 4 · Colours**: colour words (Color Lab V3: celadon, crimson, eggshell …), HEX codes or phrases.
6. **Search bar**: search *objects* (the segmented database objects) or *recipes*; a preset or your own priority with
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
| `ERROR: rebuilt embeddings differ …` / file not found in `work\` | data incomplete: `DOWNLOAD_DATA_ORIGINAL_AND_SEGMENTATION.bat`, then `CHECK_DATA.bat` |
| model not found / tries to go online | run `SETUP.bat` again (models are missing) |
| CUDA out of memory | close other GPU programs; or run on the CPU: `set CUDA_VISIBLE_DEVICES=-1` before `RUN_GLAZE_SEARCH.bat` |
