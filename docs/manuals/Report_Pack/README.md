# Glaze Search · Report Pack

Every current Glaze Search document in one folder, numbered in the order a report usually needs them: overview → method →
worked examples → evaluation → decision. The files are copies; the originals stay in [`docs/manuals`](..) (the app's
📖 How to use page links to them). Pack made on 7 November 2026.

## The documents

| # | File | Pages | What it covers | Use it for | Made |
|---|---|---|---|---|---|
| 01 | [Quick Start](01_Quick_Start.pdf) | 14 | Install in five steps, then one photo followed click by click to the recipe list and to similar pots; real screenshots | Introduction, system walkthrough | 6 Nov |
| 02 | [User Manual](02_User_Manual.pdf) | 29 | All six pages of the app, six worked examples (photo, words with typos, a formula), what every number means | Appendix: user guide | 5 Nov |
| 03 | [Process, models and parameters](03_Process_Models_Parameters_MindMap.pdf) | 8 | Data → preprocessing → training → evaluation; the 10 steps of a search with their settings | Method overview, system diagram | 4 Nov |
| 04 | [Pretrained models, illustrated](04_Pretrained_Models_Illustrated.pdf) | 16 | Grounding DINO, SAM 2.1, OpenCLIP, DINOv2, MiniLM: zero-shot, frozen or with a trained head; activity diagram and 13 figures from one photo | Method: models and algorithms | 5 Nov |
| 05 | [Photo attributes report](05_Photo_Attributes_Report.pdf) | 4 | The six photo attributes (shape, surface, transparency, texture, effects, defects) and their tests | Method + results: attributes | Oct |
| 06 | [System flow mind map](06_System_Flow_MindMap_earlier_layout.pdf) | 5 | The whole system as a mind map (earlier page layout) | Overview figure (check page names against 01) | Oct |
| 07 | [Example: glaze recipes from a bowl photo](07_Example_Glaze_Recipes_Bowl_Photo.pdf) | 6 | One bowl photo + one typed sentence through every step and engine, the pretrained model of each step, the team's parts coloured | Worked example: recipe search | 6 Nov |
| 08 | [Example: glaze recipes, "sand satin" cases](08_Example_Glaze_Recipes_Sand_Cases.pdf) | 11 | Colour only → colour + menus → typed words → photo → formula → all; framed screenshots and the score of #1 worked out in every case | Worked example: the three engines and Auto | 7 Nov |
| 09 | [Example: similar pottery from a bowl photo](09_Example_Similar_Pottery_Bowl_Photo.pdf) | 12 | Colour only → photo (Balanced) → the three presets → + words → + formula; z-scores and shares worked out for #1 | Worked example: object search | 7 Nov |
| 10 | [Evaluation, all stages](10_Evaluation_All_Stages.pdf) | 13 | Every metric (R@K, P@10, MAD@K, accuracy, AUROC …) stage by stage, from the data and labels to the final ranking; the limits to state | Results, limitations | 5 Nov |
| 11 | [Model decision and engine evaluation](11_Model_Decision_and_Engine_Evaluation.pdf) | 7 | What `test_similarity (3).zip` contains; colour match vs smart search vs colour ranking vs two hybrids on 395 held-out recipes; the decision tree and the Auto engine | Results: engine comparison, design decision | 6 Nov |

## A report outline that uses them

1. **Introduction and system overview:** 01, 03, 06.
2. **Data and photo processing** (segmentation, colours, attributes): 04 part A, 05, 10 part A.
3. **Models:** 04 (pretrained models and trained heads), 03 (parameters), 11 section 1 (the team's colour-match model).
4. **Search engines:** 07 and 08 (recipes: colour ranking, colour match, smart search, Auto), 09 (similar pottery).
5. **Evaluation:** 10 (every stage), 11 (engine comparison and robustness).
6. **Discussion and limitations:** 10 ("limits to state"), 11 (latent-space report, hybrids), the notes in 08 and 09.
7. **Appendix:** 02 (user manual).

## Headline numbers (with their source)

| Result | Number | Source |
|---|---|---|
| Typed sentence with typos → right recipe in the top 10 (smart search) | 74.2 % (pretrained MiniLM alone 62.0 %; random 0.9 %) | 10 |
| Colour-match model: recipe text → its own chemistry in the top 10 | 8.2 % (random 0.8 %) | 10, 11 |
| Values read from typed words | cone 99.3 %, colour 93.3 %, surface 89.1 % | 10 |
| Segmentation upgrade on reviewer-flagged photos | 148 of 168 better, 2 worse | 10 |
| Photo attribute "form" | 83.6 % (most-common guess 55.5 %) | 10 |
| Engines on 395 held-out recipes, typed sentence: right recipe in the top 10 | smart search 79 % · colour match 43 % | 11 |
| Only a colour: results of the same colour in the top 10 | colour ranking 88 % · smart search 59 % | 11 |
| Colour + cone / surface menus: same colour AND surface in the top 10 | colour match 63 % · smart search 54 % | 11 |
| Hybrid (smart search + colour-match learned match) | best weight 0 → no gain | 11 |
| Colour-match space + a new image head, photo → right recipe in the top 10 | 27 % (smart search 45 %) | 11 |

## Changes after some of these documents were made

The app changed on 6–7 November 2026; documents made earlier do not show these changes yet:

| Change | Shown in | Not yet in |
|---|---|---|
| **Auto engine** (default on Glaze recipes): photo, words or formula → smart search; colour + cone / surface menus → colour match; only a colour → colour ranking. Decision file `configs/glaze_decision.json` | 08, 11 | 01, 02, 03, 04 (they describe the Smart search / Colour match buttons, which are still there) |
| Changed words clear the cone / surface they had set; a value picked by hand stays | 08 | 01, 02 |
| Formulas with subscripts (SiO₂, Al₂O₃) are read; a usable formula ticks "formula" automatically | 08 | 02 |
| The page's example colour (#3054A5) no longer joins a search that has its own words, photo or formula | 08 | 01, 02 |
| Similar pottery: clicking a cut-out opens the recipe's photo on Glazy (catalog `image_path`) | – | 01, 02 |

The models themselves did not change: colour match `70cd01c0c673`, smart search `78afd7dfbe2c`.

## How the newer documents were made

07, 08, 09 and 11 were built from real runs of the app on a test copy with its own logs; their screenshots are the real
pages and every number is the app's own output. 11 is reproducible with `python scripts/eval_glaze_engines.py`.
