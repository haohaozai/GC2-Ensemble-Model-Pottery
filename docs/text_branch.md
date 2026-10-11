# Text branch: recipe tags → text encoder → projection head → shared space

Status (2026-10-11):

| Piece | State |
|---|---|
| Cleaned recipe data + data-quality report | ✅ working on `data/glazy_20260531.csv` |
| Deterministic descriptions | ✅ |
| Pretrained text embeddings (frozen encoder) | ✅ 35,461 × 384, cached, manifest |
| Text-only retrieval baseline + filters + name lookup | ✅ working (quality limits below) |
| Projection head + checkpoint format + joint-training interface | ✅ ready, **not trained** |
| Shared-space (cross-modal) retrieval | ⏳ needs a trained checkpoint from joint training |

There is **no trained projection yet**. A randomly initialised head is 128-d, like the
other branches, but it is not aligned with them. The code refuses to use an untrained head
for shared-space search.

## Setup

```bash
python -m venv .venv            # Python 3.10+
.venv/Scripts/pip install -r requirements-text.txt   # Windows; use .venv/bin/pip elsewhere
```

For GPU, install a CUDA build of torch first. The device is chosen automatically
(cuda → mps → cpu), or set it with `--device`. The first run downloads the encoder
(~90 MB) into the Hugging Face cache. Later runs load the pinned revision offline.

## Commands

Run these from the repo root. Every command accepts `--config` and `--output-dir`.

```bash
python -m text_branch preprocess        # CSV -> recipes_clean.jsonl + data_quality_report.{md,json}
python -m text_branch embed             # -> baseline/{embeddings.npy,id_map.csv,manifest.json}
python -m text_branch search "Shiny transparent celadon glaze" --top-k 10 \
    --cone 10 --atmosphere reduction --surface glossy --transparency transparent
python -m text_branch lookup "Floating Green"       # exact / normalised name + alias lookup
python -m text_branch demo                          # -> demo_results.md
python -m text_branch project --checkpoint path/to/trained_head.pt   # -> projected/
python -m text_branch search "..." --checkpoint path/to/trained_head.pt  # shared-space search

python -m pytest                                    # unit tests, no model download
TEXT_BRANCH_MODEL_TESTS=1 python -m pytest          # + real-model smoke tests
```

All outputs go to `outputs/text_branch/`, which is git-ignored. Nothing writes to `data/`.

| File | Contents |
|---|---|
| `recipes_clean.jsonl` | One line per source row: `recipe_id`, `raw`, `normalised`, `metadata`, `flags`, `description`, `text_available`, `unavailable_reason` |
| `data_quality_report.md/.json` | Missing fields, unrecognised/malformed values, review flags, unusable records |
| `preprocess_manifest.json` | Source sha256, preprocessing version, cleaning-config hash |
| `baseline/embeddings.npy` | float32 `[35461, 384]`, L2-normalised, **pretrained encoder space** |
| `baseline/id_map.csv` | `row, recipe_id, description_sha256, description`. Row *i* of the matrix belongs to that recipe |
| `baseline/manifest.json` | Model ID, requested/resolved revision, dim, prompts, library versions, preprocessing version, file hashes |
| `projected/` | Same layout, produced only from a **trained** checkpoint (`space: "shared"`) |
| `projected_UNTRAINED_DEV_ONLY/` | Only with `project --allow-untrained`. The manifest says `trained: false` |
| `cache/` | Embedding cache keyed by encoder fingerprint + description sha256 |

## Which fields affect the embedding

Source columns are mapped in `configs/text_branch.yaml` (`cleaning.columns`).

| Glazy column | Role | Normalised as |
|---|---|---|
| `Subtype` | **embedded** | hierarchy split on ` - ` (e.g. `glaze / iron / celadon`). Commas inside a level are kept |
| `Surface` | **embedded** | one of 9 labels (`glossy` … `stony matte`) |
| `Transparency` | **embedded** | one of 4 labels |
| `Cone` | **embedded** | string cone range on the Orton order (`022` … `14`) |
| `Atmospheres` | **embedded** | set of 7 labels, split on `;`, in canonical order |
| `ID` | metadata (row key) | string |
| `Name`, `Other Names` | metadata. Used for name/alias lookup only | `Other Names` split on `,`/`;` for lookup |
| `Country`, `State`, `Type` | metadata | `Type` decides scope (`Recipe` only) |

Creator, dates, ingredients, oxides and free-text descriptions are never read. Country can be
appended to descriptions with `description_metadata_fields: [country]`. It is off by default
and never counts as a semantic tag.

Description format (fixed field order; missing fields are left out):

```
Subtype: iron celadon glaze. Surface: glossy. Transparency: transparent. Firing cone: 9 to 10. Atmosphere: reduction.
```

### Cleaning rules

- Whitespace is collapsed, Unicode is NFC-normalised and labels are lower-cased. Synonyms come only from the explicit map in the YAML (e.g. `semi gloss → semi-glossy`, `lustre → luster`). Every mapping is recorded as an `info` flag.
- **Cones stay strings**: `06` (≈1000 °C) ≠ `6` (≈1220 °C). Ordering uses an explicit list, never numbers. `05 1/2` sits between `06` and `05`.
- **`error` flag**: the value cannot be used and becomes unknown (`None`). The raw value is kept. Examples: unrecognised label, malformed or reversed cone range (`10 - 9` is flagged, not swapped).
- **`warning` flag**: the value is kept but needs review. Examples: contradictions such as subtype *clear* + transparency *opaque*, duplicated atmospheres, unknown subtype root.
- Nothing is inferred from names: a recipe called "Celadon Food Safe" with no tags stays tagless.
- Records with no usable semantic tag get `unavailable_reason = no_usable_semantic_tags` and are never embedded. Records of `Type` Material or Analysis get `out_of_scope_type`.

Results on the current export (see `data_quality_report.md`):
- 43,083 rows; 35,478 recipes in scope; 35,461 embedded; 11,393 unique descriptions.
- Flagged: 69 reversed cone ranges, 73 *clear* glazes tagged opaque, 18 matte/glossy subtype–surface conflicts and 1 duplicated atmosphere.
- 17 recipes have no usable tags.
- 2,308 recipes have only `Subtype: glaze.` These are low-information records that tie exactly.

## Encoder

- `sentence-transformers/all-MiniLM-L6-v2`, pinned to commit `1110a243…`, **384-d**, frozen. It runs in eval mode with `requires_grad=False`.
- Queries go through `encode_query` and recipes through `encode_document`, so each model's own prompt conventions apply. MiniLM is symmetric, so both prompts are empty. Override them with `encoder.query_prompt` / `document_prompt` for models that document a prefix without configuring it.
- Output is float32 and L2-normalised; cosine similarity is the dot product.
- Reproducibility: the same batch size gives bit-identical vectors (verified). A different batch size gives differences of at most 1.2e-7 from padding.
- Cache invalidation: model, revision, prompts, max sequence length or sentence-transformers minor version → new cache file. An edited description → a cache miss for that description only.
- To change the model, set `encoder.model_name` and `revision` (`null` or a new commit), then re-run `embed`. Search refuses to mix vectors from different encoders.

## Search semantics

- Recipes are ranked by cosine similarity between the pretrained query and recipe embeddings.
- Filters are applied **before** the top-K cut.
- Filters are explicit. Free-text filter extraction is not implemented. Each filter field matches if **any** of its listed values matches:
  - `cone`: the requested cone or range must lie **within** the recipe's range. `9 - 11` matches cone 10; `6` does not match `06`. Use `--cone-match overlap` to accept partial overlap.
  - `atmospheres`: the recipe lists at least one requested atmosphere.
  - `surfaces` / `transparencies`: exact canonical label. Synonyms are accepted. Unknown labels raise an error instead of silently returning nothing.
- **Unknown values are excluded from strict filters by default.** A recipe with no surface tag never matches `--surface glossy`. `--include-unknown` keeps these recipes.
- Exact ties (identical descriptions) are broken by source order (recipe ID), which says nothing about relevance.
- Results carry the recipe ID, score, description, name, tags, state, country, other names and review flags.

## Baseline quality: what the demo shows

`outputs/text_branch/demo_results.md` is a **smoke test, not an evaluation**. The queries and tag
checks are synthetic, and no manually reviewed relevance labels exist yet.

| Query | Filters | Top-5 matching every hand-mapped tag | First full match |
|---|---|---|---|
| Shiny transparent celadon glaze | — | 0/5 (all celadons, mixed surface/transparency) | rank 17 |
| Shiny transparent celadon glaze | cone 10, reduction | 0/5 | rank 15 |
| Shiny transparent celadon glaze | + surface glossy, transparency transparent | **5/5** | rank 1 |
| Matte opaque glaze | — / cone 6 + oxidation | 3/5 | rank 1 |
| Glossy transparent glaze | cone 06 | 1/5 (a matte, opaque kiln wash ranks 1st) | rank 4 |
| Glossy transparent glaze | cone 6 | 0/5 (crystalline glazes dominate) | rank 97 |

Why it fails: the pretrained encoder barely separates antonym attributes, and subtype nouns
dominate. A controlled probe with the query "Glossy transparent glaze." scored the
*glossy + opaque* description (0.750) **above** *glossy + transparent* (0.743). Swapping to
*matte + opaque* lowered the score by only ~0.01. "Crystalline" (close to "transparent" in
everyday English, but a different glaze effect) raised it to 0.784. Longer descriptions also
score lower against short queries. Until joint training or a better encoder helps,
**pass hard constraints (surface, transparency, cone, atmosphere) as filters**, which
are exact.

## Projection head and joint training

`TextProjectionHead(input_dim, output_dim=128, hidden_dim=None, dropout=0.0)` is a linear
layer (or 2-layer MLP if `hidden_dim` is set) followed by L2 normalisation. `input_dim` comes
from the encoder (`build_projection_head(encoder_dim, ProjectionConfig)`). The shared dimension
is `projection.output_dim`, with 128 as the initial default.

The encoder is frozen, so joint training should consume the precomputed baseline vectors:

```python
import torch
import torch.nn.functional as F
from text_branch import (TextEmbeddingStore, build_projection_head, save_projection_checkpoint)
from text_branch.config import ProjectionConfig

store = TextEmbeddingStore.from_directory("outputs/text_branch/baseline")   # verifies hashes
text_head = build_projection_head(store.dim, ProjectionConfig(output_dim=128))
opt = torch.optim.AdamW(text_head.parameters(), lr=1e-3)                     # head params only

for batch in loader:                       # recipe IDs that have text AND the other modality
    ids = [i for i in batch.recipe_ids if i in store]
    t = text_head(store.lookup(ids, device))          # [B, 128] unit vectors; grads flow to the head
    v = image_branch(batch.images_for(ids))           # [B, 128] unit vectors (other team)
    logits = t @ v.T / temperature
    same = store.same_text_mask(ids).to(device)       # identical descriptions
    eye = torch.eye(len(ids), dtype=torch.bool, device=device)
    logits = logits.masked_fill(same & ~eye, float("-inf"))   # drop unreliable negatives
    target = torch.arange(len(ids), device=device)            # same recipe = positive
    loss = (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2
    ...

save_projection_checkpoint("checkpoints/text_head.pt", text_head, encoder=store.encoder,
                           trained=True, training={"run_id": ..., "modalities": ["text", "image"],
                                                   "objective": "symmetric InfoNCE", "epochs": ...,
                                                   "split": ..., "metrics": {...}})
```

**Positives and negatives.** The same recipe in different modalities is a reliable positive
pair. Other recipes in the batch are **not** automatically reliable negatives. Many recipes
share an identical description (2,308 are just `Subtype: glaze.`), and different recipes can
look or behave alike. `same_text_mask` marks the identical-text cases. Teams may also want to
mask pairs with near-identical formulas or images. Recipes without text (`id in store` is
False) should be skipped for the text loss, not given an empty description.

**Providing a trained checkpoint.** `save_projection_checkpoint(..., trained=True, training={...})`
stores the architecture, weights, encoder metadata (model, revision, dim, prompts) and
preprocessing version. A non-empty `training` dict is required. Loading
(`load_projection_checkpoint`) rejects:
- a different encoder, revision, prompt or dim;
- a different preprocessing version;
- weights that do not fit the stored architecture;
- `trained: false`, unless `require_trained=False` is passed for development.

Shared-space search and `project` refuse untrained checkpoints. Recipes and queries are always
projected with the same checkpoint.

Python API (all in `text_branch`): `normalise_recipe_tags(row)`, `build_recipe_description(tags)`,
`encode_recipes(recipes, encoder)`, `encode_query(text, encoder, projection=None)`,
`project_embeddings(embeddings, checkpoint)`, `search_recipes(query, top_k, filters, index=...)`,
`RecipeSearchIndex.lookup_name(name)`.

## Still pending (integration)

1. Joint contrastive training with the image and formula branches, which produces the first trained checkpoint. Agree on the shared dimension (128 is only a default) and the train/validation split by recipe ID.
2. A manually labelled relevance set (query → relevant recipe IDs). Retrieval quality is unmeasured until it exists.
3. Optional: try other encoders, a natural-language description template, or free-text → filter extraction. Each change needs re-running `preprocess`/`embed` and, for the template, bumping `PREPROCESSING_VERSION`.
4. Review the flagged records in `data_quality_report.md` with someone who knows the data. Nothing has been auto-corrected.
