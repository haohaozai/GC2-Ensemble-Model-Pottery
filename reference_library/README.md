# Reference library (static, convention only)

One folder per attribute value: `<attribute>/<value_slug>/`

| file | content |
|---|---|
| `describe.md` | definition, identifying features, recognition rule, values not to confuse with, sources (readable) |
| `describe.json` | the same for the apps (Glaze Search (i) window reads `/lib/<attribute>/<value_slug>/describe.json`) |
| `reference_1.png` | photo from your manual (Ceramic Attribute Taxonomy, craze documentaries.pdf), page in the caption |
| `example_1.png`, `example_2.png` | real database objects whose value Claude confirmed (highest model probability, one per recipe); orange box = where Grounding DINO found the value |

`index.json` lists every attribute -> value -> folder. Texts come from your manuals; text marked *general description* is not
defined there. Rebuild: `.venv\Scripts\python.exe multimodal_experiments\build_reference_library.py` (does not change any search
or label - static reference only).
