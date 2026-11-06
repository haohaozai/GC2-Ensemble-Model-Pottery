# Customize the demo

Edit `demo_content.json` and restart Streamlit (or use Rerun).

- `title`: browser tab and page heading, e.g. `Ceramic Glaze Explorer`.
- `font`: one of `System default`, `Arial`, `Helvetica`, `Georgia`, `Verdana`, `Trebuchet MS`, `Times New Roman`, `Courier New`. Fonts are browser/system fonts and may fall back depending on the viewer.
- `intro`: Markdown text below the title. Use a JSON string with `\n\n` for paragraph breaks, or a JSON array of paragraphs. Supports **bold**, *italic*, links, headings and lists. Avoid raw HTML and untrusted content.

Example:

```json
"intro": [
  "Find ceramic glaze recipes from a photo.",
  "**Instructions:** Upload a photo or choose a sample image.",
  "### About this demo\nThis search uses colour and recipe metadata."
]
```

The intro uses `st.markdown` instead of `st.caption` so multiline formatting is retained.

**Icon fix:** The app now avoids applying the chosen font to every nested element. Streamlit's Material icon glyphs (the expander arrow and uploader icon) keep their own font instead of appearing as overlapping words such as `arrow_down` or `add`.

- `samples`: map image filenames inside `sample_images/` to descriptions.

The UI shows 3–8 results in a four-column grid, with at most two rows. Ranking scores are hidden in the UI but remain in retrieval and CSV export.


## Button colour and recipe details

Edit `demo_content.json`:

```json
"button_color": "#2E6D64",
"recipe_title_field": "Name",
"recipe_fields": [
  {"field": "recipe_id", "label": "Recipe ID"},
  {"field": "Cone", "label": "Firing cone"},
  {"field": "Surface", "label": "Surface"}
]
```

`button_color` must be a six-digit hex colour. `recipe_title_field` controls the heading below each thumbnail. `recipe_fields` controls the details shown below it, in order; remove entries to hide fields, add entries to show additional catalog columns. The `field` values must match the **exact column names** in the retrieval results (see Export CSV). `label` can be any display text. Missing or empty values are skipped. The recipe link remains displayed regardless of these settings. No similarity percentage is shown by default.

The model diagnostics section is now a full-width collapsible expander instead of a popover, to prevent label/icon overlap.

## Diversity toggle text

Change the explanatory caption and the `?` tooltip without editing Python:

```json
"diversity_caption": "Broader variety: at most 2 recipes per cluster.",
"diversity_help": "Limits each K-means cluster to two recipes where possible, showing a wider variety of glaze groups. Ranking scores are unchanged."
```

These settings change **display text only**. The retrieval logic still limits diversified results to two recipes per cluster. Restart or rerun Streamlit after editing the JSON.
