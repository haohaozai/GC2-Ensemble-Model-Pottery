import re
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent


def standardize_cone(value):
    """Normalize cone fractions and leading zeros, retaining ranges and missing values."""
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if not text:
        return np.nan

    def replace_number(match):
        whole = int(match.group(1))
        numerator = match.group(2)
        denominator = match.group(3)
        number = whole + int(numerator) / int(denominator) if numerator else whole
        return f'{number:g}'

    return re.sub(r'(?<![\w.])(\d+)(?:\s+(\d+)\s*/\s*(\d+))?(?![\w./])', replace_number, text)


# Load and merge the CSV files.
df1 = pd.read_csv(BASE_DIR / '7k_matched_all_data.csv')
df2 = pd.read_csv(BASE_DIR / 'all_combined.csv')
df2_filtered = df2[df2['object_index'] == 1]
merged_df = pd.merge(df1, df2_filtered, on='recipe_id', how='inner')

# Keep glaze recipes only.
glaze_mask = merged_df['Subtype'].astype(str).str.contains('glaze', case=False, na=False)
merged_df = merged_df[glaze_mask].copy()

# Clear glazes have a clear color name and no hex color.
clear_mask = merged_df['Subtype'].astype(str).str.contains('clear', case=False, na=False)
merged_df.loc[clear_mask, 'color1_name'] = 'clear'
merged_df.loc[clear_mask, 'color1_hex'] = np.nan

# Rename output fields and normalize cone notation.
merged_df = merged_df.rename(columns={
    'color1_name': 'color_name',
    'color1_hex': 'hex_color',
})
merged_df['Cone'] = merged_df['Cone'].apply(standardize_cone)

# Join the image filenames by recipe_id and build the CDN image paths.
paths_df = pd.read_csv(BASE_DIR / 'recipe_id_paths.csv', dtype={'recipe_id': 'int64', 'filename': 'string'})
if paths_df['recipe_id'].duplicated().any():
    raise ValueError('Duplicate recipe_id values found in recipe_id_paths.csv')
merged_df = merged_df.merge(paths_df[['recipe_id', 'filename']], on='recipe_id', how='left', validate='many_to_one')
base_url = 'https://cdn.glazy.org/public/uploads/recipes/'
merged_df['image_path'] = (
    base_url + merged_df['recipe_id'].astype(str).str[-2:] + '/' + merged_df['filename']
)
missing_paths = merged_df['image_path'].isna().sum()
print(f'Number of rows missing an image path: {missing_paths}')

columns_to_keep = [
    'recipe_id', 'image_path', 'color_name', 'hex_color', 'Cone', 'Surface', 'Transparency', 'Atmospheres',
    'SiO2', 'Al2O3', 'CaO', 'K2O', 'Na2O', 'Fe2O3', 'MgO', 'TiO2', 'B2O3',
    'P2O5', 'ZnO', 'CuO', 'CoO', 'SrO', 'Li2O', 'ZrO2', 'SnO2', 'BaO',
    'Cr2O3', 'MnO', 'MnO2', 'FeO', 'NiO', 'PbO', 'F', 'V2O5'
]
result_df = merged_df[columns_to_keep].copy()

missing_colors = result_df['color_name'].isna().sum()
print(f'Number of rows missing a color name: {missing_colors}')
if missing_colors > 0:
    missing_dir = BASE_DIR / 'missing_images'
    missing_dir.mkdir(parents=True, exist_ok=True)
    result_df.loc[result_df['color_name'].isna(), ['recipe_id']].to_csv(
        missing_dir / 'missing_rows.csv', index=False
    )

print(f'Total of: {len(result_df)} rows')
output_path = BASE_DIR / 'merged_output.csv'
result_df.to_csv(output_path, index=False)
print(f'Merge, update, and checks complete! Saved to {output_path}')
