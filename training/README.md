# Training Data
`training.ipynb:` Notebook training the shared latent space.

Includes training the:    
**Oxide stream:** Each glaze is represented by 26 oxide composition features. These are normalized into relative proportions, square-root transformed, and reduced to 16 principal components using Principal Component Analysis (PCA). The resulting features are passed through a trainable linear projection head to produce a 256-dimensional embedding.     
**Text stream:** Color name, surface, firing cone, transparency, and firing atmosphere, are combined into a single text string.   
```
f"color: {get('color_name')}, "   
         f"Cone {get('Cone')}, {get('Surface')}, "   
         f"{get('Transparency')}, {get('Atmospheres')}")
``` 
A pretrained MiniLM Transformer converts this description into a 384-dimensional embedding. A separate trainable linear projection head maps this embedding into the same 256-dimensional space.   
Both projection heads apply L2 normalization, allowing the similarity between text and chemistry embeddings to be measured using cosine similarity.   

The two projection heads are trained using a symmetric InfoNCE (Information Noise Contrastive Estimation) loss. The objective encourages embeddings from matching text-chemistry pairs to have higher cosine similarity than non-matching pairs within a training batch.
The loss is calculated in both directions: text-to-chemistry and chemistry-to-text. This encourages bidirectional alignment between the two modalities.

**Output:** `shared_space_heads.pt` (projection heads), `glaze_retrieval_vectors.npz` (recipe embedding vectors).
You can find these in [glaze_retrieval/retrieval_assets](../glaze_retrieval/retrieval_assets)

# Dataset creation
`/dataset_creation` holds the scripts and intermediate CSVs to generate `training_data.csv`.    
`merge_csv_updated.py`: python script to join all the data together and perform additional filtering and data processing/cleaning.   
`merged_output.csv`: Resulting dataset from `merge_csv_updated.py`. Note that this is the same as `training_data.csv`, it has just been renamed for clarity.
`7k_matched_all_data.csv`: row-level data for each recipe, filtered to the ~7,000 recipes with a photo.     
`all_combined.csv`: the extracted color-name and hex-code for the ~7,000 images after doing one-pass through the image segmentation and color extraction pipeline.     
`recipe_id_paths.csv`: Original image paths for the thumbnail images.



