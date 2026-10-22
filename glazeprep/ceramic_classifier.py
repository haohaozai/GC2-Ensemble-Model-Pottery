"""Zero-shot ceramic attribute candidate tagger. Similarity scores are NOT probabilities."""
import numpy as np
import torch
import open_clip
from PIL import Image

# Independent attribute groups from the ceramic taxonomy; avoid inventing defect labels.
LABELS = {
    'texture': {
        'smooth':'a ceramic surface with smooth even texture',
        'rough granular':'a ceramic surface with visible rough grains',
        'dimpled':'a ceramic glaze with small shallow dimples',
        'ribbed grooved':'a ceramic surface with repeated raised ribs and grooves',
        'carved incised':'a ceramic surface with engraved carved lines',
        'pitted':'a ceramic surface with many tiny pits',
        'cratered':'a ceramic surface with crater shaped depressions',
        'wrinkled':'a ceramic glaze with wrinkled texture',
        'bubbled':'a ceramic glaze with raised bubbles',
        'pebbled':'a ceramic surface with pebble like texture',
    },
    'glaze_effect': {
        'crystalline':'crystalline ceramic glaze with starburst crystals',
        'oil spot':'ceramic glaze with round metallic oil spots',
        'speckled':'ceramic glaze with small contrasting speckles',
        'mottled':'ceramic glaze with irregular patches of mixed colors',
        'dripping':'ceramic glaze with vertical running drips',
        'crackle':'decorative ceramic glaze with interconnected crackle lines',
        'breaking':'ceramic glaze changing color along raised edges',
        'metallic':'ceramic glaze with metallic sheen',
        'iridescent':'ceramic glaze with rainbow iridescent sheen',
        'marbled':'ceramic glaze with swirling marble patterns',
        'layered':'ceramic glaze with visible overlapping glaze layers',
        'no obvious glaze effect':'plain ceramic glaze without a distinctive special effect',
    },
    'defect': {
        'pinholes':'ceramic glaze with small pinhole defects',
        'blistering':'ceramic glaze with burst blisters and raised bubbles',
        'crawling':'ceramic glaze pulled into islands exposing bare clay',
        'crazing':'ceramic glaze with fine unintentional network cracks',
        'shivering':'ceramic glaze flaking away at an edge',
        'pitting':'ceramic glaze with small crater like defects',
        'clouding':'ceramic glaze with cloudy uneven opacity',
        'uneven coverage':'ceramic glaze with visibly uneven coverage',
        'no obvious visible defect':'ceramic glaze without an obvious visible defect',
    },
    'decoration': {
        'floral':'ceramic pottery decorated with painted flowers',
        'geometric':'ceramic pottery decorated with geometric patterns',
        'abstract':'ceramic pottery decorated with abstract painted art',
        'striped':'ceramic pottery decorated with stripes',
        'landscape':'ceramic pottery decorated with a landscape scene',
        'plain':'plain undecorated ceramic pottery',
        'animal':'ceramic pottery decorated with animal images',
        'botanical':'ceramic pottery decorated with plants and leaves',
        'calligraphy':'ceramic pottery decorated with calligraphic writing',
        'figurative':'ceramic pottery decorated with human figures',
        'dotted':'ceramic pottery decorated with regular dotted motifs',
        'repeating motif':'ceramic pottery decorated with repeated motifs',
        'mixed decoration':'ceramic pottery with mixed decorative patterns',
    },
    'form': {
        'bowl':'a photograph of a ceramic bowl',
        'vase':'a photograph of a ceramic vase',
        'mug':'a photograph of a ceramic mug with a handle',
        'plate':'a photograph of a ceramic plate',
        'teapot':'a photograph of a ceramic teapot',
        'test tile':'a photograph of a rectangular ceramic glaze test tile',
        'cup':'a photograph of a ceramic cup',
        'jar':'a photograph of a ceramic jar',
        'pitcher':'a photograph of a ceramic pitcher',
        'planter':'a photograph of a ceramic planter',
        'sculpture':'a photograph of a ceramic sculpture',
        'tile':'a photograph of a flat ceramic tile',
        'dish':'a photograph of a ceramic shallow dish',
        'bottle':'a photograph of a ceramic bottle',
        'vessel':'a photograph of a ceramic vessel',
    },
}

class CeramicClassifier:
    def __init__(self, model_name='ViT-B-32', pretrained='laion2b_s34b_b79k'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'  Loading OpenCLIP {model_name} on {self.device}...', flush=True)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device)
        self.model.eval()
        tokenizer = open_clip.get_tokenizer(model_name)
        self.keys = {}
        self.embeddings = {}
        with torch.inference_mode():
            for group, descriptions in LABELS.items():
                self.keys[group] = list(descriptions)
                texts = [f'A close-up photograph of {description}.' for description in descriptions.values()]
                tokens = tokenizer(texts).to(self.device)
                emb = self.model.encode_text(tokens)
                self.embeddings[group] = torch.nn.functional.normalize(emb.float(), dim=-1)
        print('  Visual tagger ready.', flush=True)

    def predict(self, image: Image.Image):
        # This is a zero-shot candidate ranker; ambiguous or subtle defects need review.
        tensor = self.preprocess(image.convert('RGB')).unsqueeze(0).to(self.device)
        result = {'scores': {}, 'needs_review': True}
        with torch.inference_mode():
            emb = torch.nn.functional.normalize(self.model.encode_image(tensor).float(), dim=-1)
            for group, labels in self.keys.items():
                similarities = (emb @ self.embeddings[group].T)[0].cpu().numpy()
                order = np.argsort(-similarities)
                first, second = int(order[0]), int(order[1])
                margin = float(similarities[first] - similarities[second])
                # Small ranking margin means the two best descriptions are nearly tied.
                result[group] = labels[first] if margin >= 0.015 else 'Needs review'
                result['scores'][group] = {
                    'top_label': labels[first],
                    'top_similarity': round(float(similarities[first]), 4),
                    'runner_up': labels[second],
                    'margin': round(margin, 4),
                }
        return result
